import logging
from typing import Dict, Tuple

import clip
import numpy as np
import torch
from torch import Tensor, nn
from liv.models.model_liv import LIV

from lerobot.policies.gcr.configuration_grc import GCRConfig
from lerobot.policies.gcr.load_pretrained import load_liv
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import OBS_IMAGE, REWARD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
epsilon = 1e-8

CLIP_DATASET_STATS = {
    "mean": [[[0.48145466]], [[0.4578275]], [[0.40821073]]],  # (c,1,1)
    "std": [[[0.26862954]], [[0.26130258]], [[0.27577711]]],  # (c,1,1)
}


class GCR(PreTrainedPolicy):
    """Image classifier built on top of a pre-trained encoder."""

    name = "gcr"
    config_class = GCRConfig

    def __init__(
        self,
        config: GCRConfig,
    ):
        super().__init__(config)
        self.config = config

        if config.img_key is None:
            self.img_key = [key for key in self.config.input_features if key.startswith(OBS_IMAGE)][0]
        else:
            self.img_key = config.img_key

        if config.load_pretrained:
            self.model = load_liv().module
        else:
            self.model = LIV(
                modelid=config.model_id,
                device=config.device,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                langweight = config.lang_weight,
                visionweight = config.vision_weight,
                clipweight=config.clip_weight,
                gamma=config.discount,
                metric = config.metric,
                num_negatives=config.num_negatives,
                grad_text=True,
                scratch=False,
            )

        self.clip_loss_img = nn.CrossEntropyLoss()
        self.clip_loss_txt = nn.CrossEntropyLoss()

    def predict(self, img: Tensor, next_img: Tensor, goal: Tensor, modality: str = "vision", normalize: bool = False) -> torch.Tensor:
        """Forward pass of the classifier for inference."""
        # needs implementation
        img = self.normalize_img(img)
        next_img = self.normalize_img(next_img)
        
        if modality == "vision":
            goal = self.normalize_img(goal)
            
        e_img = self.model(img, modality="vision")
        e_next_img = self.model(next_img, modality="vision")
        e_goal = self.model(goal, modality=modality)
        
        v = self.model.sim(e_img, e_goal)
        v_next = self.model.sim(e_next_img, e_goal)
        
        reward = v_next - v
        if normalize and self.config.discount < 0.999:
            reward /= 1.0 - self.config.discount
            
        return reward
        
    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        output_dict = dict()

        b_im = batch[self.config.img_key]
        b_lang = batch[self.config.lang_key]
        b_reward = batch[REWARD]

        bs = b_im.shape[0]
        img_stack_size = b_im.shape[1]
        H = b_im.shape[-2]
        W = b_im.shape[-1]
        b_im_r = b_im.reshape(bs*img_stack_size, 3, H, W)

        # Encode visual and text inputs
        e_img = self.model(b_im_r, modality="vision")
        b_token = clip.tokenize(b_lang)
        e_lang = self.model(b_token, modality="text")

        e_img = e_img.reshape(bs, img_stack_size, -1)
        e0 = e_img[:, 0]  # initial, o_0
        eg = e_img[:, 1]  # final, o_g
        es0_vip = e_img[:, 2]  # o_t
        es1_vip = e_img[:, 3]  # o_t+1
        eg_img = e_img[:, -1]
        full_loss = 0

        # CLIP Loss
        if self.config.clip_weight != 0:
            clip_loss = self.compute_clip_loss(eg_img, e_lang)
            clip_loss = self.config.clip_weight * clip_loss
            output_dict['clip_loss'] = clip_loss.item()
            full_loss += clip_loss

        # VIP Loss (Visual)
        vip_loss_visual = self.compute_vip_loss(e0, es0_vip, es1_vip, eg, b_reward, self.config.num_negatives)
        output_dict['vip_loss_visual'] = vip_loss_visual.item()
        full_loss += self.config.vision_weight * vip_loss_visual

        # VIP Loss (Language)
        if self.config.lang_weight != 0:
            vip_loss_lang = self.compute_vip_loss(e0, es0_vip, es1_vip, e_lang, b_reward, self.config.num_negatives)
            output_dict['vip_loss_lang'] = vip_loss_lang.item()
            full_loss += self.config.lang_weight * vip_loss_lang

        output_dict['full_loss'] = full_loss.item()
        return full_loss, output_dict

    def compute_gcr_contrastive_loss(self, pos_goal_1, pos_goal_2, neg_goal):
        """
        Returns the scalar *mean* contrastive loss  L_P + L_N , ready to be added
        to the VIP loss.  All inputs must already be encoded via `model.sim`.
        """

        e_pos_1 = self.model(self.normalize_img(pos_goal_1), modality="vision")
        e_pos_2 = self.model(self.normalize_img(pos_goal_2), modality="vision")
        e_neg = self.model(self.normalize_img(neg_goal), modality="vision")

        sim_pos = self.model.module.sim(e_pos_1, e_pos_2)
        sim_neg = self.model.module.sim(e_pos_1, e_neg)

        if self.config.contrastive_mode.lower() == "sc":
            # Eq. 3:  −ω₁ S(g,g′) + ω₂ S(g,g_neg)
            loss = (-self.config.contrastive_omega1 * sim_pos + self.config.contrastive_omega2 * sim_neg).mean()

        elif self.config.contrastive_mode.lower() == "ic":
            # Eq. 4:  -log   exp{ω₁ S(g,g′)} / E[exp{ω₂ S(g,g_neg)}]
            logits_pos = self.config.contrastive_omega1 * sim_pos
            logits_neg = self.config.contrastive_omega2 * sim_neg

            # Monte-Carlo estimate of denominator (batch-wise expectation)
            log_den = torch.logsumexp(logits_neg, dim=0) - np.log(len(logits_neg))
            loss = -(logits_pos.mean() - log_den)  # negative log-prop

        return loss

    def compute_clip_loss(self, image_features, text_features):
        # A simplified function call of CLIP.forward()
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        ground_truth = torch.arange(len(image_features), dtype=torch.long, device=image_features.device)
        clip_loss = self.clip_loss_img(logits_per_image, ground_truth) + self.clip_loss_txt(logits_per_text, ground_truth)
        return clip_loss / 2

    def compute_vip_loss(self, e0, es0_vip, es1_vip, eg, b_reward, num_negatives=0):
        r = b_reward.to(e0.device)

        V_0 = self.model.sim(e0, eg)
        V_s = self.model.sim(es0_vip, eg)
        V_s_next = self.model.sim(es1_vip, eg)

        # Rescale Value
        V_0 = V_0 / (1 - self.config.discount)
        V_s = V_s / (1 - self.config.discount)
        V_s_next = V_s_next / (1 - self.config.discount)

        # Compute VIP Loss
        V_loss = (1 - self.config.discount) * -V_0.mean() + torch.log(epsilon + torch.mean(torch.exp(-(r + self.config.discount * V_s_next - V_s))))

        # Optionally, add additional "negative" observations
        if num_negatives > 0:
            V_s_neg = []
            V_s_next_neg = []
            for _ in range(num_negatives):
                perm = torch.randperm(es0_vip.size()[0])
                es0_vip_shuf = es0_vip[perm]
                es1_vip_shuf = es1_vip[perm]

                V_s_neg.append(self.model.sim(es0_vip_shuf, eg))
                V_s_next_neg.append(self.model.sim(es1_vip_shuf, eg))

            V_s_neg = torch.cat(V_s_neg)
            V_s_next_neg = torch.cat(V_s_next_neg)
            r_neg = -torch.ones(V_s_neg.shape).to(V_0.device)
            V_s_neg = V_s_neg / (1 - self.config.discount)
            V_s_next_neg = V_s_next_neg / (1 - self.config.discount)
            V_loss = V_loss + torch.log(epsilon + torch.mean(torch.exp(-(r_neg + self.config.discount * V_s_next_neg - V_s_neg))))

        return V_loss

    def get_optim_params(self) -> dict:
        """Return optimizer parameters for the policy."""
        return [{
            "params": self.model.parameters(),
            "lr": getattr(self.config, "learning_rate", 1e-5),
            "weight_decay": getattr(self.config, "weight_decay", 0.001),
        }]

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        raise NotImplementedError("GCR does not select actions")

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not produce action chunks.
        """
        raise NotImplementedError("GCR does not predict action chunks")

    def reset(self):
        """
        This method is required by PreTrainedPolicy but not used for reward classifiers.
        The reward classifier is not an actor and does not select actions.
        """
        pass