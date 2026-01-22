import copy
from collections import deque
from typing import Any, Optional

import torch
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, XVLAModel
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_TOKENS
from .configuration_xpvla import XPVLAConfig
from .critics.backbone import CriticBackboneBundle, CriticBackboneConfig, CriticBackbone


# -----------------------------
# Critic algorithm interface (slot)
# -----------------------------

class CriticAlgorithm(nn.Module):
    """
    Plug-in critic algorithm API.
    We do NOT implement any head here yet; this is the contract the policy expects.

    Your future implementations:
      - IQL (H=1) baseline
      - V-Former-style IQL with chunks (H>1)
      - Distributional evaluation critic with true distributional Bellman backups (H>1)
    """

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def get_optim_params(self) -> dict:
        return {"params": [p for p in self.parameters() if p.requires_grad]}

    def reset(self) -> None:
        return

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
        """
        Returns (loss, logging_dict).
        Expected to consume:
          - batch["state"], batch["action"], batch["next_state"], batch["next_action"], batch["reward"], batch["done"]
        and optionally action masks / chunk_len / gamma^H depending on your final spec.
        """
        raise NotImplementedError

    @torch.no_grad()
    def scalar_q(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Returns a scalar Q estimate for advantage computation / label sanity checks.
        Not used by the policy extraction step (labels are precomputed offline by your label script),
        but useful for debugging and for future on-the-fly labeling variants.
        """
        raise NotImplementedError


def make_critic_algorithm(critic_type: str, critic_cfg: dict[str, Any]) -> CriticAlgorithm:
    """
    Factory stub. We will implement it once we write the critic heads.
    For now it returns a placeholder module.
    """
    # Placeholder: identity module so policy class can instantiate cleanly.
    class _NoopCritic(CriticAlgorithm):
        def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Any]]:
            # Always return zero loss in placeholder.
            loss = torch.zeros((), device=next(self.parameters()).device) if len(list(self.parameters())) else torch.zeros(())
            return loss, {"critic_placeholder": True}

        @torch.no_grad()
        def scalar_q(self, batch: dict[str, Tensor]) -> Tensor:
            # Return zeros shaped [B]
            # Attempt to infer batch size from action tensor.
            act = batch.get("action", None)
            if act is None:
                return torch.zeros((1,), device=loss.device if "loss" in locals() else "cpu")  # type: ignore
            B = act.shape[0]
            return torch.zeros((B,), device=act.device)

    return _NoopCritic(critic_cfg)


# -----------------------------
# Top-level CFGRL-AC policy
# -----------------------------

class XPVLA(PreTrainedPolicy):
    """
    Wrapper policy around XVLAPolicy that implements:
      - label dropout during policy extraction training by selecting uncond/cond tokens per sample
      - true CFG guided sampling at sampler level (per solver step), without modifying XVLA internals
      - optional reuse of image encoding across the uncond/cond branches
    """

    config_class = XPVLAConfig
    name = "xpvla"

    def __init__(self, config: XPVLAConfig, **kwargs):
        super().__init__(config)

        # Load backbone
        self.xvla: XVLAPolicy = XVLAPolicy.from_pretrained(
            config.xvla_pretrained_path,
            config=self.config.xvla,
            revision=config.xvla_revision,
        )

        self.critic_vlm = self.xvla.model.vlm.clone()
        self.reset()

    def reset(self) -> None:
        self._queues = {ACTION: deque(maxlen=self.xvla.config.n_action_steps)}

    def get_optim_params(self) -> dict:
        # Optimize only XVLA parameters during policy_extract; critic params handled elsewhere
        if self.config.train_mode == "policy":
            return self.xvla.get_optim_params()
        else:
            return {

            }

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """
        Training path for policy extraction:
          - choose conditioned vs unconditioned tokens per sample via label dropout
          - call XVLA forward once with the selected input_ids under OBS_LANGUAGE_TOKENS
        Value/critic training is not implemented here (you asked to do critic head later).
        """
        batch = dict(batch)
        if self.config.train_mode == "policy":
            return self._policy_extraction_step(batch)
        else:
            return self._critic_training_step(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:  # noqa: ARG002
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.xvla.n_action_steps])

        return self._queues[ACTION].popleft()

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        inputs = self._build_model_inputs(batch)
        actions = self._generate_actions_guided(**inputs, steps=self.config.num_denoising_steps)
        return actions

    def _policy_extraction_step(self, batch):
        ids_u = batch[OBS_LANGUAGE_TOKENS]
        ids_c = batch[self.config.cond_tokens_key]

        # with prob p, train unconditional branch (use ids_u)
        p = float(self.config.advantage_label_dropout_p)
        if p > 0:
            drop = (torch.rand(ids_u.shape[0], device=ids_u.device) < p).view(-1, 1)
            input_ids = torch.where(drop, ids_u, ids_c)
        else:
            input_ids = ids_c

        # XVLA expects tokens under OBS_LANGUAGE_TOKENS, overwrite
        batch[OBS_LANGUAGE_TOKENS] = input_ids

        return self.xvla.forward(batch)

    def _critic_training_step(self, batch):
        pass

    # ---------------------------
    # Guided sampler implementation
    # ---------------------------
    def _build_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids_u = batch[OBS_LANGUAGE_TOKENS]
        input_ids_c = batch[self.config.cond_tokens_key]
        batch_size = input_ids_u.shape[0]
        images, image_mask = self.xvla._prepare_images(batch)
        domain_id = self.xvla._get_domain_id(batch, batch_size, images.device)
        proprio = self.xvla._prepare_state(batch, batch_size, images.device)
        return {
            "input_ids_u": input_ids_u,
            "input_ids_c": input_ids_c,
            "image_input": images,
            "image_mask": image_mask,
            "domain_id": domain_id,
            "proprio": proprio,
        }

    def _generate_actions_guided(
        self,
        input_ids_u: torch.LongTensor,
        input_ids_c: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """
        True CFG guidance for X-VLA flow sampling:
          - same latent x_t
          - per-step vector field combination
          - conditioned/unconditioned differ only by VLM text tokens
        """
        model = self.xvla.model
        target_dtype = model._get_target_dtype()
        image_input = image_input.to(dtype=target_dtype)
        proprio = proprio.to(dtype=target_dtype)

        enc_u = model.forward_vlm(input_ids_u, image_input, image_mask)
        enc_c = model.forward_vlm(input_ids_c, image_input, image_mask)

        batch_size = input_ids_u.shape[0]
        action_dim = model.dim_action

        x1 = torch.randn(batch_size, model.chunk_size, action_dim, device=proprio.device, dtype=target_dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((batch_size,), i / steps, device=proprio.device, dtype=target_dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)

            proprio_m, x_t_m = model.action_space.preprocess(proprio, x_t)

            # Unconditional vector field
            v_u = model.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc_u
            )

            # Conditional vector field
            v_c = model.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc_c
            )

            # Classifier-free guidance
            v = v_u + self.config.guidance_scale * (v_c - v_u)
            action = v

        return model.action_space.postprocess(action)

    def build_critic_backbone(self) -> CriticBackbone:
        if self.xvla is None:
            raise RuntimeError("XVLA backbone must be initialized before building critic backbone.")

        xvla_model: XVLAModel = self.xvla.model

        # Deepcopy to prevent critic training from mutating the policy’s XVLA.
        xvla_model_c = copy.deepcopy(xvla_model)

        tr = xvla_model_c.transformer

        # Copy selected transformer blocks (prefix) for concat_transformer fusion.
        n = int(self.config.critic_backbone.n_reused_transformer_blocks)
        reused = nn.ModuleList([copy.deepcopy(b) for b in list(tr.blocks)[:n]])

        bundle = CriticBackboneBundle(
            xvla_model=xvla_model_c,
            vlm_proj=copy.deepcopy(tr.vlm_proj),
            aux_visual_proj=copy.deepcopy(tr.aux_visual_proj),
            action_encoder=copy.deepcopy(tr.action_encoder),
            pos_emb=copy.deepcopy(tr.pos_emb) if hasattr(tr, "pos_emb") else None,
            reused_blocks=reused,
        )

        return CriticBackbone(cfg=self.config.critic_backbone, bundle=bundle)
