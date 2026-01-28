import torch
from torch import Tensor, nn

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from .configuration_xpvla_policy import XPVLAPolicyConfig
from ..pretrained import PreTrainedPolicy
from ...configs.policies import PreTrainedConfig


# -----------------------------
# Top-level CFGRL-AC policy
# -----------------------------

@PreTrainedPolicy.register_subclass("xpvla_policy")
class XPVLAPolicy(XVLAPolicy):
    """
    Wrapper policy around XVLAPolicy that implements:
      - label dropout during policy extraction training by selecting uncond/cond tokens per sample
      - true CFG guided sampling at sampler level (per solver step), without modifying XVLA internals
      - optional reuse of image encoding across the uncond/cond branches
    """

    config_class = XPVLAPolicyConfig
    name = "xpvla_policy"

    def forward(self, batch):
        batch = dict(batch)
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

        return super().forward(batch)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, *, config=None, **kwargs):
        if config is None:
            base = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
            if not isinstance(base, XPVLAPolicyConfig):
                # upgrade: copy over XVLA fields into XPVLAPolicyConfig
                config = XPVLAPolicyConfig(**base.__dict__)
            else:
                config = base
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _get_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        if self.config.use_advantage_conditioning:
            inputs = self._build_conditional_model_inputs(batch)
            return self._generate_actions_guided(**inputs, steps=self.config.num_denoising_steps)
        else:
            return super()._get_action_chunk(batch)

    def _build_conditional_model_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        input_ids_u = batch[OBS_LANGUAGE_TOKENS]
        input_ids_c = batch[self.config.cond_tokens_key]
        batch_size = input_ids_u.shape[0]
        images, image_mask = self._prepare_images(batch)
        domain_id = self._get_domain_id(batch, batch_size, images.device)
        proprio = self._prepare_state(batch, batch_size, images.device)
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
        model = self.model
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
