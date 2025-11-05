from typing import Sequence, Dict, Any
import torch

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import DONE, ACTION, REWARD


class OnDiskReplayBuffer(torch.utils.data.Dataset):
    """
    Dataset wrapper that returns (state, action, reward, next_state, done, truncated, complementary_info)
    one transition per index, respecting episode boundaries.
    """
    def __init__(self, cfg: TrainRLServerPipelineConfig, dataset: LeRobotDataset, task_name: str = "replay_buffer"):
        self.cfg = cfg
        self.dataset = dataset
        self.state_keys = list(cfg.policy.input_features.keys())
        self.load_next_state = isinstance(cfg.policy, SACConfig)
        self.task_name = task_name
        self.num_frames = len(dataset)

        # Probe a sample to discover available keys
        probe = dataset[0]
        self.has_done_key = (DONE in probe)

        # cache complementary_info keys (string prefix)
        self.complementary_prefix = "complementary_info."
        self.complementary_info_keys = [k for k in probe if k.startswith(self.complementary_prefix)]
        self.has_complementary_info = len(self.complementary_info_keys) > 0

    def __len__(self) -> int:
        return self.num_frames

    def _infer_done(self, idx: int, cur_ep: int) -> bool:
        """Done if this is the last frame or next frame belongs to a new episode."""
        if idx == self.num_frames - 1:
            return True
        next_ep = int(self.dataset[idx + 1]["episode_index"].item())
        return next_ep != cur_ep

    def add(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        next_state: dict[str, torch.Tensor],
        done: bool,
        truncated: bool,
        complementary_info: dict[str, torch.Tensor] | None = None,
    ):
        frame = {
            **{k: state[k].cpu() for k in self.state_keys},
            ACTION: action.cpu(),
            REWARD: torch.tensor([reward], dtype=torch.float32).cpu(),
            DONE: torch.tensor([done], dtype=torch.bool).cpu(),
            "task": self.task_name
        }

        # Add complementary_info if available
        if self.has_complementary_info:
            for key in self.complementary_info_keys:
                val = complementary_info[key]
                # Convert tensors to CPU
                if isinstance(val, torch.Tensor):
                    if val.ndim == 0:
                        val = val.unsqueeze(0)
                    frame[f"complementary_info.{key}"] = val.cpu()
                # Non-tensor values can be used directly
                else:
                    frame[f"complementary_info.{key}"] = val

        # handle episode save logic
        self.dataset.add_frame(frame)

        if done or truncated:
            self.dataset.save_episode()

    def get_iterator(self):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=2,
        )
        dl_iter = cycle(dataloader)

        return dl_iter

    def to_lerobot_dataset(self, **kwargs):
        pass

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]

        # --- state ---
        state: Dict[str, torch.Tensor] = {k: sample[k] for k in self.state_keys}

        # --- action ---
        action: torch.Tensor = sample[ACTION]

        # --- reward ---
        reward = sample[REWARD]
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        else:
            reward = reward.to(dtype=torch.float32)

        # --- done/truncated ---
        if self.has_done_key:
            done_val = bool(sample[DONE].item())
        else:
            cur_ep = int(sample["episode_index"].item())
            done_val = self._infer_done(idx, cur_ep)
        done = torch.tensor(done_val, dtype=torch.bool)
        truncated = torch.tensor(False, dtype=torch.bool)

        # --- next_state (stay within episode) ---
        next_state = state  # default: terminal transition copies current state
        if not done_val and idx < self.num_frames - 1 and self.load_next_state:
            next_sample = self.dataset[idx + 1]
            if int(next_sample["episode_index"].item()) == int(sample["episode_index"].item()):
                next_state = {k: next_sample[k] for k in self.state_keys}

        # --- complementary_info (optional, no manual batch dim here) ---
        complementary_info = None
        if self.has_complementary_info:
            complementary_info = {}
            off = len(self.complementary_prefix)
            for key in self.complementary_info_keys:
                clean_key = key[off:]
                val = sample[key]
                complementary_info[clean_key] = val  # leave as-is; DataLoader will stack

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "truncated": truncated,
            "complementary_info": complementary_info,
        }
