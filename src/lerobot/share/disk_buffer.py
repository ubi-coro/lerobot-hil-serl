import datetime
import functools
import os
from copy import copy
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
import torch

from lerobot.configs.default import DatasetBufferConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, MultiLeRobotDataset
from lerobot.datasets.utils import cycle
from lerobot.rl.buffer import BatchTransition, random_shift
from lerobot.utils.constants import DONE, ACTION, REWARD, OBS_IMAGE


class DRQCollate:
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        use_drq: bool = True,
        image_aug_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        state_keys: Optional[List[str]] = None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_drq = use_drq
        self.state_keys = state_keys

        if image_aug_fn is None:
            # default DrQ shift (you can torch.compile it here if you like)
            try:
                self.image_aug_fn = torch.compile(functools.partial(random_shift, pad=4))
            except Exception:
                self.image_aug_fn = functools.partial(random_shift, pad=4)
        else:
            self.image_aug_fn = image_aug_fn

    def _stack_dict(self, list_of_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        keys = self.state_keys or list(list_of_dicts[0].keys())
        for k in keys:
            out[k] = torch.stack([d[k] for d in list_of_dicts], dim=0).to(self.device)
        return out

    def __call__(self, batch: List[Dict[str, Any]]) -> BatchTransition:
        # batch is a list of items returned by __getitem__
        # 1) Stack states/next_states
        state_list = [b["state"] for b in batch]
        next_state_list = [b["next_state"] for b in batch]

        batch_state = self._stack_dict(state_list)
        batch_next_state = self._stack_dict(next_state_list)

        # 2) Identify image keys (like sample() does via prefix)
        image_keys = [k for k in batch_state if k.startswith(OBS_IMAGE)] if self.use_drq else []

        # 3) Apply DrQ augmentation in a single big batch, then split back
        if self.use_drq and image_keys:
            cat_images = []
            for k in image_keys:
                cat_images.append(batch_state[k])       # (B,C,H,W)
                cat_images.append(batch_next_state[k])  # (B,C,H,W)
            all_images = torch.cat(cat_images, dim=0)    # (2*K*B, C, H, W)

            augmented = self.image_aug_fn(all_images)     # same shape

            B = batch_state[image_keys[0]].shape[0]
            for i, k in enumerate(image_keys):
                # for each key, first B are state, next B are next_state
                start = 2 * i * B
                batch_state[k] = augmented[start : start + B]
                batch_next_state[k] = augmented[start + B : start + 2 * B]

        # 4) Stack the rest
        actions = torch.stack([b["action"] for b in batch], dim=0).to(self.device)
        rewards = torch.stack([b["reward"] for b in batch], dim=0).to(self.device)
        dones = torch.stack([b["done"] for b in batch], dim=0).to(self.device).float()
        truncateds = torch.stack([b["truncated"] for b in batch], dim=0).to(self.device).float()

        # 5) Complementary info (if present)
        comp_info = None
        if batch[0]["complementary_info"] is not None:
            comp_info = {}
            keys = batch[0]["complementary_info"].keys()
            for k in keys:
                comp_info[k] = torch.stack([b["complementary_info"][k] for b in batch], dim=0).to(self.device)

        return BatchTransition(
            state=batch_state,
            action=actions,
            reward=rewards,
            next_state=batch_next_state,
            done=dones,
            truncated=truncateds,
            complementary_info=comp_info,
        )


class OnDiskReplayBuffer(torch.utils.data.Dataset):
    """
    Dataset wrapper that returns (state, action, reward, next_state, done, truncated, complementary_info)
    one transition per index, respecting episode boundaries.
    """
    def __init__(
        self,
        ds_cfg: DatasetBufferConfig,
        policy_cfg: PreTrainedConfig,
        fps: int = 30,
        robot_type: str | None = None,
        device: str = "cpu",
        num_workers: int = 0,
        resume: bool = True
    ):
        self.ds_cfg = ds_cfg
        self.policy_cfg = policy_cfg
        self.fps = fps
        self.device = device
        self.robot_type = robot_type
        self.num_workers = num_workers
        self.resume = resume
        self.policy_name = "_" + policy_cfg.pretrained_path.name if policy_cfg.pretrained_path is not None else ""
        self.state_keys = list(policy_cfg.input_features.keys())

        # if called with load_dir, we expect the directory to be non-empty
        root_exists = self._root_exists()
        assert not root_exists or resume

        self.read_ds, self.write_ds = None, None
        if root_exists:
            self._make_read_ds()

        """
        # cache complementary_info keys (string prefix)
        self.complementary_prefix = "complementary_info."
        self.complementary_info_keys = [k for k in probe if k.startswith(self.complementary_prefix)]
        self.has_complementary_info = len(self.complementary_info_keys) > 0
        """

    def __len__(self) -> int:
        return 0 if self.read_ds is None else len(self.read_ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.read_ds[idx]

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
        done_val = bool(sample[DONE].item())
        done = torch.tensor(done_val, dtype=torch.bool)
        truncated = torch.tensor(False, dtype=torch.bool)

        # --- next_state (stay within episode) ---
        next_state = state  # default: terminal transition copies current state
        if not done_val and idx < len(self.read_ds) - 1:
            next_sample = self.read_ds[idx + 1]
            if int(next_sample["episode_index"].item()) == int(sample["episode_index"].item()):
                next_state = {k: next_sample[k] for k in self.state_keys}

        # --- complementary_info (optional, no manual batch dim here) ---
        complementary_info = None
        """
        if self.has_complementary_info:
            complementary_info = {}
            off = len(self.complementary_prefix)
            for key in self.complementary_info_keys:
                clean_key = key[off:]
                val = sample[key]
                complementary_info[clean_key] = val  # leave as-is; DataLoader will stack
        """

        return {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "truncated": truncated,
            "complementary_info": complementary_info,
        }

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
        if self.write_ds is None:
            self._make_write_ds(state, action)

        frame = {
            **{k: state[k].cpu() for k in state},
            ACTION: action.cpu(),
            REWARD: torch.tensor([reward], dtype=torch.float32).cpu(),
            DONE: torch.tensor([done], dtype=torch.bool).cpu(),
            "task": self.ds_cfg.single_task
        }

        """
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
        """

        # handle episode save logic
        self.write_ds.add_frame(frame)

        if done or truncated:
            self.write_ds.save_episode()

    def get_iterator(
        self,
        batch_size: int,
        async_prefetch: bool = True,
        queue_size: int = 2
    ):
        prefetch = queue_size if self.num_workers > 0 else None
        pin_memory = (self.device == "cuda") or (hasattr(self.device, "type") and self.device.type == "cuda")
        dataloader = torch.utils.data.DataLoader(
            self.read_ds,
            num_workers=self.num_workers,
            batch_size=batch_size,
            shuffle=True,
            sampler=None,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=prefetch,
            collate_fn=DRQCollate(state_keys=self.state_keys, device=self.device)
        )
        dl_iter = cycle(dataloader)

        return dl_iter

    def save_episode(self):
        self.write_ds.save_episode()

    def _root_exists(self):
        root = Path(self.ds_cfg.root)
        if self.ds_cfg.load_dir:
            return root.exists() and any(p.is_dir() for p in root.iterdir())
        return root.exists()

    def _make_read_ds(self):

        new_ds_cfg = copy(self.ds_cfg)
        if self.ds_cfg.load_dir:
            root = Path(self.ds_cfg.root)
            root_name = root.name
            subdirs = [p.name for p in root.iterdir() if p.is_dir()]
            repo_items = [f"{root_name}/{name}" for name in sorted(subdirs)]
            new_ds_cfg.root = root.parent
            new_ds_cfg.repo_id = "[" + ",".join(repo_items) + "]"

        else:
            new_ds_cfg.root = self.ds_cfg.root
            new_ds_cfg.repo_id = self.ds_cfg.repo_id

        self.read_ds = make_dataset(TrainPipelineConfig(dataset=new_ds_cfg, policy=self.policy_cfg))

        if not isinstance(self.read_ds, MultiLeRobotDataset):
            self.write_ds = self.read_ds

        if self.read_ds is None:
            self.read_ds = self.write_ds

    def _make_write_ds(self, state: dict[str, torch.Tensor], action: torch.Tensor):

        # build features
        features = {
            ACTION: {"dtype": "float32", "shape": action.shape, "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }

        num_cameras = 0
        for key in self.state_keys:
            if len(state[key].shape) == 3:
                features[key] = {
                    "dtype": "video",
                    "shape": state[key].shape,
                    "names": ["channels", "height", "width"],
                }
                num_cameras += 1
            else:
                features[key] = {"dtype": "float32", "shape": state[key].shape, "names": None}

        # automatically generate root for new dataset in the multi dataset
        if isinstance(self.read_ds, MultiLeRobotDataset):
            t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            root = Path(self.ds_cfg.root) / f"{t}{self.policy_name}"
            repo_id = f"{Path(self.ds_cfg.root).name}/{t}{self.policy_name}"
        else:
            root, repo_id = self.ds_cfg.root, self.ds_cfg.repo_id

        self.write_ds = LeRobotDataset.create(
            repo_id,
            root=root,
            fps=self.fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=self.ds_cfg.video,
            batch_encoding_size=self.ds_cfg.video_encoding_batch_size,
        )

        # merge into multi dataset by reloading the folder
        # from then on we can sample the new dataset with correct delta indices
        if not isinstance(self.read_ds, MultiLeRobotDataset):
            self._make_read_ds()

            # reload write_ds from the read dataset so both point to the same object
            for ds in self.read_ds._datasets:
                if Path(ds.root) == Path(root):
                    self.write_ds = ds
                    break

        # start image writer
        self.write_ds.start_image_writer(
            num_processes=self.ds_cfg.num_image_writer_processes,
            num_threads=self.ds_cfg.num_image_writer_threads_per_camera * num_cameras
        )

    def to_lerobot_dataset(self, **kwargs):
        pass

    def from_lerobot_dataset(self, **kwargs):
        pass
