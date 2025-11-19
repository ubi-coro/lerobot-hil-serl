import random
from typing import Optional, List

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

from lerobot.utils.constants import REWARD


# from your code:
# from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotLIVWrapper(IterableDataset):
    """
    LIV-style wrapper around a LeRobotDataset.

    Each sample is:
        (images, reward, text)

    where `images` has shape (5, C, H, W) and corresponds to:
        [o_start, o_end, o_t, o_{t+1}, o_text]

    - o_start: randomly sampled start frame in the episode
    - o_end:   randomly sampled end frame (after start)
    - o_t, o_{t+1}: intermediate frames between start and end
    - o_text:  a frame near the end (used as "text image" in original LIV)

    The reward is the self-supervised temporal reward used in LIV:
        reward = float(s0_ind == end_ind) - 1   # usually -1, 0 at very end

    `text` is taken from the episode's task string.
    """

    def __init__(
        self,
        base_dataset: "LeRobotDataset",
        camera_key: Optional[str] = None,
        num_episodes: Optional[int] = None,
        doaug: str = "none",
        alpha: float = 0.95,
        resize_to: int = 256,
        crop_size: int = 224,
    ):
        super().__init__()
        self.base = base_dataset
        self.alpha = alpha
        self.doaug = doaug

        # Pick a camera key if not provided
        if camera_key is None:
            if len(self.base.meta.camera_keys) == 0:
                raise ValueError("LeRobotLIVWrapper: no camera_keys found in LeRobotDataset.")
            camera_key = self.base.meta.camera_keys[0]
        self.camera_key = camera_key

        # --- Pre / post transforms similar to LIV ---
        self.preprocess = torch.nn.Sequential(
            transforms.Resize(resize_to, antialias=None),
            transforms.CenterCrop(crop_size),
        )

        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(crop_size, scale=(0.2, 1.0), antialias=None),
            )
        elif doaug in ["metaworld"]:
            self.aug = torch.nn.Sequential(
                transforms.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02, hue=0.02),
                transforms.RandomAffine(20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            )
        else:
            self.aug = lambda x: x  # identity

        # Build episode boundaries in global index space: list of (ep_idx, start, end)
        self._episodes = self._build_episode_index(num_episodes=num_episodes)

    def _build_episode_index(self, num_episodes: Optional[int]) -> List[tuple]:
        """
        Build a list of (episode_index, start_idx, end_idx_exclusive) in the hf_dataset index space.
        Assumes frames from each episode are stored contiguously.
        """
        ds = self.base.hf_dataset

        # Get episode_index column without transforms
        # hf_dataset["episode_index"] returns a list; they may be ints or tensors
        ep_col = ds["episode_index"]
        ep_col = [int(e) if not isinstance(e, torch.Tensor) else int(e.item()) for e in ep_col]

        episodes = []
        if len(ep_col) == 0:
            return episodes

        current_ep = ep_col[0]
        start_idx = 0
        for i, ep in enumerate(ep_col):
            if ep != current_ep:
                episodes.append((current_ep, start_idx, i))
                current_ep = ep
                start_idx = i
        # last episode
        episodes.append((current_ep, start_idx, len(ep_col)))

        # Optionally restrict to first N episodes
        if num_episodes is not None:
            episodes = episodes[:num_episodes]

        return episodes

    def _sample(self):
        """
        Sample a LIV-style training tuple from the underlying LeRobotDataset.

        Returns:
            images: Tensor (5, C, H, W)
            reward: float
            text:   str
        """
        # 1) Sample an episode
        ep_id, ep_start, ep_end = random.choice(self._episodes)
        vidlen = ep_end - ep_start  # number of frames in this episode

        if vidlen < 2:
            # too short, resample
            return self._sample()

        # 2) Sample indices as in LIV
        # start_ind, end_ind in [0, vidlen-1] with end > start
        start_ind = random.randint(0, vidlen - 2)
        end_ind = random.randint(start_ind + 1, vidlen - 1)

        # end_text_ind similarly near the end (using alpha)
        end_text_min = max(int(self.alpha * vidlen) - 1, start_ind + 1)
        end_text_min = min(end_text_min, vidlen - 1)
        end_text_ind = random.randint(end_text_min, vidlen - 1)

        # intermediate indices
        s0_ind = random.randint(start_ind, end_ind)
        s1_ind = min(s0_ind + 1, end_ind)

        # Map to global indices
        start_idx_global = ep_start + start_ind
        end_idx_global = ep_start + end_ind
        s0_idx_global = ep_start + s0_ind
        s1_idx_global = ep_start + s1_ind
        end_text_idx_global = ep_start + end_text_ind

        # 3) Self-supervised reward (LIV uses -1 step cost, 0 at final)
        reward = float(s0_ind == end_ind) - 1.0

        # 4) Fetch frames from base dataset
        # Each item is a dict; we take the camera image + task string
        im0_item = self.base[start_idx_global]
        imT_item = self.base[end_idx_global]
        imt0_item = self.base[s0_idx_global]
        imt1_item = self.base[s1_idx_global]
        im_text_item = self.base[end_text_idx_global]

        # Get the task string (same for all frames in an episode)
        text = im0_item.get("task", "")

        # Extract camera images, convert to float [0,1]
        def get_img(it):
            img = it[self.camera_key]
            # LeRobot's hf_transform_to_torch usually returns CxHxW uint8
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            return img

        im0 = get_img(im0_item)
        imT = get_img(imT_item)
        imt0 = get_img(imt0_item)
        imt1 = get_img(imt1_item)
        im_text = get_img(im_text_item)

        # 5) Apply preprocess & augmentation
        if isinstance(self.preprocess, torch.nn.Module):
            im0 = self.preprocess(im0)
            imT = self.preprocess(imT)
            imt0 = self.preprocess(imt0)
            imt1 = self.preprocess(imt1)
            im_text = self.preprocess(im_text)

        if self.doaug == "rctraj":
            # augment all frames in the same way
            allims = torch.stack([im0, imT, imt0, imt1, im_text], dim=0)
            allims = self.aug(allims)
            im0, imT, imt0, imt1, im_text = allims
        else:
            im0 = self.aug(im0)
            imT = self.aug(imT)
            imt0 = self.aug(imt0)
            imt1 = self.aug(imt1)
            im_text = self.aug(im_text)

        return {
            self.camera_key: torch.stack([im0, imT, imt0, imt1, im_text], dim=0),
            REWARD: reward,
            "task": text
        }

    def __iter__(self):
        # Stateless, infinite iterator like in LIVBuffer
        while True:
            yield self._sample()
