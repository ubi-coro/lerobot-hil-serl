from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import datasets

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.datasets.utils import DEFAULT_DATA_PATH, load_info, write_info
from lerobot.utils.constants import ACTION, OBS_STATE
from share.policies.cfgrl_common.action_providers import PolicyActionProvider
from share.policies.cfgrl_critic import label_chunk_dataset
from share.policies.cfgrl_critic.label_dataset import load_local_chunk_dataset
from tests.share.policies.helpers import DEFAULT_IMAGE_KEY, make_critic


def _write_chunk_dataset(root: Path, *, chunk_size: int, action_dim: int, state_dim: int, image_size: tuple[int, int]) -> None:
    height, width = image_size
    num_rows = 6
    features = {
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        f"state.{OBS_STATE}": {"dtype": "float32", "shape": (state_dim,), "names": None},
        f"state.{DEFAULT_IMAGE_KEY}": {
            "dtype": "float32",
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        ACTION: {"dtype": "float32", "shape": (chunk_size, action_dim), "names": None},
    }
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": "test",
        "total_episodes": 2,
        "total_frames": num_rows,
        "total_tasks": 1,
        "total_videos": 0,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 10,
        "splits": {},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": None,
        "features": features,
    }
    write_info(info, root)

    hf_dataset = datasets.Dataset.from_dict(
        {
            "timestamp": np.linspace(0.0, 0.5, num_rows, dtype=np.float32),
            "frame_index": np.arange(num_rows, dtype=np.int64),
            "episode_index": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "index": np.arange(num_rows, dtype=np.int64),
            "task_index": np.zeros(num_rows, dtype=np.int64),
            f"state.{OBS_STATE}": np.random.randn(num_rows, state_dim).astype(np.float32),
            f"state.{DEFAULT_IMAGE_KEY}": np.random.randn(num_rows, 3, height, width).astype(np.float32),
            ACTION: np.random.randn(num_rows, chunk_size, action_dim).astype(np.float32),
        },
        features=datasets.Features(
            {
                "timestamp": datasets.Value("float32"),
                "frame_index": datasets.Value("int64"),
                "episode_index": datasets.Value("int64"),
                "index": datasets.Value("int64"),
                "task_index": datasets.Value("int64"),
                f"state.{OBS_STATE}": datasets.Sequence(datasets.Value("float32"), length=state_dim),
                f"state.{DEFAULT_IMAGE_KEY}": datasets.Array3D(shape=(3, height, width), dtype="float32"),
                ACTION: datasets.Array2D(shape=(chunk_size, action_dim), dtype="float32"),
            }
        ),
    )
    data_path = root / DEFAULT_DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(data_path)


def test_label_chunk_dataset_writes_labels_and_metadata(tmp_path):
    critic = make_critic(chunk_size=4)

    class MockSamplingPolicy:
        def __init__(self, chunk_size: int, action_dim: int):
            self.chunk_size = chunk_size
            self.action_dim = action_dim

        def sample_actions(self, obs, condition=None, guidance_scale=None, num_samples: int = 1, noise=None):
            batch_size = obs[OBS_STATE].shape[0]
            base = torch.linspace(-0.5, 0.5, self.chunk_size * self.action_dim).view(
                1, 1, self.chunk_size, self.action_dim
            )
            return base.expand(batch_size, num_samples, -1, -1).clone()

    dataset_root = tmp_path / "chunk_dataset"
    _write_chunk_dataset(
        dataset_root,
        chunk_size=critic.config.chunk_size,
        action_dim=critic.action_dim,
        state_dim=critic.state_dim,
        image_size=critic.config.backbone.vision_backbone.image_size,
    )
    provider = PolicyActionProvider(policy=MockSamplingPolicy(critic.config.chunk_size, critic.action_dim))

    summary = label_chunk_dataset(
        critic,
        dataset_root,
        label_field="cfgrl_optimal",
        percentile=50.0,
        batch_size=2,
        action_provider=provider,
        device="cpu",
    )

    info = load_info(dataset_root)
    labeled_dataset, _ = load_local_chunk_dataset(dataset_root)

    assert summary.label_field == "cfgrl_optimal"
    assert summary.chunk_size == critic.config.chunk_size
    assert "cfgrl_optimal" in info["features"]
    assert "cfgrl_optimal" in labeled_dataset.column_names
    assert "cfgrl_labeling" in info
    assert info["cfgrl_labeling"]["cfgrl_optimal"]["chunk_size"] == critic.config.chunk_size
