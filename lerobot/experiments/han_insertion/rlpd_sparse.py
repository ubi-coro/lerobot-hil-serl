from dataclasses import dataclass, field

from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse")
@dataclass
class HAN_Insertion_RLPD_Sparse(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_2"

    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_no_vision")
@dataclass
class HAN_Insertion_RLPD_Sparse_NoVision(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_no_vision_1"

    use_vision: bool = False

    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })


@MPNetConfig.register_subclass("han_insertion_rlpd_sparse_dgn")
@dataclass
class HAN_Insertion_RLPD_Sparse_DGN(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/rlpd_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_dgn_0"

    primitives: dict[str, MPConfig] = field(default_factory=lambda: {
        "press": MPConfig(
            transitions={"insert": "start_insertion"},
            tff=InsertionPrimitive().tff
        ),
        "insert": InsertionPrimitive(
            sparse_reward=True,
            transitions={"terminal": "terminal_false"}  # automatically after timeout
        ),
        "terminal": MPConfig(is_terminal=True),
    })

    def __post_init__(self):
        self.primitives["insert"].policy.config.noise_config.enable = True
        super().__post_init__()