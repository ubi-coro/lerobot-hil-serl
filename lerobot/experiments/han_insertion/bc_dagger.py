from dataclasses import dataclass, field

from lerobot.experiments.han_insertion.base import InsertionPrimitive, HAN_Insertion
from lerobot.scripts.server.mp_nets import MPNetConfig, MPConfig


@MPNetConfig.register_subclass("han_insertion_bc_dagger")
@dataclass
class HAN_Insertion_BC_DAgger(HAN_Insertion):
    root: str = "/home/jannick/data/paper/hil-amp/dagger_reward_sparse_cam_toWindow_terminate_early_init_large_demos_itv_2"

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
        super().__post_init__()
        self.primitives["insert"].policy.config.use_bc_dagger = True