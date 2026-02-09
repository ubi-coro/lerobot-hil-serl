import abc
from dataclasses import dataclass, field
from typing import List

import draccus


@dataclass
class SimConfig(draccus.ChoiceRegistry, abc.ABC):
    env: str
    viewer: str
    image_keys: List[str]
    calibration_dir: str


    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@SimConfig.register_subclass("aloha")
@dataclass
class AlohaSimConfig(SimConfig):
    env: str = "aloha"
    viewer: str = "mujoco"
    image_keys: List[str] = field(default_factory=lambda: [
        "wrist_cam_right",
        "wrist_cam_left",
        "teleoperator_pov",
        "overhead_cam",
    ])
    calibration_dir: str = ".cache/calibration/aloha_sim"


