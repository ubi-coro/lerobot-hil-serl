import logging
from typing import Any
from numpy._typing import NDArray

from lerobot.cameras import Camera, ColorMode
from .configuration_mujoco_camera import MujocoCameraConfig
from lerobot.sim.mujoco_utils.sim_singleton import SimManager

logger = logging.getLogger(__name__)

class MujocoCamera(Camera):
    def __init__(self, config: MujocoCameraConfig):
        super().__init__(config)
        self.name = config.mujoco_id
        self.sim = SimManager.get()

    @property
    def is_connected(self) -> bool:
        return True

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return []

    def connect(self, warmup: bool = True) -> None:
        pass

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        return self.sim.get_observation(self.name).squeeze()

    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        return self.read()

    def disconnect(self) -> None:
        logger.info(f"{self} disconnected.")
