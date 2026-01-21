import logging
from dataclasses import dataclass

from lerobot.cameras import CameraConfig

logger = logging.getLogger(__name__)

@CameraConfig.register_subclass("mujoco_camera")
@dataclass
class MujocoCameraConfig(CameraConfig):
    name = ""
    fps: int | None = 30
    width: int | None = 640
    height: int | None = 480
