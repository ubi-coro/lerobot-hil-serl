import logging
from typing import Any

import numpy as np
from numpy._typing import NDArray

from lerobot.cameras import Camera, ColorMode

logger = logging.getLogger(__name__)
class SimCamera(Camera):
    @property
    def is_connected(self) -> bool:
        return True

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        pass

    def connect(self, warmup: bool = True) -> None:
        pass

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        return self.async_read()

    def async_read(self, timeout_ms: float = ...) -> NDArray[Any]:
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def disconnect(self) -> None:
        logger.info(f"{self} disconnected.")
