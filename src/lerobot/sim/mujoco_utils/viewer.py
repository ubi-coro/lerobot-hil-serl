import time

import mujoco, mujoco.viewer

from abc import ABC, abstractmethod
import cv2
import logging


def create_viewer(key, **kwargs):
    return AbstractViewer.create_viewer(key, **kwargs)

class ViewerRegistry:
    _registry = {}

    @classmethod
    def register_subclass(cls, key):
        def decorator(subclass):
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def create_viewer(cls, key, **kwargs):
        if key not in cls._registry:
            raise ValueError(f"Unknown viewer type: {key}")
        return cls._registry[key](**kwargs)


class AbstractViewer(ViewerRegistry, ABC):
    def start(self): pass

    def stop(self): pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @abstractmethod
    def is_running(self): pass

    @abstractmethod
    def sync(self, observation): pass


@AbstractViewer.register_subclass("mujoco")
class MujocoViewer(AbstractViewer):
    def __init__(self, model, data, hz=30 ,**kwargs):
        self.model = model
        self.data = data
        self.viewer = None
        self.hz = hz
        self._frame_dt = 1.0 / self.hz
        self._last_render = 0.0

        if kwargs:
            logging.debug(f"Unused parameters in MujocoViewer: {kwargs}")


    def start(self):
        if not self.is_running():
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
            # self.viewer.__enter__()

    def stop(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_running(self):
        return self.viewer.is_running() if self.viewer else False

    def sync(self, observation):
        if not self.viewer:
            raise RuntimeError(f"Viewer is not running")
        now = time.perf_counter()
        if now - self._last_render >= self._frame_dt:
            self.viewer.sync()
            self._last_render = now


@AbstractViewer.register_subclass("camera")
class CVViewer(AbstractViewer):
    def __init__(self, image_keys=None, **kwargs):
        self.image_keys = image_keys
        self.running = False

        if kwargs:
            logging.debug(f"Unused parameters in CVViewer: {kwargs}")

    def start(self):
        self.running = True

    def stop(self):
        cv2.destroyAllWindows()
        self.running = False

    def is_running(self):
        return self.running

    def sync(self, observation):
        if not self.image_keys:
            self.image_keys = list(observation['pixels'].keys())
        for key in self.image_keys:
            image = observation['pixels'][key].squeeze(0)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(key, image_bgr)

        # close with Esc
        if cv2.waitKey(1) == 27:
            self.running = False


@AbstractViewer.register_subclass("both")
class MujocoCameraViewer(AbstractViewer):
    def __init__(self, model, data, image_keys, **kwargs):
        self.model = model
        self.data = data
        self.viewer = None
        self.image_keys = image_keys
        self.running = False
        if kwargs:
            logging.debug(f"Unused parameters in MujocoViewer: {kwargs}")


    def start(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_HAZE] = 0
            self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_CULL_FACE] = 0
            self.running = True

    def stop(self):
        if self.viewer is not None:
            cv2.destroyAllWindows()
            self.viewer.close()
            self.viewer = None
            self.running = False

    def is_running(self):
        return self.viewer.is_running()  if self.viewer else False

    def sync(self, observation):
        if self.viewer:
            self.viewer.sync()
        if not self.image_keys:
            self.image_keys = list(observation['pixels'].keys())
        for key in self.image_keys:
            image = observation['pixels'][key].squeeze(0)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imshow(key, image_bgr)
        cv2.waitKey(1)

@AbstractViewer.register_subclass("headless")
class HeadlessViewer(AbstractViewer):
    def __init__(self, **kwargs):
        self.running = False
        if kwargs:
            logging.debug(f"Unused parameters in HeadlessViewer: {kwargs}")

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running

    def sync(self, observation):
        pass