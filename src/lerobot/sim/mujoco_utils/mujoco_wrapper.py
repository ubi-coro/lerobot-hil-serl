import gc
import time

import mujoco

from lerobot.sim.mujoco_utils.viewer import PassiveViewer, Viewer


class _MJ:
    def __init__(self):
        self.model = None
        self.data = None
        self.viewer: Viewer = Viewer(self.model, self.data)
        self.rt = False
        self._t0_wall = None
        self._t0_sim = None

    def step(self):
        assert self.model is not None
        mujoco.mj_step(self.model, self.data)
        self.viewer.step()
        if not self.rt:
            return

        target = time.perf_counter() - self._t0_wall
        sim_elapsed = self.data.time - self._t0_sim
        ahead = sim_elapsed - target
        if ahead > 0:
            time.sleep(min(ahead, 0.002))


    def no_physics_sync(self):
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.force_sync()

    def init_viewer(self, hz=60):
        self.viewer = PassiveViewer(MJ.model, MJ.data, hz=hz)
        self.viewer.start()
        self._t0_wall = time.perf_counter()
        self._t0_sim = self.data.time

    def stop(self):
        """Stop simulation and close viewer."""

        self.viewer.stop()
        self.model = None
        self.data = None
        self.rt = False
        self._t0_wall = None
        self._t0_sim = None
        gc.collect()

    def idle(self, idle_time):
        t_start = self.data.time
        while self.data.time - t_start < idle_time:
            self.step()


MJ = _MJ()

def init_mujoco(xml_path: str, display=True, rt=True, hz=60):
    if MJ.model is not None or MJ.data is not None:
        MJ.stop()
    MJ.model = mujoco.MjModel.from_xml_path(xml_path)
    MJ.data = mujoco.MjData(MJ.model)
    MJ.rt = rt
    if display:
        MJ.init_viewer(hz=hz)

def init_mujoco_string(xml_str: str, display=True, rt=True, hz=60):
    if MJ.model is not None or MJ.data is not None:
        MJ.stop()
    MJ.model = mujoco.MjModel.from_xml_string(xml_str)
    MJ.data = mujoco.MjData(MJ.model)
    MJ.rt = rt
    if display:
        MJ.init_viewer(hz=hz)

def init_with_existing_sim(model, data, display=True, rt=True, hz=60):
    if MJ.model is not None or MJ.data is not None:
        MJ.stop()
    MJ.model = model
    MJ.data = data
    MJ.rt = rt
    if display:
        MJ.init_viewer(hz=hz)