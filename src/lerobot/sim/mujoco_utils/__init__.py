import os
os.environ['MUJOCO_GL'] = 'egl'

from .sim_singleton import SimManager, SimSession
from .viewer import ViewerRegistry, AbstractViewer, MujocoViewer, CVViewer, MujocoCameraViewer