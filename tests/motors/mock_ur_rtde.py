# fake_rtde.py

import time
import threading
import numpy as np
from collections import defaultdict

# -----------------------------------------------------------------------------
# Shared mock state for each (hostname) “robot”
# -----------------------------------------------------------------------------
#
# We keep a dict mapping hostname → a simple state object:
#   {
#       'pose':  np.array([x, y, z, Rx, Ry, Rz]),
#       'vel':   np.array([vx, vy, vz, wx, wy, wz]),
#       'wrench':np.array([Fx, Fy, Fz, Mx, My, Mz]),
#       'mode':  'idle' or 'force',
#       'lock':  threading.Lock()
#   }
#
# Whenever a ControlInterface sends a command (servoL, speedL, or forceMode),
# it updates that state (pose/vel/wrench).  The ReceiveInterface simply reads it.
#
# A background thread per “robot” advances pose based on the last set velocity,
# so that continuous speedL commands produce smooth motion over time.

_mock_states = {}
_states_lock = threading.Lock()


def _get_or_create_state(hostname):
    """Return the shared state dict for this hostname, creating it if needed."""
    with _states_lock:
        if hostname not in _mock_states:
            # Initialize at origin, zero velocity/force
            st = {
                'pose':   np.zeros(6, dtype=float),   # [x,y,z, Rx,Ry,Rz]
                'vel':    np.zeros(6, dtype=float),   # [vx,vy,vz, wx,wy,wz]
                'wrench': np.zeros(6, dtype=float),   # [Fx,Fy,Fz, Mx,My,Mz]
                'mode':   'idle',                     # 'idle' or 'force'
                'lock':   threading.Lock(),
            }
            # Start a background “integrator” thread that moves pose = pose + vel*dt
            st['run_flag'] = True
            st['thread'] = threading.Thread(
                target=_background_integrator, args=(hostname,), daemon=True
            )
            st['thread'].start()
            _mock_states[hostname] = st
        return _mock_states[hostname]


def _background_integrator(hostname):
    """
    Periodically (at ~200 Hz) update pose = pose + vel * dt
    based on the shared state’s 'vel'.  This simulates continuous motion.
    """
    RATE = 200.0   # Hz
    dt = 1.0 / RATE
    state = _mock_states[hostname]
    while state['run_flag']:
        t0 = time.time()
        with state['lock']:
            # Integrate translational and rotational velocities into pose
            # pose[:3] += vel[:3] * dt
            state['pose'][:3] += state['vel'][:3] * dt
            # For simplicity, keep rotations fixed since we don’t simulate orientation dynamics
            # state['pose'][3:] += state['vel'][3:] * dt
            # We’ll ignore Rx,Ry,Rz updates in this mock
        elapsed = time.time() - t0
        to_sleep = dt - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)


# -----------------------------------------------------------------------------
# Fake RTDEControlInterface
# -----------------------------------------------------------------------------
class RTDEControlInterface:
    """
    A fake RTDEControlInterface that simulates sending commands to a TF_UR.
    Methods update a shared “robot state” internally so that RTDEReceiveInterface
    can later return matching pose/velocity/force.
    """

    def __init__(self, hostname: str = None, *args):
        self.hostname = hostname or 'default'
        self._state = _get_or_create_state(self.hostname)
        # Keep a last‐wrench buffer if in forceMode
        with self._state['lock']:
            self._state['wrench'] = np.zeros(6, dtype=float)
            self._state['mode'] = 'idle'

    def initPeriod(self):
        """
        In real TF_UR, this returns a timestamp token for synchronization.
        Here we just return the current time.
        """
        return time.time()

    def waitPeriod(self, t0):
        """
        In real TF_UR, blocks until the next RTDE tick.  Here, we sleep
        so that our loop runs at ~125 Hz by default.  If dt between
        commands is small, this will sleep a little; otherwise it does nothing.
        """
        RATE = 125.0
        dt = 1.0 / RATE
        elapsed = time.time() - t0
        to_sleep = dt - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)

    def setTcp(self, tcp_pose):
        """
        Pretend to set the TCP offset.  We do nothing here.
        """
        pass

    def setPayload(self, mass, cog=None):
        """
        Pretend to set the payload mass/COG.  Always succeed.
        """
        return True

    def moveJ(self, joints, speed, acceleration):
        """
        Pretend to execute a joint move.  We’ll sleep a fixed 0.1 s
        and leave pose unchanged (since we don’t simulate joint→TCP).
        """
        time.sleep(0.1)
        return True

    def servoL(self, pose, vel, acc, dt, lookahead_time, gain):
        """
        In real TF_UR, this commands a single Cartesian pose at the next tick.
        Here, we immediately set the shared pose to `pose`, zero velocity,
        and return True.
        """
        with self._state['lock']:
            self._state['pose'] = np.array(pose, dtype=float)
            self._state['vel'] = np.zeros(6, dtype=float)
        return True

    def speedL(self, speed6, acc, dt):
        """
        In real TF_UR, this commands a Cartesian velocity for `dt` seconds.
        In our mock, we update the shared state’s velocity to `speed6`
        so that the background integrator moves pose += vel*dt.
        """
        speed6 = np.array(speed6, dtype=float)
        with self._state['lock']:
            self._state['vel'] = speed6.copy()
        # Immediately integrate one step so that a subsequent getActualTCPPose()
        # will be slightly updated.  (Background integrator also runs concurrently.)
        with self._state['lock']:
            self._state['pose'][:3] += speed6[:3] * dt
            # We ignore rotational integration for simplicity.
        return True

    def forceMode(self, selection_vector, wrench, type=None, limits=None, mask=None):
        """
        In real TF_UR, engage force control.  We record the requested wrench so
        that getActualTCPForce() can return it.  Also set mode = 'force'.
        """
        wrench = np.array(wrench, dtype=float)
        with self._state['lock']:
            self._state['mode'] = 'force'
            self._state['wrench'] = wrench.copy()
        return True

    def forceModeStop(self):
        """
        Exit force mode: set mode back to 'idle' and zero wrench.
        """
        with self._state['lock']:
            self._state['mode'] = 'idle'
            self._state['wrench'] = np.zeros(6, dtype=float)

    def speedStop(self):
        """
        Stop any speedL motion: zero out the velocity.
        """
        with self._state['lock']:
            self._state['vel'] = np.zeros(6, dtype=float)

    def servoStop(self):
        """
        Stop any servoL motion: zero out the velocity.
        """
        with self._state['lock']:
            self._state['vel'] = np.zeros(6, dtype=float)

    def stopScript(self):
        """
        In real TF_UR, this would stop the RTDE script.  Here we do nothing.
        """
        pass

    def disconnect(self):
        """
        In real TF_UR, this would close sockets.  Here we do nothing.
        """
        pass


# -----------------------------------------------------------------------------
# Fake RTDEReceiveInterface
# -----------------------------------------------------------------------------
class RTDEReceiveInterface:
    """
    A fake RTDEReceiveInterface that returns data from the shared “robot state”
    for the given hostname.  That state gets updated whenever the fake
    RTDEControlInterface sends servoL, speedL, or forceMode commands.
    """

    def __init__(self, hostname: str = None):
        self.hostname = hostname or 'default'
        self._state = _get_or_create_state(self.hostname)

    def getActualTCPPose(self):
        """
        Return the current [x, y, z, Rx, Ry, Rz] from shared state.
        In this mock, Rx,Ry,Rz stay zero.
        """
        with self._state['lock']:
            return self._state['pose'].copy()

    def getActualTCPSpeed(self):
        """
        Return the current [vx, vy, vz, ωx, ωy, ωz] from shared state.
        """
        with self._state['lock']:
            return self._state['vel'].copy()

    def getActualTCPForce(self):
        """
        Return the last commanded wrench if in force mode, or zero otherwise.
        """
        with self._state['lock']:
            return self._state['wrench'].copy()

    def getActualQ(self):
        """
        Return a dummy 6‐element joint configuration.  We’ll just return zeros.
        """
        return np.zeros(6, dtype=float)

    def getActualQd(self):
        """
        Return a dummy 6‐element joint velocity.  We’ll return zeros.
        """
        return np.zeros(6, dtype=float)

    def disconnect(self):
        """
        Do nothing.
        """
        pass
