import time
from collections import deque

import matplotlib.pyplot as plt
import torch

from experiments.ur5e_single_test.config import UR5eSingleEnvConfig
from lerobot.processor import create_transition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.robot_utils import busy_wait


# === LIVE PLOT SETUP (NEW) ===
ROLLING_SEC = 10.0
HZ = 30.0
MAXLEN = int(ROLLING_SEC * HZ)

plt.ion()
fig, ax = plt.subplots()
ax.set_title("Gripper action and position (live)")
ax.set_xlabel("time [s]")
ax.set_ylabel("fraction [0..1]")

t0 = time.monotonic()
ts = deque(maxlen=MAXLEN)
acts = deque(maxlen=MAXLEN)
poss = deque(maxlen=MAXLEN)

(action_line,) = ax.plot([], [], label="action (cmd)")   # default colors
(pos_line,) = ax.plot([], [], label="position (obs)")
ax.legend(loc="upper right")

# First draw to enable blitting
fig.canvas.draw()
bg = fig.canvas.copy_from_bbox(ax.bbox)

def live_update(t, a, p):
    # append data
    ts.append(t)
    acts.append(a)
    poss.append(p)

    # update artists
    action_line.set_data(ts, acts)
    pos_line.set_data(ts, poss)

    # manage window: show last ROLLING_SEC
    tmin = max(0.0, ts[-1] - ROLLING_SEC) if ts else 0.0
    tmax = max(ROLLING_SEC, ts[-1]) if ts else ROLLING_SEC
    ax.set_xlim(tmin, tmax)

    # autoscale Y softly
    if acts and poss:
        ymin = min(min(acts), min(poss), 0.0)
        ymax = max(max(acts), max(poss), 1.0)
        pad = 0.05 * (ymax - ymin + 1e-6)
        ax.set_ylim(ymin - pad, ymax + pad)

    # blit
    fig.canvas.restore_region(bg)
    ax.draw_artist(action_line)
    ax.draw_artist(pos_line)
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

# === ENV SETUP ===
env_cfg = UR5eSingleEnvConfig()

env, env_processor, action_processor = env_cfg.make()

while True:
    info = {TeleopEvents.IS_INTERVENTION: True}
    action = torch.tensor([0.0] * env_cfg.action_dim, dtype=torch.float32)
    action_transition = create_transition(action=action, info=info)
    processed_action_transition = action_processor(action_transition)

    obs, reward, terminated, truncated, info = env.step(processed_action_transition[TransitionKey.ACTION])

    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    info.update(processed_action_transition[TransitionKey.INFO].copy())

    # determine which action to store
    if info.get(TeleopEvents.IS_INTERVENTION, False) and TELEOP_ACTION_KEY in complementary_data:
        action_to_record = complementary_data[TELEOP_ACTION_KEY]
    else:
        action_to_record = action_transition[TransitionKey.ACTION]

    transition = create_transition(
        observation=obs,
        action=action_to_record,
        reward=reward + processed_action_transition[TransitionKey.REWARD],
        done=terminated or processed_action_transition[TransitionKey.DONE],
        truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
        info=info,
        complementary_data=processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy(),
    )
    transition = env_processor(transition)

    gripper_action = float(transition[TransitionKey.ACTION][0, -1])
    gripper_pos = float(transition[TransitionKey.OBSERVATION][OBS_STATE][0, -1])

    #print(gripper_action, gripper_pos)

    # === LIVE PLOT UPDATE (NEW) ===
    t_rel = time.monotonic() - t0
    print(float(transition[TransitionKey.OBSERVATION][OBS_STATE][0, 0]), float(transition[TransitionKey.OBSERVATION][OBS_STATE][0, 1]))
    live_update(t_rel, float(transition[TransitionKey.OBSERVATION][OBS_STATE][0, 0]), float(transition[TransitionKey.OBSERVATION][OBS_STATE][0, 1]))

    busy_wait(1/10.0)