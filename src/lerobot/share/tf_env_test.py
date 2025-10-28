import torch

from experiments.ur5e_single_test.config import UR5eSingleEnvConfig
from lerobot.processor import create_transition, TransitionKey
from lerobot.processor.hil_processor import TELEOP_ACTION_KEY
from lerobot.teleoperators import TeleopEvents
from lerobot.utils.robot_utils import busy_wait

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
        action=action,
        reward=reward + processed_action_transition[TransitionKey.REWARD],
        done=terminated or processed_action_transition[TransitionKey.DONE],
        truncated=truncated or processed_action_transition[TransitionKey.TRUNCATED],
        info=info,
        complementary_data=processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy(),
    )
    transition = env_processor(transition)

    print(transition[TransitionKey.OBSERVATION])

    busy_wait(1/1.0)