import torch

# from experiments.aloha_bimanual_v2.config import AlohaBimanualEnvConfigV2
from lerobot.processor import create_transition, TransitionKey
from lerobot.rl.gym_manipulator import step_env_and_process_transition
from lerobot.robots.viperx import ViperXConfig, ViperX
from lerobot.teleoperators import TeleopEvents
#from lerobot.teleoperators.gello import Gello
from lerobot.teleoperators.widowx import WidowXConfig, WidowX
from lerobot.utils.constants import OBS_STATE
from lerobot.utils.robot_utils import busy_wait

motor = "elbow"

#gello = Gello(GellohaConfig(port="/dev/ttyUSB0"))
teleop = WidowX(WidowXConfig(port="/dev/ttyDXL_leader_left", id="left"))
#robot = ViperX(ViperXConfig(port="/dev/ttyDXL_follower_left", id="left"))

cfg = AlohaBimanualEnvConfigV2()
env, env_processor, action_processor = cfg.make()

#teleop.connect()
#robot.connect()

obs, info = env.reset()
env_processor.reset()
action_processor.reset()

env.robot_dict["left"].bus.disable_torque()

while True:
    #print("leader", teleop.get_action(), "| follower", robot.get_observation()[f"{motor}.pos"])

    #print({key: value / 3.1415 * 180 for key, value in robot.get_observation().items()})

    obs = env._get_observation()
    print("ROS obs:", obs["agent_pos"])

    obs = env_processor(data=create_transition(observation=obs))[TransitionKey.OBSERVATION]
    print("old obs:", obs[OBS_STATE])

    action = obs[OBS_STATE]
    action_transition = action_processor(create_transition(action=action))
    print("ROS action:", action_transition[TransitionKey.ACTION])

    busy_wait(1.0)


