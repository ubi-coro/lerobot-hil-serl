from lerobot.robots.viperx import ViperXConfig, ViperX
from lerobot.teleoperators.gello import Gello, GellohaConfig
from lerobot.teleoperators.widowx import WidowXConfig, WidowX
from lerobot.utils.robot_utils import busy_wait

motor = "elbow"

#gello = Gello(GellohaConfig(port="/dev/ttyUSB0"))
#teleop = WidowX(WidowXConfig(port="/dev/ttyDXL_leader_left", id="left"))
robot = ViperX(ViperXConfig(port="/dev/ttyDXL_follower_left", id="left"))

#teleop.connect()
robot.connect()

while True:
    #print("leader", teleop.get_action(), "| follower", robot.get_observation()[f"{motor}.pos"])

    print({key: value / 3.1415 * 180 for key, value in robot.get_observation().items()})

    busy_wait(1.0)


