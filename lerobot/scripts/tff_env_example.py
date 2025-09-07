from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.experiments import UR3_HAN_Insertion
from lerobot.scripts.server.mp_nets import reset_mp_net

mp_net = UR3_HAN_Insertion()
current_primitive = mp_net.primitives[mp_net.start_primitive]
robot = make_robot_from_config(mp_net.robot)

# full reset at the beginning of each sequence
env = current_primitive.make(mp_net, robot=robot)
obs, info = reset_mp_net(env, mp_net)

