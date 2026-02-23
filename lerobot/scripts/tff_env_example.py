import time

import torch

from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.experiments import HAN_Insertion_RLPD_Sparse_NoPriors, HAN_Insertion_RLPD_Sparse
from lerobot.scripts.server.mp_nets import reset_mp_net

mp_net = HAN_Insertion_RLPD_Sparse_NoPriors()
#mp_net = HAN_Insertion_RLPD_Sparse()
current_primitive = mp_net.primitives["insert"]

robot = make_robot_from_config(mp_net.robot)

# full reset at the beginning of each sequence
env = current_primitive.make(mp_net, robot=robot)

while True:
    current_primitive = mp_net.primitives[mp_net.start_primitive]
    obs, info = reset_mp_net(env, mp_net)

    done = False
    while not done:
        start_loop_t = time.perf_counter()

        action = env.action_space.sample()

        next_obs, reward, done, truncated, info = env.step(torch.zeros_like(action))

        if mp_net.fps:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / mp_net.fps - dt_s)