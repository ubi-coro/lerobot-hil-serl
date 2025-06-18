import time

from lerobot.experiments.nist_peg_in_hole_c_rot import URPegInHoleCRotConfig

config = URPegInHoleCRotConfig()
env = config.make()

num_episodes = 0
while num_episodes < config.num_episodes:
    print("Perform reset")
    env.reset()
    print("Start episode")

    done = False
    sum_of_rewards = 0.0
    while not done:
        t_start = time.perf_counter()

        action = env.action_space.sample() * 0

        obs, reward, terminated, truncated, info = env.step(action)

        sum_of_rewards += reward
        done = terminated | truncated

        #print(obs["observation.main_eef_wrench"])
        #print(obs["observation.main_eef_pos"][2])
        #print(len(obs["state"]))

        t_loop = time.perf_counter()- t_start
        time.sleep(max([0, 1 / config.fps - t_loop]))

    print(f"Finished episode, got {sum_of_rewards}")
    time.sleep(1.0)




