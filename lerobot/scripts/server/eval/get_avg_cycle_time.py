from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="hil-amp/eval_random_policy",
    root="/home/jannick/data/paper/hil-amp/eval_random_policy/offline-demos/insert/",
)

_from = dataset.episode_data_index["from"]
to = dataset.episode_data_index["to"]
lengths = []

for i in tqdm(range(len(_from)), desc="Retrieving Cycle Times"):
    lengths.append(to[i] - _from[i])

print(f"{sum(lengths) / len(lengths) / dataset.fps} s")