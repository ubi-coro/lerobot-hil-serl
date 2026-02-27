from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="hil-amp/eval_random_policy",
    root="/media/jstranghoener/Kingston/hil-amp/eval_hilserl_limits/offline-demos/insert/",
)

_from = dataset.episode_data_index["from"]
to = dataset.episode_data_index["to"]
lengths = []

for i in tqdm(range(len(_from)), desc="Retrieving Cycle Times"):
    lengths.append(to[i] - _from[i])

successes = 0
for frame in tqdm(dataset, desc="Success"):
    successes += int(frame["next.reward"])

print(f"Average Duration: {sum(lengths) / len(lengths) / dataset.fps} s")
print(f"Success Rate: {successes / len(lengths) * 100:.1f} %")
