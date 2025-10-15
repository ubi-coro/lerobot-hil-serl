import os
from huggingface_hub import snapshot_download

# ⚙️ 1. Download the *real* Physical-Intelligence/OpenPI* checkpoint
# (not the LeRobot port)
path = snapshot_download(
    repo_id="Physical-Intelligence/openpi",
    repo_type="model",
    allow_patterns=["**assets/**"],  # just pull the assets directory
)

print(f"Downloaded to {path}")

# ⚙️ 2. Locate trossen normalization stats
trossen_stats = []
for root, _, files in os.walk(path):
    for f in files:
        if "trossen" in f and ("norm" in f or "stats" in f):
            trossen_stats.append(os.path.join(root, f))

print("Found normalization files:")
for f in trossen_stats:
    print("  ", f)

# ⚙️ 3. Optionally inspect one (often JSON or npz)
import json, numpy as np

for f in trossen_stats:
    if f.endswith(".json"):
        print("\n---", f, "---")
        print(json.dumps(json.load(open(f)), indent=2))
    elif f.endswith(".npz"):
        arrs = np.load(f)
        print("\n---", f, "---")
        for k in arrs:
            print(k, arrs[k].shape, arrs[k].mean(), arrs[k].std())