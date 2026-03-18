Goal: pick a single block from a tray with a top-down approach.

Current idea:

- Start with one learnable approach primitive.
- Add a reset path after the first failed rollout.
- Track evaluation success rate against MP-Net version history.
