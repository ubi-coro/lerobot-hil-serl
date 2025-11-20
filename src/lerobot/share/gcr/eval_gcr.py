import os
from pathlib import Path

import torch
import clip
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
import torchvision.transforms as T

from lerobot.configs.policies import PreTrainedConfig
#from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.gui_dataset import LeRobotDataset
from lerobot.policies.gcr.configuration_grc import GCRConfig
from lerobot.policies.gcr.modeling_gcr import GCR


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

CACHE_DIR = "gcr_eval_plots"
CACHE_PATH = os.path.join(CACHE_DIR, "hoodie_episode0_cache.pt")
POLICY_PATH = "/media/internal/nvme/lerobot/models/hoodie_folding/gcr-201125-v3/checkpoints/000500/pretrained_model"


def make_preprocess():
    return T.Compose(
        [
            T.Resize(256, antialias=None),
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float32),  # uint8 -> float
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ]
    )


def collect_first_episode(dataset, camera_key: str):
    """
    Returns:
        imgs: Tensor [T, 3, 224, 224] (preprocessed, CLIP-normalized)
        raw_imgs: Tensor [T, 3, H, W] in [0,1] for visualization
        task: str
    """
    preprocess = make_preprocess()

    first_frame = dataset[0]
    initial_ep_idx = int(first_frame["episode_index"])
    task = first_frame.get("task", "")

    episode_imgs = []
    episode_raw = []

    idx = 0
    while True:
        frame = dataset[idx]
        if int(frame["episode_index"]) != initial_ep_idx:
            break

        img = frame[camera_key]  # C x H x W
        if img.dtype == torch.uint8:
            raw = img.float() / 255.0
        else:
            raw = img.clone()

        episode_raw.append(raw)

        proc = preprocess(raw)
        episode_imgs.append(proc)

        idx += 1

    imgs = torch.stack(episode_imgs, dim=0)      # [T, 3, 224, 224]
    raw_imgs = torch.stack(episode_raw, dim=0)   # [T, 3, H, W]
    return imgs, raw_imgs, task


def compute_distances_liv(liv_model, imgs: torch.Tensor, task: str, device: str = "cuda"):
    liv_model.eval()
    imgs = imgs.to(device)

    with torch.no_grad():
        embeddings = liv_model(input=imgs, modality="vision")  # [T, D]
        goal_emb_img = embeddings[-1]

        token = clip.tokenize([task]).to(device)
        goal_emb_txt = liv_model(input=token, modality="text")[0]

    distances_img = []
    distances_txt = []

    with torch.no_grad():
        for t in range(embeddings.shape[0]):
            cur = embeddings[t]
            sim_img = liv_model.sim(goal_emb_img, cur)
            sim_txt = liv_model.sim(goal_emb_txt, cur)
            distances_img.append((-sim_img).item())
            distances_txt.append((-sim_txt).item())

    return np.array(distances_img), np.array(distances_txt)


def plot_rewards_and_gif(
    distances_img,
    distances_txt,
    raw_imgs,
    task: str,
    fig_prefix: str = "gcr_eval",
    animated: bool = True,
):
    T_len = len(distances_img)
    frames = np.arange(T_len)

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(24, 6))

    ax[0].plot(frames, distances_img, color="tab:blue", label="image", linewidth=3)
    ax[0].plot(frames, distances_txt, color="tab:red", label="text", linewidth=3)
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    ax[0].set_title(f"Language Goal: {task}", fontsize=15)

    ax[1].plot(frames, distances_img, color="tab:blue", linewidth=3)
    ax[1].set_xlabel("Frame", fontsize=15)
    ax[1].set_title("Image Goal", fontsize=15)

    ax[2].plot(frames, distances_txt, color="tab:red", linewidth=3)
    ax[2].set_xlabel("Frame", fontsize=15)
    ax[2].set_title(f"Language Goal: {task}", fontsize=15)

    ax[3].imshow(raw_imgs[-1].permute(1, 2, 0).cpu().numpy())
    ax[3].axis("off")

    for i in range(3):
        y_range = np.diff(ax[i].get_ylim())[0]
        if y_range != 0:
            asp = 1
            ax[i].set_aspect(
                asp * np.diff(ax[i].get_xlim())[0] / y_range
            )

    png_name = f"{fig_prefix}.png"
    fig.savefig(png_name, bbox_inches="tight")
    print(f"Saved static plot to {png_name}")

    if not animated:
        plt.close(fig)
        return

    ax0_xlim = ax[0].get_xlim()
    ax0_ylim = ax[0].get_ylim()
    ax1_xlim = ax[1].get_xlim()
    ax1_ylim = ax[1].get_ylim()
    ax2_xlim = ax[2].get_xlim()
    ax2_ylim = ax[2].get_ylim()

    def animate(i):
        i = min(i, T_len - 1)
        for a in ax:
            a.clear()

        ax[0].plot(frames[: i + 1], distances_img[: i + 1], color="tab:blue", linewidth=3)
        ax[0].plot(frames[: i + 1], distances_txt[: i + 1], color="tab:red", linewidth=3)
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel("Frame", fontsize=15)
        ax[0].set_ylabel("Embedding Distance", fontsize=15)
        ax[0].set_title(f"Language Goal: {task}", fontsize=15)
        ax[0].set_xlim(ax0_xlim)
        ax[0].set_ylim(ax0_ylim)

        ax[1].plot(frames[: i + 1], distances_img[: i + 1], color="tab:blue", linewidth=3)
        ax[1].set_xlabel("Frame", fontsize=15)
        ax[1].set_title("Image Goal", fontsize=15)
        ax[1].set_xlim(ax1_xlim)
        ax[1].set_ylim(ax1_ylim)

        ax[2].plot(frames[: i + 1], distances_txt[: i + 1], color="tab:red", linewidth=3)
        ax[2].set_xlabel("Frame", fontsize=15)
        ax[2].set_title(f"Language Goal: {task}", fontsize=15)
        ax[2].set_xlim(ax2_xlim)
        ax[2].set_ylim(ax2_ylim)

        ax[3].imshow(raw_imgs[i].permute(1, 2, 0).cpu().numpy())
        ax[3].axis("off")

        return ax

    final_freeze = 30
    gif_name = f"{fig_prefix}.gif"
    ani = FuncAnimation(
        fig,
        animate,
        interval=40,
        repeat=False,
        frames=T_len + final_freeze,
    )
    ani.save(gif_name, dpi=100, writer=PillowWriter(fps=25))
    print(f"Saved GIF to {gif_name}")
    plt.close(fig)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Either load cached episode or build it once
    if os.path.exists(CACHE_PATH):
        cache = torch.load(CACHE_PATH, map_location="cpu")
        imgs = cache["imgs"]        # [T, 3, 224, 224]
        raw_imgs = cache["raw_imgs"]  # [T, 3, H, W]
        task = cache["task"]
        camera_key = cache["camera_key"]
        print(f"Loaded cached episode from {CACHE_PATH}")
    else:
        print("No cache found, loading dataset and extracting episode...")
        dataset = LeRobotDataset(
            root="/media/internal/nvme/lerobot/data/hoodie_folding_v3/base",
            repo_id="hoodie_folding/base",
            video_backend="pyav",
        )

        cfg_temp = PreTrainedConfig.from_pretrained(POLICY_PATH)
        kwargs = {"config": cfg_temp, "pretrained_name_or_path": Path(POLICY_PATH)}
        tmp_policy = GCR.from_pretrained(**kwargs)

        camera_key = cfg_temp.camera_key or tmp_policy.img_key

        imgs, raw_imgs, task = collect_first_episode(dataset, camera_key)

        torch.save(
            {
                "imgs": imgs.cpu(),
                "raw_imgs": raw_imgs.cpu(),
                "task": task,
                "camera_key": camera_key,
            },
            CACHE_PATH,
        )
        print(f"Saved episode cache to {CACHE_PATH}")

    # 2) Load GCR with pretrained LIV
    kwargs = {"config": PreTrainedConfig.from_pretrained(POLICY_PATH), "pretrained_name_or_path": Path(POLICY_PATH)}
    gcr = GCR.from_pretrained(**kwargs).to(device)
    liv = gcr.model  # underlying LIV encoder

    # 3) Compute distances
    distances_img, distances_txt = compute_distances_liv(liv, imgs, task, device=device)

    # 4) Plot
    prefix = os.path.join(CACHE_DIR, "hoodie_episode0_gcr_zeroshot")
    plot_rewards_and_gif(distances_img, distances_txt, raw_imgs, task, fig_prefix=prefix, animated=True)


if __name__ == "__main__":
    main()
