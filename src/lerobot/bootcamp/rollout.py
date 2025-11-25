import gymnasium as gym
import torch
import numpy as np
import random
from collections import defaultdict
from PIL import Image
import requests # For talking to the server
import base64
import io

from lerobot.common.policies.sac.sac_policy import SACPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import push_to_hub
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

# --- Constants ---

# --- Server & Data Config ---
VLA_SERVER_URL = "http://YOUR_SERVER_IP:8000/predict"
DATA_REPO_ID = "YOUR_HF_USERNAME/my-bootcamp-rollouts"
TASK_INSTRUCTION = "Fold the hoodie."

# --- Model Dimensions ---
# CRITICAL: This MUST match the VLA server's embedding dim
VLA_EMBED_DIM = 4096

# --- Policy Checkpoints ---
RESIDUAL_POLICY_CHECKPOINT = "./models/sac_policy_epoch_k.pth"
REWARD_CLASSIFIER_CHECKPOINT = None # "./models/reward_classifier.pth"

# --- Exploration Parameters ---
NUM_ROLLOUTS = 50
EXPLORE_PROBABILITY = 0.2
EXPLOIT_TEMPERATURE = 0.01  # Low temp for VLA
EXPLORE_TEMPERATURE = 1.2   # High temp for VLA
EXPLOIT_NOISE_STD = 0.1     # Small noise on the *residual*

# --- Env Config ---
ENV_ID = "Reacher-v4" # Placeholder: Your env MUST have `render("rgb_array")`

def get_vla_output(image_np, instruction, temperature):
    """
    NEW: Calls the X-VLA server via HTTP.

    Args:
        image_np: A (H, W, C) numpy array from env.render()
        instruction: The task string
        temperature: The sampling temperature for the VLA

    Returns:
        (a_beta, z_s): A (action_dim,) numpy array and a (VLA_EMBED_DIM,) numpy array
    """
    # 1. Encode image to base64
    pil_image = Image.fromarray(image_np)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    # 2. Send request
    payload = {
        "image": image_b64,
        "instruction": instruction,
        "temperature": temperature,
        "do_sample": True
    }
    try:
        response = requests.post(VLA_SERVER_URL, json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"FATAL: Could not connect to VLA server at {VLA_SERVER_URL}.")
        print(f"Error: {e}")
        print("Is the server script 'xvla_server.py' running?")
        exit()

    # 3. Decode response
    data = response.json()
    action_beta = np.array(data['action'], dtype=np.float32)
    embedding_zs = np.array(data['embedding'], dtype=np.float32)

    return action_beta, embedding_zs

def load_mlp_policy(checkpoint_path, device, obs_space, act_space, policy_class):
    """
    NEW: Loads a *lightweight MLP* SAC policy or Classifier.
    The `obs_space` it expects is now just a state vector (the embedding).
    """
    print(f"Loading {policy_class.__name__} from {checkpoint_path}...")

    policy = policy_class(
        obs_space=obs_space, # This is now {"observation.state": Box(4096,)}
        act_space=act_space,
        # This will create an MLP, not a CNN, because "image" is not in obs_space
        hidden_dims=[512, 512], # Bigger MLP as embeddings are dense
        device=device,
    )

    try:
        policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Successfully loaded pre-trained {policy_class.__name__}.")
    except FileNotFoundError:
        print(f"WARNING: No checkpoint found at {checkpoint_path}.")
        print(f"Running with a RANDOMLY INITIALIZED {policy_class.__name__}.")

    policy.eval()
    return policy

def format_embedding_for_sac(embedding_np, device):
    """
    NEW: Formats the VLA embedding vector for the `lerobot` SAC policy.
    """
    # 1. Convert to tensor, add batch dim, and move to device
    embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32, device=device).unsqueeze(0)

    # 2. Wrap in dict
    # We use "observation.state" as the key, which triggers the MLP path
    obs_dict = {"observation.state": embedding_tensor}
    return obs_dict

def unwrap_action_from_policy(action_tensor):
    return action_tensor.squeeze(0).cpu().numpy()

def collect_rollout(env, residual_policy, reward_classifier, device):
    """
    Collects a single episode rollout using the client-server setup.
    """
    episode_data = []
    obs_flat, info = env.reset()
    obs_image_np = env.render("rgb_array")

    terminated = truncated = False

    is_explore_mode = random.random() < EXPLORE_PROBABILITY

    if is_explore_mode:
        print("  Running in 'Explore' mode (high-temp VLA)...")
    else:
        print("  Running in 'Exploit' mode (VLA + Residual)...")

    while not (terminated or truncated):
        # 1. Get VLA base action (a_beta) AND embedding (z_s) from server
        vla_temp = EXPLORE_TEMPERATURE if is_explore_mode else EXPLOIT_TEMPERATURE
        action_beta, embedding_zs = get_vla_output(obs_image_np, TASK_INSTRUCTION, vla_temp)

        # 2. Format embedding for SAC policy
        sac_obs_dict = format_embedding_for_sac(embedding_zs, device)

        # 3. Get residual action (delta_a)
        with torch.no_grad():
            if is_explore_mode:
                # --- Mode A: "Bad Data" Exploration ---
                # We use the high-temp VLA action directly.
                # No residual is added.
                final_action = action_beta
            else:
                # --- Mode B: "Good Data" Exploitation ---
                residual_action_tensor = residual_policy.select_action(sac_obs_dict)
                residual_action = unwrap_action_from_policy(residual_action_tensor)

                final_action = action_beta + residual_action

                # Add small decorrelation noise *to the residual*
                noise = np.random.randn(*final_action.shape) * EXPLOIT_NOISE_STD
                final_action += noise

        # 4. Step the environment
        next_obs_flat, env_reward, terminated, truncated, info = env.step(final_action)
        next_obs_image_np = env.render("rgb_array")

        # 5. --- GET THE REWARD ---
        if reward_classifier is not None:
            with torch.no_grad():
                # We pass the *embedding* (z_s) and the *final action*
                action_tensor = torch.tensor(final_action, device=device).unsqueeze(0)
                reward_tensor = reward_classifier(sac_obs_dict, action_tensor)
                reward = reward_tensor.squeeze().cpu().item()
        else:
            reward = env_reward
        # ---------------------------------

        # 6. --- GET NEXT STATE EMBEDDING (z_s_prime) ---
        # This is the second server call, needed for the (s,a,r,s') transition
        # We *always* get this embedding with a low temperature
        # to have a consistent "next_state" representation.
        _ , embedding_zs_prime = get_vla_output(next_obs_image_np, TASK_INSTRUCTION, EXPLOIT_TEMPERATURE)

        # 7. Store the data frame (embeddings, not images)
        frame = {
            "observation.state": embedding_zs,        # <-- VLA embedding
            "next_observation.state": embedding_zs_prime, # <-- Next VLA embedding
            "action": final_action,
            "reward": np.array([reward]),
            "terminated": np.array([terminated]),
            "truncated": np.array([truncated]),
        }
        episode_data.append(frame)

        obs_flat = next_obs_flat
        obs_image_np = next_obs_image_np

    print(f"  ...finished rollout after {len(episode_data)} steps.")
    return episode_data

# ... (format_rollouts_for_lerobot_dataset remains the same) ...
def format_rollouts_for_lerobot_dataset(all_rollout_data):
    print("Formatting data for LeRobotDataset...")
    dataset_dict = defaultdict(list)
    for episode_data in all_rollout_data:
        episode_index = dataset_dict["episode_index"].max() + 1 if dataset_dict["episode_index"] else 0
        for frame in episode_data:
            for key, value in frame.items():
                dataset_dict[key].append(value)
            dataset_dict["episode_index"].append(episode_index)
    return dict(dataset_dict)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Rollout Client using device: {device}")

    # --- 1. Load Env & Define Spaces ---
    env = gym.make(ENV_ID, render_mode="rgb_array")

    # CRITICAL: Define the observation space for your *MLP* policies
    obs_space = gym.spaces.Dict({
        "observation.state": gym.spaces.Box(
            -np.inf, np.inf,
            shape=(VLA_EMBED_DIM,),
            dtype=np.float32
        )
    })
    act_space = env.action_space

    # --- 2. Load Lightweight Policies ---
    residual_policy = load_mlp_policy(
        RESIDUAL_POLICY_CHECKPOINT, device, obs_space, act_space, policy_class=SACPolicy
    )

    reward_classifier = None
    if REWARD_CLASSIFIER_CHECKPOINT:
        try:
            reward_classifier = load_mlp_policy(
                REWARD_CLASSIFIER_CHECKPOINT, device, obs_space, act_space, policy_class=Classifier
            )
        except Exception as e:
            print(f"Error loading reward classifier: {e}")
            print("Defaulting to environment rewards.")
    else:
        print("No reward classifier checkpoint specified. Using environment rewards.")

    # --- 3. Collect rollouts ---
    all_rollout_data = []
    for i in range(NUM_ROLLOUTS):
        print(f"--- Collecting rollout {i+1} / {NUM_ROLLOUTS} ---")
        episode_data = collect_rollout(env, residual_policy, reward_classifier, device)
        all_rollout_data.append(episode_data)

    env.close()

    if not all_rollout_data:
        print("No data collected. Exiting.")
        return

    dataset_dict = format_rollouts_for_lerobot_dataset(all_rollout_data)

    # The dataset now contains `observation.state` (embeddings)
    dataset = LeRobotDataset(dataset_dict)

    print(f"\nPushing dataset to {DATA_REPO_ID}...")
    push_to_hub(
        repo_id=DATA_REPO_ID,
        dataset=dataset,
        hf_token=None,
        commit_message="Add epoch K rollout data (X-VLA Embeddings + SAC)"
    )
    print("Done!")

if __name__ == "__main__":
    main()
