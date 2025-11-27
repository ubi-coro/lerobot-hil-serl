import requests
import numpy as np
import json_numpy
import torch
import logging
import time
from typing import Dict, Any, List, Tuple
import cv2 # <-- NEW: For fast image processing

# --- Configuration Constants ---
# CRITICAL: This must match the server's endpoint
SERVER_URL = "http://hkn0706:8000/act_batched"
TIMEOUT = 30  # Increased timeout for larger batch requests

# Default dimensions for padding and fallback if server fails
PROPRIO_TARGET_DIM = 20 # Proprioception dimension the server expects
ACTION_SHAPE = (30, 20) # Example shape: (Sequence Length, Action Dim)
EMBEDDING_DIM = 1024    # Example VLM Embedding Dimension (e.g., from Llama/Florence)
INSTRUCTION = "Pick up the something"

# --- Image Constants ---
TARGET_IMAGE_SIZE = (256, 256)

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class VLAClient:
    """
    Optimized VLA client that uses batched inference via a single HTTP request
    to the /act_batched endpoint.
    """
    def __init__(self, server_url: str = SERVER_URL, timeout: int = TIMEOUT):
        self.server_url = server_url
        self.timeout = timeout
        logging.info(f"VLA Client initialized, targeting BATCHED server at: {self.server_url}")

    def _pad_proprio(self, proprio_np: np.ndarray) -> np.ndarray:
        """Pads the proprio vector to the target dimension of 20."""
        current_dim = proprio_np.shape[-1]
        if current_dim < PROPRIO_TARGET_DIM:
            # Create a zero-padded vector
            padding = np.zeros(PROPRIO_TARGET_DIM - current_dim, dtype=proprio_np.dtype)
            return np.concatenate([proprio_np, padding], axis=-1)
        elif current_dim > PROPRIO_TARGET_DIM:
            # Truncate if larger (should not happen in production)
            return proprio_np[:PROPRIO_TARGET_DIM]
        return proprio_np
    
    def _preprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts the image tensor to a channel-last NumPy array (H, W, C)
        and resizes it to (256, 256).
        """
        image_np = image_tensor.cpu().numpy()
        
        # 1. Ensure Channel Last format (H, W, C)
        if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
            # Assuming Channel First (C, H, W), permute to (H, W, C)
            image_np = np.transpose(image_np, (1, 2, 0))

        # 2. Reshape if needed
        current_size = image_np.shape[:2]
        if current_size != TARGET_IMAGE_SIZE:
            # cv2.resize expects (W, H) tuple for dsize
            image_np = cv2.resize(image_np, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Ensure it's 3 channels, even if grayscale was fed in (VLA requirement)
        if image_np.ndim == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
            
        return image_np.astype(np.uint8)


    def get_actions_and_embeddings(
        self, 
        batch: Dict[str, Any], 
        device: torch.device, 
        use_next_state: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sends a batch of observations to the VLA server and returns batched 
        actions (a_beta) and embeddings (z_s).
        """
        state_key = "next_state" if use_next_state else "state"
        
        # We assume proprio comes as [B, D]
        proprio_batch = batch[state_key]['observation.state']
        
        batch_size = proprio_batch.shape[0]
        all_payloads = []

        should_skip_action = use_next_state 

        for i in range(batch_size):
            # Extract and pad proprio
            proprio_np = proprio_batch[i].cpu().numpy()
            proprio_np_padded = self._pad_proprio(proprio_np)

            # --- PREPROCESSING IMAGES ---
            # NOTE: Assuming 'observation.images.side' is image0 and 'observation.images.up' is image1
            image0_np = self._preprocess_image(batch[state_key]['observation.images.side'][i])
            image1_np = self._preprocess_image(batch[state_key]['observation.images.up'][i])
            # ----------------------------

            # Build payload
            payload = {
                "proprio": json_numpy.dumps(proprio_np_padded),
                "language_instruction": INSTRUCTION,
                "image0": json_numpy.dumps(image0_np),
                "image1": json_numpy.dumps(image1_np),
                "domain_id": 0,
                "steps": 10, 
                "skip_action_generation": should_skip_action
            }
            all_payloads.append(payload)

        # --- 2. Send single batched request ---
        try:
            response = requests.post(self.server_url, json=all_payloads, timeout=self.timeout)
            response.raise_for_status()
            results = response.json() # This is a LIST of results
            
            if len(results) != batch_size:
                raise ValueError(f"Server returned {len(results)} items, expected {batch_size}.")

            # 3. Parse the list of results
            vla_actions = [np.array(r["action"], dtype=np.float32) for r in results]
            vla_embeddings = [np.array(r["embedding"], dtype=np.float32) for r in results]
            
        except requests.exceptions.RequestException as e:
            logging.error(f"FATAL BATCH REQUEST FAILED: {e}. Returning zero tensors.")
            action_zeros = np.zeros((batch_size, *ACTION_SHAPE), dtype=np.float32)
            embedding_zeros = np.zeros((batch_size, EMBEDDING_DIM), dtype=np.float32)
            
            vla_action_batch = torch.tensor(action_zeros, dtype=torch.float32, device=device)
            vla_embedding_batch = torch.tensor(embedding_zeros, dtype=torch.float32, device=device)
            return vla_action_batch, vla_embedding_batch
        except ValueError as e:
            logging.error(f"FATAL RESPONSE ERROR: {e}. Returning zero tensors.")
            action_zeros = np.zeros((batch_size, *ACTION_SHAPE), dtype=np.float32)
            embedding_zeros = np.zeros((batch_size, EMBEDDING_DIM), dtype=np.float32)
            
            vla_action_batch = torch.tensor(action_zeros, dtype=torch.float32, device=device)
            vla_embedding_batch = torch.tensor(embedding_zeros, dtype=torch.float32, device=device)
            return vla_action_batch, vla_embedding_batch


        # 4. Convert lists of numpy arrays to a batched torch tensor
        vla_action_batch = torch.tensor(np.array(vla_actions), dtype=torch.float32, device=device)
        vla_embedding_batch = torch.tensor(np.array(vla_embeddings), dtype=torch.float32, device=device)

        return vla_action_batch, vla_embedding_batch

# ----------------------------------------------------------------------
# Demonstration / Benchmark Logic 
# ----------------------------------------------------------------------

def simulate_rlpd_batch(batch_size: int, proprio_dim: int) -> Dict[str, Any]:
    """
    Creates a dummy batch structure resembling the RLPD output, 
    generating images in CHW (Channel First) to test preprocessing.
    """
    
    # Generate images in Channel First (C, H, W) to test the transposition logic
    dummy_image_side = np.random.randint(0, 255, size=(batch_size, 3, 300, 400), dtype=np.uint8)
    dummy_image_up = np.random.randint(0, 255, size=(batch_size, 3, 300, 400), dtype=np.uint8)

    dummy_proprio = np.random.rand(batch_size, proprio_dim).astype(np.float32)
    dummy_instruction = ["Move the gripper to the target position"] * batch_size
    
    # 2. Structure it like the RLPD batch (state, next_state)
    state_dict = {
        'observation.state': torch.from_numpy(dummy_proprio), # [B, D]
        'observation.images.side': torch.from_numpy(dummy_image_side), # [B, C, H, W]
        'observation.images.up': torch.from_numpy(dummy_image_up), # [B, C, H, W]
    }
    # For next state, we use the same shapes/data
    next_state_dict = {
        'observation.state': torch.from_numpy(dummy_proprio), 
        'observation.images.side': torch.from_numpy(dummy_image_side),
        'observation.images.up': torch.from_numpy(dummy_image_up),
    }

    return {
        "state": state_dict,
        "next_state": next_state_dict,
        "instruction": dummy_instruction,
    }

def main():
    # --- Benchmark Settings (matching user snippet) ---
    NUM_REQUESTS = 5
    BATCH_SIZE = 16
    SOURCE_PROPRIO_DIM = 7 
    
    device = torch.device("cpu") # Use CPU for local test if CUDA is not available
    client = VLAClient()
    
    print(f"--- Batched Client Demonstration ---")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Source Proprio Dim: {SOURCE_PROPRIO_DIM} (Will be padded to {PROPRIO_TARGET_DIM})")
    print(f"Target Image Size: {TARGET_IMAGE_SIZE}")
    print(f"Target Server: {client.server_url}")

    # 1. Simulate the data loading step
    dummy_batch = simulate_rlpd_batch(BATCH_SIZE, SOURCE_PROPRIO_DIM)

    # 2. Simulate the warm-up (using next_state=False)
    print("\nSimulating warm-up request (Action Generation: ON)...")
    try:
        # This call should include action generation
        client.get_actions_and_embeddings(dummy_batch, device, use_next_state=False)
        print("Warm-up complete. Starting benchmark.")
    except Exception as e:
        print(f"⚠️ Initial request failed: {e}")
        print("Please ensure the VLA server is running and accessible.")
        return

    # 3. Main Benchmark Loop
    timings_ms = []
    print("-" * 30)

    for i in range(NUM_REQUESTS):
        start_time = time.perf_counter()
        
        # Simulate two server calls for the RL transition (s and s')
        a_s, e_s = client.get_actions_and_embeddings(dummy_batch, device, use_next_state=False)
        a_s_prime, e_s_prime = client.get_actions_and_embeddings(dummy_batch, device, use_next_state=True)

        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # Total time for s and s' batched calls
        timings_ms.append(elapsed_ms)
        
        total_samples = BATCH_SIZE * 2
        time_per_sample = elapsed_ms / total_samples
        
        logging.info(f"✅ Request {i+1}/{NUM_REQUESTS}: "
                     f"Total time (s + s'): {elapsed_ms:.2f} ms | "
                     f"Avg. per Sample: {time_per_sample:.2f} ms")
        
        # Verify output shapes and logic
        assert a_s.shape == (BATCH_SIZE, *ACTION_SHAPE), f"a_s shape mismatch: {a_s.shape}"
        assert e_s.shape == (BATCH_SIZE, EMBEDDING_DIM), f"e_s shape mismatch: {e_s.shape}"

        # In skip mode (a_s_prime), the action should theoretically be all zeros.
        if np.allclose(a_s_prime.cpu().numpy(), 0):
            logging.info("  Validation: a_s_prime is all zeros (Skip successful).")
        else:
            logging.warning("  Validation: a_s_prime is NOT all zeros (Skip may have failed).")


    print("-" * 30)

    # 4. Calculate and Print Statistics
    if timings_ms:
        timings_np = np.array(timings_ms)
        print("📊 Benchmark Statistics (Total Time for s and s' calls):")
        print(f"  Total successful requests: {len(timings_np)}")
        print(f"  Average Roundtrip Time: {np.mean(timings_np):.2f} ms")
        
        per_item_times = timings_np / (BATCH_SIZE * 2)
        print("\n  --- Final Per-Item Statistics ---")
        print(f"  Average time per individual sample: {np.mean(per_item_times):.2f} ms")
    else:
        print("No successful requests were timed.")

if __name__ == "__main__":
    main()