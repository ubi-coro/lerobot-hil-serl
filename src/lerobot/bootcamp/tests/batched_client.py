import requests
import numpy as np
import json_numpy
import time

# --- MODIFIED: Point to the new batched endpoint ---
server_url = "http://hkn0706:8000/act_batched"
timeout = 30  # <-- Increased timeout for larger batch
num_requests = 5  # Number of *batched* requests to send
BATCH_SIZE = 2    # <-- Number of samples per batch

# Prepare inputs (using 20-dim proprio from your custom code)
proprio_np = np.zeros(20, dtype=np.float32) 
image_np = np.zeros((256, 256, 3), dtype=np.uint8)
instruction = "Move the gripper to the target position"

# --- MODIFIED: Create a *list* of payloads ---
print(f"Preparing a batch of {BATCH_SIZE} payloads...")
single_payload = {
    "proprio": json_numpy.dumps(proprio_np),
    "language_instruction": instruction,
    "image0": json_numpy.dumps(image_np),
    "domain_id": 0,
    "steps": 10
}
# Create a list by repeating the same payload
list_of_payloads = [single_payload] * BATCH_SIZE

timings_ms = []  # <-- List to store all timings
print(f"🚀 Sending {num_requests} batched requests (size {BATCH_SIZE}) to {server_url}...")

# --- Optional: Warm-up request ---
try:
    print("Sending warm-up batched request...")
    # Send the whole list as the JSON body
    requests.post(server_url, json=list_of_payloads, timeout=timeout)
    print("Warm-up complete. Starting benchmark.")
except Exception as e:
    print(f"⚠️ Warm-up request failed: {e}. Aborting.")
    print("   (Is the 'xvla_server_batched.py' server running?)")
    exit()

print("-" * 30)

# --- Main Test Loop ---
for i in range(num_requests):
    start_time = time.perf_counter()
    
    try:
        # --- MODIFIED: Send the entire list as the JSON payload ---
        response = requests.post(server_url, json=list_of_payloads, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        
        # The result is now a LIST of responses
        results = response.json()
        
        # --- Stop Timer ---
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        timings_ms.append(elapsed_ms)
        
        # Calculate per-item time
        time_per_item = elapsed_ms / BATCH_SIZE
        
        print(f"✅ Request {i+1}/{num_requests}: OK "
              f"({elapsed_ms:.2f} ms total, {time_per_item:.2f} ms/item)")
        
        if len(results) != BATCH_SIZE:
            print(f"  ⚠️ Warning: Sent batch of {BATCH_SIZE} but received {len(results)} results.")

    except Exception as e:
        print(f"⚠️ Request {i+1}/{num_requests} Failed: {e}")
        pass

print("-" * 30)

# --- Calculate and Print Statistics ---
if timings_ms:
    timings_np = np.array(timings_ms)
    # Calculate stats for the *total* batch time
    print("📊 Batched Benchmark Statistics (Total Batch Time):")
    print(f"  Total successful requests: {len(timings_np)}")
    print(f"  Average batch time: {np.mean(timings_np):.2f} ms")
    print(f"  Median batch time:  {np.median(timings_np):.2f} ms")
    print(f"  Min batch time:     {np.min(timings_np):.2f} ms")
    print(f"  Max batch time:     {np.max(timings_np):.2f} ms")
    
    # Calculate stats for the *per-item* time
    per_item_times = timings_np / BATCH_SIZE
    print("\n  --- Per-Item Statistics (Avg. over batch) ---")
    print(f"  Average item time: {np.mean(per_item_times):.2f} ms")
    print(f"  Median item time:  {np.median(per_item_times):.2f} ms")
    print(f"  Min item time:     {np.min(per_item_times):.2f} ms")
    print(f"  Max item time:     {np.max(per_item_times):.2f} ms")
else:
    print("No successful requests were timed.")