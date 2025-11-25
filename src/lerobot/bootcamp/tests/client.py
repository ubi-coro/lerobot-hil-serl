import requests
import numpy as np
import json_numpy
import time  # <-- Import the time module

server_url = "http://hkn0812:8000/act"
timeout = 5
num_requests = 20  # <-- Number of requests to send

# Prepare inputs
# Note: Using 20-dim proprio from your custom code
proprio = np.zeros(20, dtype=np.float32) 
image = np.zeros((256, 256, 3), dtype=np.uint8)
instruction = "Move the gripper to the target position"

payload = {
    "proprio": json_numpy.dumps(proprio),
    "language_instruction": instruction,
    "image0": json_numpy.dumps(image),
    "domain_id": 0,
    "steps": 10
}

timings_ms = []  # <-- List to store all timings
print(f"🚀 Sending {num_requests} requests to {server_url}...")

# --- Optional: Warm-up request ---
# The first request can be slower; let's get it out of the way.
try:
    print("Sending warm-up request...")
    requests.post(server_url, json=payload, timeout=timeout)
    print("Warm-up complete. Starting benchmark.")
except Exception as e:
    print(f"⚠️ Warm-up request failed: {e}. Aborting.")
    exit()

print("-" * 30)

# --- Main Test Loop ---
for i in range(num_requests):
    start_time = time.perf_counter()
    
    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
        response.raise_for_status()  # Check for HTTP errors
        result = response.json()
        
        # --- Stop Timer ---
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        timings_ms.append(elapsed_ms)
        
        actions = np.array(result["action"], dtype=np.float32)
        print(f"✅ Request {i+1}/{num_requests}: OK ({elapsed_ms:.2f} ms)")

    except Exception as e:
        print(f"⚠️ Request {i+1}/{num_requests} Failed: {e}")
        # Optionally, you could append a 'None' or skip this one
        pass

print("-" * 30)

# --- Calculate and Print Statistics ---
if timings_ms:
    timings_np = np.array(timings_ms)
    print("📊 Benchmark Statistics:")
    print(f"  Total successful requests: {len(timings_np)}")
    print(f"  Average time: {np.mean(timings_np):.2f} ms")
    print(f"  Median time:  {np.median(timings_np):.2f} ms")
    print(f"  Min time:     {np.min(timings_np):.2f} ms")
    print(f"  Max time:     {np.max(timings_np):.2f} ms")
    print(f"  Std. Dev.:    {np.std(timings_np):.2f} ms")
else:
    print("No successful requests were timed.")