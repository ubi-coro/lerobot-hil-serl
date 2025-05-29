import torch
from torchvision import transforms
import time
from transformers import AutoImageProcessor, AutoModel

# Load the pre-trained DINO-v2-base model
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name, use_fast=False)
model = AutoModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Generate a random image
random_image = torch.rand(3, 224, 224)

# Preprocess the random image
inputs = processor(images=random_image, return_tensors="pt").to(device)

# Warm-up
with torch.no_grad():
    _ = model(**inputs)

# Benchmark inference
times = []
n_runs = 1000
with torch.no_grad():
    for _ in range(n_runs):
        start_time = time.time()
        outputs = model(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        times.append(elapsed)

avg_time = sum(times) / len(times)

print(f"Average inference time over {n_runs} runs: {avg_time*1000:.2f} ms")
print(f"Average fps: {1/avg_time:.2f} fps")
print(f"Variance of inference time: {torch.var(torch.tensor(times)).item()*1000:.2f} ms^2")

"""
On A6000:

Average inference time over 1000 runs: 4.87 ms
Average fps: 205.38 fps
Variance of inference time: 0.01 ms^2
"""
