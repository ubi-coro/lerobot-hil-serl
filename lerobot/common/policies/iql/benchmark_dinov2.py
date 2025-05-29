import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import time
from transformers import AutoImageProcessor, AutoModel

# Load the pre-trained DINO-v2-base model
model_name = "facebook/dinov2-base"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load an example image
url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.jpeg'
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB")

# Preprocess the image
inputs = processor(images=img, return_tensors="pt").to(device)

# Warm-up
with torch.no_grad():
    _ = model(**inputs)

# Benchmark inference
times = []
n_runs = 100
with torch.no_grad():
    for _ in range(n_runs):
        start_time = time.time()
        outputs = model(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        times.append(elapsed)

avg_time = sum(times) / len(times)

print(f"Average inference time over {n_runs} runs: {avg_time*1000:.2f} ms")
