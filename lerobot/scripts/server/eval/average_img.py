from PIL import Image
import numpy as np
import os

def average_images():
    # Hard-coded paths (replace with your own file paths)
    path1 = "/mnt/nvme0n1p3/Paper & Projects (Disk)/HIL-AMP/paper/img/Screenshot from 2025-09-08 20-04-13.png"
    path2 = "/mnt/nvme0n1p3/Paper & Projects (Disk)/HIL-AMP/paper/img/Screenshot from 2025-09-08 20-04-21.png"
    path3 = "/mnt/nvme0n1p3/Paper & Projects (Disk)/HIL-AMP/paper/img/Screenshot from 2025-09-08 20-04-31.png"

    # Load images
    img1 = Image.open(path1).convert("RGBA")
    img2 = Image.open(path2).convert("RGBA")
    img3 = Image.open(path3).convert("RGBA")

    # Convert to numpy arrays
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    arr3 = np.array(img3, dtype=np.float32)

    # Compute average
    avg_arr = (arr1 + arr3) / 2.0
    avg_arr = np.clip(avg_arr, 0, 255).astype(np.uint8)

    # Convert back to image
    avg_img = Image.fromarray(avg_arr)

    # Save result in directory of first path
    out_dir = os.path.dirname(path1)
    out_path = os.path.join(out_dir, "average.png")
    avg_img.save(out_path)
    print(f"Saved averaged image to: {out_path}")

if __name__ == "__main__":
    average_images()
