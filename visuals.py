import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
tensor_path = "bowlers/SL Malinga.pt"  # Update to the .pt file you want
output_dir = "bowler_images"
os.makedirs(output_dir, exist_ok=True)

# === LOAD TENSOR ===
tensor = torch.load(tensor_path)
tensor_np = tensor.numpy()
rows, cols = tensor_np.shape

# === 1. GRAYSCALE IMAGE ===
value_map = {
    -1: 0,   # wicket
    0: 100,  # dot/extras
    1: 140,
    2: 180,
    3: 200,
    4: 220,
    6: 255
}
gray_array = np.vectorize(lambda x: value_map.get(x, 100))(tensor_np).astype(np.uint8)
gray_img = Image.fromarray(gray_array, mode='L')
gray_img = gray_img.resize((cols * 10, rows * 10), Image.NEAREST)
gray_img.save(os.path.join(output_dir, "grayscale.jpg"))
print("✅ Saved:", os.path.join(output_dir, "grayscale.jpg"))

# === 2. COLOR-CODED RGB IMAGE ===
def map_color(val):
    if val == -1:
        return [255, 0, 0]       # Red: wicket
    elif val == 0:
        return [0, 0, 255]       # Blue: dot/extras
    elif val == 1:
        return [0, 128, 255]     # Light blue: single
    elif val == 2:
        return [0, 255, 0]       # Green: double
    elif val == 3:
        return [100, 255, 100]   # Light green: triple
    elif val == 4:
        return [255, 255, 0]     # Yellow: boundary
    elif val == 6:
        return [255, 140, 0]     # Orange: six
    else:
        return [150, 150, 150]   # Gray: unknown

color_array = np.array([[map_color(val) for val in row] for row in tensor_np], dtype=np.uint8)
color_img = Image.fromarray(color_array, mode='RGB')
color_img = color_img.resize((cols * 10, rows * 10), Image.NEAREST)
color_img.save(os.path.join(output_dir, "color.jpg"))
print("✅ Saved:", os.path.join(output_dir, "color.jpg"))

