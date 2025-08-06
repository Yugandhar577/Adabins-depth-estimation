# from models import UnetAdaptiveBins
# import model_io
# from PIL import Image
# from infer import InferenceHelper

# MIN_DEPTH = 1e-3
# MAX_DEPTH_NYU = 10
# N_BINS = 256 

# model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
# pretrained_path = "./pretrained/AdaBins_nyu.pt"
# model, _, _ = model_io.load_checkpoint(pretrained_path, model)

# infer_helper = InferenceHelper(dataset='nyu')

# img = Image.open("example_image3.jpg")
# img_resized = img.resize((384, 384))
# bin_centers, predicted_depth = infer_helper.predict_pil(img_resized)

# import matplotlib.pyplot as plt
# import numpy as np
# depth_array = np.squeeze(np.array(predicted_depth))  # shape becomes (384, 384)
# depth_vis = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
# plt.imshow(depth_vis, cmap='plasma')
# plt.colorbar(label="Normalized Depth")
# plt.title("Predicted Depth Map")
# plt.show()

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models import UnetAdaptiveBins
import model_io
from infer import InferenceHelper
import torch.nn.functional as F

# Constants
MIN_DEPTH = 1e-3
MAX_DEPTH_NYU = 10
N_BINS = 256

# Load Model
model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_NYU)
pretrained_path = "./pretrained/AdaBins_nyu.pt"
model, _, _ = model_io.load_checkpoint(pretrained_path, model)

infer_helper = InferenceHelper(dataset='nyu')

# Load Original Image
img = Image.open("example_image3.jpg")
original_size = img.size  # (width, height)

# Resize for Model Input
img_resized = img.resize((384, 384))
bin_centers, predicted_depth = infer_helper.predict_pil(img_resized)

# Convert depth to numpy
depth_array = np.array(predicted_depth).squeeze()  # shape (384, 384)

# Resize depth back to original image size
depth_tensor = torch.tensor(depth_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape (1,1,384,384)

depth_resized = F.interpolate(
    depth_tensor,
    size=(original_size[1], original_size[0]),  # (height, width)
    mode="bilinear",
    align_corners=False
).squeeze().numpy()

# Display Full-Size Depth Map
plt.imshow(depth_resized, cmap='plasma')
plt.colorbar(label="Depth")
plt.title("Predicted Depth Map (Full Size)")
plt.show()
