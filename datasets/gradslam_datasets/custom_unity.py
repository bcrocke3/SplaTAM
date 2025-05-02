import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# FOR VISUAL GUI DEBUGGING
import matplotlib
matplotlib.use("TkAgg")

class FromUnityDataset:
    def __init__(self, config_dict, basedir, sequence, **kwargs):
        self.basedir = os.path.join(basedir, sequence, "results")
        self.rgb_files = sorted([f for f in os.listdir(self.basedir) if f.startswith("frame") and f.endswith(".png")])
        self.depth_files = sorted([f for f in os.listdir(self.basedir) if f.startswith("depth") and f.endswith(".png")])

        # Camera intrinsics from the YAML file
        camera_params = config_dict.get("camera_params", {})
        self.fx = camera_params.get("fx", 1.0)
        self.fy = camera_params.get("fy", 1.0)
        self.cx = camera_params.get("cx", 0.0)
        self.cy = camera_params.get("cy", 0.0)
        self.depth_scale = camera_params.get("png_depth_scale", 1.0)

        self.image_height = camera_params.get("image_height", 480)
        self.image_width = camera_params.get("image_width", 640)

        self.device = kwargs.get("device", "cpu")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):

        # Load RGB image
        rgb_path = os.path.join(self.basedir, self.rgb_files[idx])
        rgb = Image.open(rgb_path).convert("RGB")
        
        rgb = rgb.resize((self.image_width, self.image_height))

        rgb = np.array(rgb, dtype=np.float32)
        rgb = torch.tensor(rgb).to(self.device)  # (H, W, C)
        print(f"Final RGB tensor shape: {rgb.shape}")
        # After loading image


        # Visualize RGB image
        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.title("RGB Image")
        # plt.imshow(rgb.cpu().numpy())  # Convert (C, H, W) -> (H, W, C) for visualization
        # plt.axis("off")

        # Load depth image
        depth_path = os.path.join(self.basedir, self.depth_files[idx])
        depth = Image.open(depth_path)

        depth = depth.resize((self.image_width, self.image_height))

        depth = np.array(depth, dtype=np.float32) / self.depth_scale  # Scale depth values
        depth = np.expand_dims(depth, axis=-1)  # Add channel dimension (H, W) -> (H, W, 1)
        depth = torch.tensor(depth).to(self.device) 
        print(f"Final depth tensor shape: {depth.shape}")
        print()

        # Visualize Depth image
        # plt.subplot(1, 2, 2)
        # plt.title("Depth Image")
        # plt.imshow(depth[:, :, 0].cpu().numpy(), cmap="viridis")  # Use the first channel of (H, W, C)
        # plt.axis("off")

        # plt.show()

        # Camera intrinsics
        intrinsics = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # Identity pose (placeholder, as no pose data is provided)
        pose = torch.eye(4, dtype=torch.float32, device=self.device)

        return rgb, depth, intrinsics, pose