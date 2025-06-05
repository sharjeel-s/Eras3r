# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed scannet++ with random masking
# Based on scannetpp.py
# --------------------------------------------------------
import os.path as osp
import cv2
import numpy as np
import random # Added import
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetppMasked(BaseStereoViewDataset): # Renamed class
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        #assert self.split == 'train'
        self.loaded_data = self._load_data()

    def _load_data(self):
        # added the test split
        # fraction used is 0.1
        # test set is from the pairs and not the images, could try this later
        if self.split == 'train':
            npz_file = 'train_split.npz'
        elif self.split == 'test':
            npz_file = 'test_split.npz'
        else:
            raise ValueError(f"Unknown split: {self.split}. Only 'train' and 'test' are supported.")
        
        npz_path = osp.join(self.ROOT, npz_file)
        if not osp.exists(npz_path):
             raise FileNotFoundError(f"Split file not found: {npz_path}")

        with np.load(npz_path) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)

    def __len__(self):
        return len(self.pairs)

    def apply_numpy_mask_to_pil_image(self, pil_image, numpy_mask):
        """Applies a NumPy mask (0 or 1) to a PIL Image."""
        if numpy_mask.shape[:2] != pil_image.size[::-1]:
            raise ValueError("Mask and image dimensions do not match.")
        img_array = np.array(pil_image).astype(float) / 255.0
        masked_array = img_array * numpy_mask[..., None]
        masked_pil_image = Image.fromarray((masked_array * 255).astype(np.uint8))
        return masked_pil_image

    def _generate_random_polygon_mask(self, height, width, rng):
        """Generates a random polygonal mask."""
        mask = np.ones((height, width), dtype=np.uint8) # Start with all ones (keep all)
        # Ensure at least 3 vertices for a polygon
        num_vertices = rng.integers(3, 8) # Random number of vertices (e.g., 3 to 7)
        vertices = np.array([
            # Ensure vertices are within bounds
            [rng.integers(0, max(0, width - 1)), rng.integers(0, max(0, height - 1))]
            for _ in range(num_vertices)
        ], dtype=np.int32)

        # Check if vertices form a valid polygon (at least 3 points)
        if vertices.shape[0] >= 3:
             # Create the polygon mask (setting polygon area to 0)
             cv2.fillPoly(mask, [vertices], 0)
        # else: mask remains all ones if vertices are invalid (e.g., width/height is 0)

        return mask # 0 where masked, 1 otherwise
    
    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        apply_mask_to_first = rng.choice([True, False])
        height, width = resolution

        for i, view_idx in enumerate([image_idx1, image_idx2]):
            scene_id = self.sceneids[view_idx]
            scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

            intrinsics = self.intrinsics[view_idx]
            camera_pose = self.trajectories[view_idx]
            basename = self.images[view_idx]

            # Load RGB image
            rgb_image = imread_cv2(osp.join(scene_dir, 'images', basename + '.jpg'))
            # Load depthmap
            depthmap = imread_cv2(osp.join(scene_dir, 'depth', basename + '.png'), cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            
            if (i == 0 and apply_mask_to_first) or (i == 1 and not apply_mask_to_first):
                mask = self._generate_random_polygon_mask(height, width, rng)
                if isinstance(rgb_image, Image.Image):
                    masked_rgb_image = self.apply_numpy_mask_to_pil_image(rgb_image, mask)
                else:  # Assume it's a NumPy array
                    masked_rgb_image = rgb_image * mask[..., None]
            else:
                mask = np.ones((height, width), dtype=np.uint8)
                masked_rgb_image = rgb_image
            
            views.append(dict(
                img=masked_rgb_image, # Use masked image
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=self.scenes[scene_id] + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
                mask=mask,  # Include the mask
            ))
        return views
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    # Example usage: Replace with your actual data path
    DATA_ROOT = "data/scannetpp_processed" 
    if not osp.isdir(DATA_ROOT):
        print(f"Error: Data root directory not found: {DATA_ROOT}")
        print("Please update DATA_ROOT in the script.")
    else:
        dataset = ScanNetppMasked(split='train', ROOT=DATA_ROOT, resolution=224, aug_crop=16)

        if len(dataset) == 0:
            print("Dataset is empty. Check the data path and split files.")
        else:
            print(f"Dataset loaded with {len(dataset)} pairs.")
            # Visualize a few examples
            num_examples_to_show = 3
            count = 0
            for idx in np.random.permutation(len(dataset)):
                if count >= num_examples_to_show:
                    break
                
                views = dataset[idx]
                if views is None: # Skip if the pair failed to load
                    continue 

                assert len(views) == 2
                print(f"\n--- Example {count+1} ---")
                print(f"Pair Index: {idx}")
                print(f"View 1: {view_name(views[0])}")
                print(f"View 2: {view_name(views[1])}")

                # --- Visualization ---
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle(f"Pair Index: {idx}")

                for i, view in enumerate(views):
                    # Original Image
                    ax = axes[i, 0]
                    ax.imshow(view['original_img'])
                    ax.set_title(f"View {i+1}: Original")
                    ax.axis('off')

                    # Mask
                    ax = axes[i, 1]
                    ax.imshow(view['mask'], cmap='gray')
                    ax.set_title(f"View {i+1}: Mask (0=masked)")
                    ax.axis('off')

                    # Masked Image
                    ax = axes[i, 2]
                    ax.imshow(view['img'])
                    ax.set_title(f"View {i+1}: Masked Image")
                    ax.axis('off')
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                plt.show()
                # --- End Visualization ---

                # Optional: 3D Viz (uncomment if needed)
                # viz = SceneViz()
                # poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
                # cam_size = max(auto_cam_size(poses), 0.001)
                # for view_idx in [0, 1]:
                #     pts3d = views[view_idx]['pts3d'] # Note: pts3d might not be calculated by default
                #     valid_mask = views[view_idx]['valid_mask'] # Note: valid_mask might not be calculated by default
                #     colors = rgb(views[view_idx]['original_img']) # Use original colors for point cloud
                #     # Check if pts3d and valid_mask exist before using
                #     if 'pts3d' in views[view_idx] and 'valid_mask' in views[view_idx]:
                #          viz.add_pointcloud(pts3d, colors, valid_mask)
                #     viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                #                    focal=views[view_idx]['camera_intrinsics'][0, 0],
                #                    color=(idx*255, (1 - idx)*255, 0),
                #                    image=rgb(views[view_idx]['img']), # Show masked image in camera frustum
                #                    cam_size=cam_size)
                # viz.show()
                
                count += 1