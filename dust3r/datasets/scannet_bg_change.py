import os
import os.path as osp
import cv2
import numpy as np
import random # Added import
from PIL import Image

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ScanNetppNewBG(BaseStereoViewDataset): # Renamed class
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
            npz_file = 'all_metadata.npz'
        elif self.split == 'test':
            npz_file = 'all_metadata.npz'
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
        
        self.scenes_len = len(self.scenes)
        

    def __len__(self):
        return len(self.pairs)

    def fill_in(self, pil_image, fill_in_image, numpy_mask):
        """Applies a NumPy mask (0 or 1) to a PIL Image."""
        if numpy_mask.shape[:2] != pil_image.size[::-1]:
            raise ValueError("Mask and image dimensions do not match.")
        img_array = np.array(pil_image).astype(float) / 255.0
        masked_array = img_array * (1 - numpy_mask[..., None]) + fill_in_image
        masked_pil_image = Image.fromarray((masked_array * 255).astype(np.uint8))
        return masked_pil_image
    
    def _generate_random_polygon_mask(self, height, width, min_radius=0.1, max_radius=0.7, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        num_edges = rng.integers(3, 10)  # Random number of edges between 3 and 9
        mask = np.ones((height, width), dtype=np.uint8)
        # Random center point
        cx = rng.integers(int(0.2*width), int(0.8*width))
        cy = rng.integers(int(0.2*height), int(0.8*height))
        # Random angles that sum to 2*pi
        angles = rng.uniform(0, 1, size=num_edges)
        angles = angles / angles.sum() * 2 * np.pi
        angles = np.cumsum(angles)
        angles = np.insert(angles, 0, 0)[:-1]  # Start from 0
        # Generate vertices
        vertices = []
        for angle in angles:
            r = rng.uniform(min_radius, max_radius) * height
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            # Clip to image boundaries
            x = np.clip(x, 0, width-1)
            y = np.clip(y, 0, height-1)
            vertices.append([x, y])
        vertices = np.array([vertices], dtype=np.int32)
        # Draw polygon on mask (set inside to 0)
        cv2.fillPoly(mask, vertices, 0)

        if(rng.random() < 0.5):
            mask = 1 - mask

        return mask
    
    def _fill_in_picture(self, scene_id, resolution, rng):
        scene_idx = rng.integers(self.scenes_len)
        scene = self.scenes[scene_idx]
        if scene == scene_id:
            scene = self.scenes[(scene_idx + 1) % self.scenes_len]

        scene_dir = osp.join(self.ROOT, scene, 'images')
        new_img = rng.choice(os.listdir(scene_dir)) 
        rgb_image = imread_cv2(osp.join(scene_dir, new_img))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_map = imread_cv2(osp.join(scene_dir[:-6], 'depth', new_img[:-4] + '.png'), cv2.IMREAD_UNCHANGED)
        depth_map = depth_map.astype(np.float32) / 1000
        depth_map[~np.isfinite(depth_map)] = 0  # invalid
        return rgb_image, depth_map

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        apply_mask_to_first = rng.choice([True, False])
        height, width = resolution
        
        scene_id = self.sceneids[image_idx1]
        rgb_image_fill_in, depth_map_fill_in = self._fill_in_picture(scene_id, resolution, rng)
        rgb_image_fill_in = cv2.resize(rgb_image_fill_in, resolution)
        depth_map_fill_in = cv2.resize(depth_map_fill_in, resolution)
        
        
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

            # Getting the image to fill in the masked out part
            

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            
            # Create a mask for the image and fill in with the 3rd image
            
            if (i == 0 and apply_mask_to_first) or (i == 1 and not apply_mask_to_first):
                mask = self._generate_random_polygon_mask(height, width, rng=rng)
                rgb_image_fill_in = rgb_image_fill_in * mask[..., None] # Fill in the masked area with the new image
                if isinstance(rgb_image, Image.Image):
                    masked_rgb_image = self.fill_in(rgb_image, rgb_image_fill_in, mask)

                else:  # Assume it's a NumPy array
                    masked_rgb_image = rgb_image * mask[..., None]
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
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