import os.path as osp
import cv2
import numpy as np
import PIL

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
import dust3r.datasets.utils.cropping as cropping

def unproject_depth_image_to_camera_coords_vectorized(depth_image, K):
    H, W = depth_image.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))
    u_coords = u_coords.flatten()
    v_coords = v_coords.flatten()
    depth_values = depth_image.flatten()

    valid_indices = depth_values > 0
    u_coords_valid = u_coords[valid_indices]
    v_coords_valid = v_coords[valid_indices]
    depth_values_valid = depth_values[valid_indices]

    z_c = depth_values_valid
    x_c = (u_coords_valid - cx) * z_c / fx
    y_c = (v_coords_valid - cy) * z_c / fy

    points_camera_coords = np.vstack((x_c, y_c, z_c)).T
    original_pixels = np.vstack((u_coords_valid, v_coords_valid)).T

    return points_camera_coords, original_pixels

def project_world_points_to_pixels_vectorized(points_world, K_target, extrinsics_target_cam_to_world, H, W):
    num_points = points_world.shape[0]
    points_world_hom = np.hstack((points_world, np.ones((num_points, 1))))

    T_world_to_target_cam = np.linalg.inv(extrinsics_target_cam_to_world)
    points_target_cam_hom = (T_world_to_target_cam @ points_world_hom.T).T
    points_target_cam = points_target_cam_hom[:, :3]

    valid_depth_mask = points_target_cam[:, 2] > 0
    valid_points_target_cam = points_target_cam[valid_depth_mask]

    projected_pixels = np.full((num_points, 2), np.nan)
    projected_depths = np.full(num_points, np.nan)

    if valid_points_target_cam.shape[0] == 0:
        return projected_pixels, projected_depths

    z_c = valid_points_target_cam[:, 2]
    u_c = (K_target[0, 0] * valid_points_target_cam[:, 0] / z_c) + K_target[0, 2]
    v_c = (K_target[1, 1] * valid_points_target_cam[:, 1] / z_c) + K_target[1, 2]

    u_c_round = np.round(u_c).astype(int)
    v_c_round = np.round(v_c).astype(int)

    pixel_in_bounds_mask = (u_c_round >= 0) & (u_c_round < W) & \
                           (v_c_round >= 0) & (v_c_round < H)

    original_indices_of_valid_projections = np.where(valid_depth_mask)[0][pixel_in_bounds_mask]

    projected_pixels[original_indices_of_valid_projections, 0] = u_c[pixel_in_bounds_mask]
    projected_pixels[original_indices_of_valid_projections, 1] = v_c[pixel_in_bounds_mask]
    projected_depths[original_indices_of_valid_projections] = z_c[pixel_in_bounds_mask]

    return projected_pixels, projected_depths

def projected_pixels_mask(source_depth_image, target_depth_image,
                               source_intrinsics, source_extrinsics_cam_to_world,
                               target_intrinsics, target_extrinsics_cam_to_world,
                               output_mask_path, depth_tolerance=0.1):

    H, W = target_depth_image.shape
    source_depth_image_meters = source_depth_image / 1000.0
    target_depth_image_meters = target_depth_image / 1000.0

    source_points_cam, source_original_pixels = \
        unproject_depth_image_to_camera_coords_vectorized(source_depth_image_meters, source_intrinsics)

    source_points_cam_hom = np.hstack((source_points_cam, np.ones((source_points_cam.shape[0], 1))))
    points_in_world_coords = (source_extrinsics_cam_to_world @ source_points_cam_hom.T).T[:, :3]

    target_projected_pixels, target_projected_depths = \
        project_world_points_to_pixels_vectorized(points_in_world_coords, target_intrinsics,
                                                  target_extrinsics_cam_to_world, H, W)


    source_image_mask = np.zeros((H, W), dtype=np.uint8)
    target_image_mask = np.zeros((H, W), dtype=np.uint8)

    valid_geom_proj_mask_per_initial_pixel = \
        ~np.isnan(target_projected_pixels[:, 0])

    for i in range(len(source_original_pixels)):
        if valid_geom_proj_mask_per_initial_pixel[i]:
            proj_u, proj_v = target_projected_pixels[i, 0], target_projected_pixels[i, 1]
            proj_depth = target_projected_depths[i]

            proj_u_int, proj_v_int = int(round(proj_u)), int(round(proj_v))

            if 0 <= proj_u_int < W and 0 <= proj_v_int < H:
                actual_target_depth = target_depth_image_meters[proj_v_int, proj_u_int]

                if actual_target_depth > 0 and np.abs(proj_depth - actual_target_depth) < depth_tolerance:
                    target_image_mask[proj_v_int, proj_u_int] = 255
                    source_image_mask[source_original_pixels[i, 1], source_original_pixels[i, 0]] = 255

    return source_image_mask


class ScanNetpp_CoVis(BaseStereoViewDataset):
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
            npz_file = 'train_metadata_updated_2.npz'
        elif self.split == 'test':
            npz_file = 'test_metadata_updated_2.npz'
        else:
            raise ValueError(f"Unknown split: {self.split}. Only 'train' and 'test' are supported.")
        with np.load(osp.join(self.ROOT, npz_file)) as data:
            self.scenes = data['scenes']
            self.sceneids = data['sceneids']
            self.images = data['images']
            self.intrinsics = data['intrinsics'].astype(np.float32)
            self.trajectories = data['trajectories'].astype(np.float32)
            self.pairs = data['pairs'][:, :2].astype(int)


    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):

        image_idx1, image_idx2 = self.pairs[idx]

        views = []
        
        scene_id = self.sceneids[image_idx1]
        scene_dir = osp.join(self.ROOT, self.scenes[scene_id])

        intrinsics_1 = self.intrinsics[image_idx1]
        intrinsics_2 = self.intrinsics[image_idx2]
        camera_pose_1 = self.trajectories[image_idx1]
        camera_pose_2 = self.trajectories[image_idx2]
        basename_1 = self.images[image_idx1]
        basename_2 = self.images[image_idx2]

        rgb_path_1 = osp.join(scene_dir, 'images', basename_1 + '.jpg')
        depth_path_1 = osp.join(scene_dir, 'depth', basename_1 + '.png')
        rgb_path_2 = osp.join(scene_dir, 'images', basename_2 + '.jpg')
        depth_path_2 = osp.join(scene_dir, 'depth', basename_2 + '.png')

        # Load RGB images
        rgb_image_1 = imread_cv2(rgb_path_1)
        rgb_image_2 = imread_cv2(rgb_path_2)
        # Load depthmaps
        depthmap_1 = imread_cv2(depth_path_1, cv2.IMREAD_UNCHANGED)
        depthmap_1 = depthmap_1.astype(np.float32) / 1000
        depthmap_1[~np.isfinite(depthmap_1)] = 0  # invalid

        depthmap_2 = imread_cv2(depth_path_2, cv2.IMREAD_UNCHANGED)
        depthmap_2 = depthmap_2.astype(np.float32) / 1000
        depthmap_2[~np.isfinite(depthmap_2)] = 0  # invalid

        # Calculate covisibility masks
        covis_dir = osp.join(self.ROOT[:-8], 'Co_Visibility', self.scenes[scene_id])
        path_a = osp.join(covis_dir, f'{basename_1}_to_{basename_2}.png')
        path_b = osp.join(covis_dir, f'{basename_2}_to_{basename_1}.png')
        
        if osp.exists(path_a):
            covisibility_mask_1 = imread_cv2(osp.join(covis_dir, f'{basename_1}_to_{basename_2}.png'), cv2.IMREAD_UNCHANGED)
            covisibility_mask_2 = imread_cv2(osp.join(covis_dir, f'{basename_2}_to_{basename_1}.png'), cv2.IMREAD_UNCHANGED)    
        else:
            print("making and not reading")
            print(f"Creating covisibility masks for {basename_1} and {basename_2} with scene {self.scenes[scene_id]} and id {idx}")
            covisibility_mask_1 = projected_pixels_mask(
                depthmap_1, depthmap_2, intrinsics_1, camera_pose_1,
                intrinsics_2, camera_pose_2, output_mask_path=None)
            covisibility_mask_2 = projected_pixels_mask(
                depthmap_2, depthmap_1, intrinsics_2, camera_pose_2,
                intrinsics_1, camera_pose_1, output_mask_path=None)
            cv2.imwrite(path_a, covisibility_mask_1)
            cv2.imwrite(path_b, covisibility_mask_2)  
        # Crop and resize if necessary
        
        rgb_image_1, depthmap_1, intrinsics_1 = self._crop_resize_if_necessary(
            rgb_image_1, depthmap_1, intrinsics_1, resolution, rng=rng, info=image_idx1)
        rgb_image_2, depthmap_2, intrinsics_2 = self._crop_resize_if_necessary(
            rgb_image_2, depthmap_2, intrinsics_2, resolution, rng=rng, info=image_idx2)
        scale_final = max(resolution[0] / covisibility_mask_1.shape[0], resolution[1] / covisibility_mask_1.shape[1]) + 1e-8
        covisibility_mask_1 = cv2.resize(covisibility_mask_1, resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)
        covisibility_mask_2 = cv2.resize(covisibility_mask_2, resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)


        views.append(dict(
            img=rgb_image_1,
            depthmap=depthmap_1.astype(np.float32),
            camera_pose=camera_pose_1.astype(np.float32),
            camera_intrinsics=intrinsics_1.astype(np.float32),
            dataset='ScanNet++',
            label=self.scenes[scene_id] + '_' + basename_1,
            instance=f'{str(idx)}_{str(image_idx1)}',
            covisibility_mask=covisibility_mask_1,  # Placeholder for covisibility mask
        ))
        views.append(dict(
            img=rgb_image_2,
            depthmap=depthmap_2.astype(np.float32),
            camera_pose=camera_pose_2.astype(np.float32),
            camera_intrinsics=intrinsics_2.astype(np.float32),
            dataset='ScanNet++',
            label=self.scenes[scene_id] + '_' + basename_2,
            instance=f'{str(idx)}_{str(image_idx2)}',
            covisibility_mask=covisibility_mask_2,  # Placeholder for covisibility mask
        ))

        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = ScanNetpp(split='train', ROOT="data/scannetpp_processed", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
