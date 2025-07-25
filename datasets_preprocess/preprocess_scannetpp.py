#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to pre-process the scannet++ dataset.
# Usage:
# python3 datasets_preprocess/preprocess_scannetpp.py --scannetpp_dir /path/to/scannetpp --precomputed_pairs /path/to/scannetpp_pairs --pyopengl-platform egl
# --------------------------------------------------------
import os
import argparse
import os.path as osp
import re
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
import trimesh.exchange.ply
import numpy as np
import cv2
import PIL.Image as Image

from dust3r.datasets.utils.cropping import rescale_image_depthmap
import dust3r.utils.geometry as geometry

inv = np.linalg.inv
norm = np.linalg.norm
REGEXPR_DSLR = re.compile(r'^DSC(?P<frameid>\d+).JPG$')
REGEXPR_IPHONE = re.compile(r'frame_(?P<frameid>\d+).jpg$')

DEBUG_VIZ = None  # 'iou'
if DEBUG_VIZ is not None:
    import matplotlib.pyplot as plt  # noqa


OPENGL_TO_OPENCV = np.float32([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp_dir', required=True)
    parser.add_argument('--precomputed_pairs', required=True)
    parser.add_argument('--output_dir', default='data/scannetpp_processed')
    parser.add_argument('--target_resolution', default=920, type=int, help="images resolution")
    parser.add_argument('--pyopengl-platform', type=str, default='', help='PyOpenGL env variable')
    return parser


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    return np.linalg.inv(pose)  # returns cam2world


def get_frame_number(name, cam_type='dslr'):
    if cam_type == 'dslr':
        regex_expr = REGEXPR_DSLR
    elif cam_type == 'iphone':
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f'wrong {cam_type=} for get_frame_number')
    matches = re.match(regex_expr, name)
    return matches['frameid']


def load_sfm(sfm_dir, cam_type='dslr'):
    # load cameras
    with open(osp.join(sfm_dir, 'cameras.txt'), 'r') as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in tqdm(raw, position=1, leave=False):
        camera = camera.split(' ')
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]

    # load images
    with open(os.path.join(sfm_dir, 'images.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in tqdm(zip(raw[0::2], raw[1::2]), total=len(raw) // 2, position=1, leave=False):
        image = image.split(' ')
        points = points.split(' ')

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, 'duplicate db image: ' + img_name
        img_idx[img_name] = idx  # register image name

        current_points2D = {int(i): (float(x), float(y))
                            for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(intrinsics=intrinsics[int(image[-2])],
                              path=img_name,
                              frame_id=get_frame_number(img_name, cam_type),
                              cam_to_world=pose_from_qwxyz_txyz(image[1: -2]),
                              sparse_pts2d=current_points2D)

    # load 3D points
    with open(os.path.join(sfm_dir, 'points3D.txt'), 'r') as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith('#')]  # skip header

    points3D = {}
    observations = {idx: [] for idx in img_infos.keys()}
    for point in tqdm(raw, position=1, leave=False):
        point = point.split()
        point_3d_idx = int(point[0])
        points3D[point_3d_idx] = tuple(map(float, point[1:4]))
        if len(point) > 8:
            for idx, point_2d_idx in zip(point[8::2], point[9::2]):
                observations[idx].append((point_3d_idx, int(point_2d_idx)))

    return img_idx, img_infos, points3D, observations


def subsample_img_infos(img_infos, num_images, allowed_name_subset=None):
    img_infos_val = [(idx, val) for idx, val in img_infos.items()]
    if allowed_name_subset is not None:
        img_infos_val = [(idx, val) for idx, val in img_infos_val if val['path'] in allowed_name_subset]

    if len(img_infos_val) > num_images:
        img_infos_val = sorted(img_infos_val, key=lambda x: x[1]['frame_id'])
        kept_idx = np.round(np.linspace(0, len(img_infos_val) - 1, num_images)).astype(int).tolist()
        img_infos_val = [img_infos_val[idx] for idx in kept_idx]
    return {idx: val for idx, val in img_infos_val}


def undistort_images(intrinsics, rgb, mask):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = geometry.colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion, (width, height), 1, (width, height), True)
        map1, map2 = cv2.initUndistortRectifyMap(K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

    undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = geometry.opencv_to_colmap_intrinsics(new_K)
    return width, height, new_K, undistorted_image, undistorted_mask


def process_scenes(root, pairsdir, output_dir, target_resolution):
    os.makedirs(output_dir, exist_ok=True)

    # default values from
    # https://github.com/scannetpp/scannetpp/blob/main/common/configs/render.yml
    znear = 0.05
    zfar = 20.0

    listfile = osp.join(pairsdir, 'scene_list.json')
    with open(listfile, 'r') as f:
        scenes = json.load(f)

    # for each of these, we will select some dslr images and some iphone images
    # we will undistort them and render their depth
    renderer = pyrender.OffscreenRenderer(0, 0)
    for scene in tqdm(scenes, position=0, leave=True):
        data_dir = os.path.join(root, 'data', scene)
        dir_dslr = os.path.join(data_dir, 'dslr')
        dir_iphone = os.path.join(data_dir, 'iphone')
        dir_scans = os.path.join(data_dir, 'scans')

        assert os.path.isdir(data_dir) and os.path.isdir(dir_dslr) \
            and os.path.isdir(dir_iphone) and os.path.isdir(dir_scans)

        output_dir_scene = os.path.join(output_dir, scene)
        scene_metadata_path = osp.join(output_dir_scene, 'scene_metadata.npz')
        if osp.isfile(scene_metadata_path):
            continue

        pairs_dir_scene = os.path.join(pairsdir, scene)
        pairs_dir_scene_selected_pairs = os.path.join(pairs_dir_scene, 'selected_pairs.npz')
        assert osp.isfile(pairs_dir_scene_selected_pairs)
        selected_npz = np.load(pairs_dir_scene_selected_pairs)
        selection_orig, pairs_orig = selected_npz['selection'], selected_npz['pairs']

        # --- Filter for DSLR images only ---
        original_selection_indices = {name: i for i, name in enumerate(selection_orig)}
        dslr_selection = [name for name in selection_orig if name.startswith('DSC')]
        dslr_original_indices = {original_selection_indices[name] for name in dslr_selection}

        if pairs_orig.size > 0:
            dslr_pairs_mask = np.isin(pairs_orig[:, 0], list(dslr_original_indices)) & \
                              np.isin(pairs_orig[:, 1], list(dslr_original_indices))
            filtered_pairs_old_indices = pairs_orig[dslr_pairs_mask]

            # Remap pair indices to match the new dslr_selection order
            old_dslr_idx_to_new_idx_map = {original_selection_indices[name]: new_idx
                                           for new_idx, name in enumerate(dslr_selection)}

            if len(filtered_pairs_old_indices) > 0:
                 final_dslr_pairs = np.array([[old_dslr_idx_to_new_idx_map[p[0]], old_dslr_idx_to_new_idx_map[p[1]]] + list(p[2:])
                                             for p in filtered_pairs_old_indices])
            else:
                 final_dslr_pairs = np.empty((0, pairs_orig.shape[1]), dtype=pairs_orig.dtype)
        else:
            final_dslr_pairs = np.empty((0, 0), dtype=pairs_orig.dtype)

        # Use the filtered DSLR data from now on
        selection = dslr_selection
        pairs = final_dslr_pairs
        print(f"Scene {scene}: Kept {len(selection)} DSLR images and {len(pairs)} DSLR-only pairs.")
        # --- End DSLR Filter ---


        # set up the output paths
        output_dir_scene_rgb = os.path.join(output_dir_scene, 'images')
        output_dir_scene_depth = os.path.join(output_dir_scene, 'depth')
        os.makedirs(output_dir_scene_rgb, exist_ok=True)
        os.makedirs(output_dir_scene_depth, exist_ok=True)

        ply_path = os.path.join(dir_scans, 'mesh_aligned_0.05.ply')

        sfm_dir_dslr = os.path.join(dir_dslr, 'colmap')
        rgb_dir_dslr = os.path.join(dir_dslr, 'resized_images')
        mask_dir_dslr = os.path.join(dir_dslr, 'resized_anon_masks')

        # sfm_dir_iphone = os.path.join(dir_iphone, 'colmap')
        # rgb_dir_iphone = os.path.join(dir_iphone, 'rgb')
        # mask_dir_iphone = os.path.join(dir_iphone, 'rgb_masks')

        # load the mesh
        with open(ply_path, 'rb') as f:
            mesh_kwargs = trimesh.exchange.ply.load_ply(f)
        mesh_scene = trimesh.Trimesh(**mesh_kwargs)

        # read colmap reconstruction, we will only use the intrinsics and pose here
        img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr = load_sfm(sfm_dir_dslr, cam_type='dslr')
        dslr_paths = {
            "in_colmap": sfm_dir_dslr,
            "in_rgb": rgb_dir_dslr,
            "in_mask": mask_dir_dslr,
        }

        # img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone = load_sfm(
            # sfm_dir_iphone, cam_type='iphone')
        # iphone_paths = {
            # "in_colmap": sfm_dir_iphone,
            # "in_rgb": rgb_dir_iphone,
            # "in_mask": mask_dir_iphone,
        # }

        mesh = pyrender.Mesh.from_trimesh(mesh_scene, smooth=False)
        pyrender_scene = pyrender.Scene()
        pyrender_scene.add(mesh)

        selection_dslr = [imgname + '.JPG' for imgname in selection if imgname.startswith('DSC')]
        # selection_iphone = [imgname + '.jpg' for imgname in selection if imgname.startswith('frame_')]

        # resize the image to a more manageable size and render depth
        missing_images_in_scene = set() # Keep track of images missing from COLMAP
        for selection_cam, img_idx, img_infos, paths_data in [(selection_dslr, img_idx_dslr, img_infos_dslr, dslr_paths)]:
            rgb_dir = paths_data['in_rgb']
            mask_dir = paths_data['in_mask']
            for imgname in tqdm(selection_cam, position=1, leave=False):
                img_base_name = imgname[:-4] # Get base name (e.g., DSC01234)
                if imgname not in img_idx:
                    print(f"Warning: Image '{imgname}' from selection not found in COLMAP images.txt for scene '{scene}'. Skipping.")
                    missing_images_in_scene.add(img_base_name)
                    continue # Skip processing this image

                imgidx = img_idx[imgname]
                img_infos_idx = img_infos[imgidx]
                rgb = np.array(Image.open(os.path.join(rgb_dir, img_infos_idx['path'])))
                mask = np.array(Image.open(os.path.join(mask_dir, img_infos_idx['path'][:-3] + 'png')))

                _, _, K, rgb, mask = undistort_images(img_infos_idx['intrinsics'], rgb, mask)

                # rescale_image_depthmap assumes opencv intrinsics
                intrinsics = geometry.colmap_to_opencv_intrinsics(K)
                image, mask, intrinsics = rescale_image_depthmap(
                    rgb, mask, intrinsics, (target_resolution, target_resolution * 3.0 / 4))

                W, H = image.size
                intrinsics = geometry.opencv_to_colmap_intrinsics(intrinsics)

                # update inpace img_infos_idx
                img_infos_idx['intrinsics'] = intrinsics
                rgb_outpath = os.path.join(output_dir_scene_rgb, img_infos_idx['path'][:-3] + 'jpg')
                image.save(rgb_outpath)

                depth_outpath = os.path.join(output_dir_scene_depth, img_infos_idx['path'][:-3] + 'png')
                # render depth image
                renderer.viewport_width, renderer.viewport_height = W, H
                fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
                camera = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy, znear=znear, zfar=zfar)
                camera_node = pyrender_scene.add(camera, pose=img_infos_idx['cam_to_world'] @ OPENGL_TO_OPENCV)

                depth = renderer.render(pyrender_scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
                pyrender_scene.remove_node(camera_node)  # dont forget to remove camera

                depth = (depth * 1000).astype('uint16')
                # invalidate depth from mask before saving
                depth_mask = (mask < 255)
                depth[depth_mask] = 0
                Image.fromarray(depth).save(depth_outpath)

        # --- Filtering based on missing images ---
        valid_selection = [img for img in selection if img not in missing_images_in_scene]
        if len(missing_images_in_scene) > 0:
            print(f"Scene {scene}: Removed {len(missing_images_in_scene)} images missing from COLMAP: {list(missing_images_in_scene)}")

        # Filter pairs based on valid selection
        selection_indices = {name: i for i, name in enumerate(selection)} # Map original names to original indices
        original_indices_to_keep = {selection_indices[name] for name in valid_selection} # Set of original indices to keep

        if pairs.size > 0: # Check if pairs array is not empty
            valid_pairs_mask = np.isin(pairs[:, 0], list(original_indices_to_keep)) & \
                               np.isin(pairs[:, 1], list(original_indices_to_keep))
            filtered_pairs_old_indices = pairs[valid_pairs_mask]

            # Remap pair indices to match the new valid_selection order
            old_idx_to_new_idx_map = {selection_indices[name]: new_idx for new_idx, name in enumerate(valid_selection)}
            if len(filtered_pairs_old_indices) > 0:
                 final_pairs = np.array([[old_idx_to_new_idx_map[p[0]], old_idx_to_new_idx_map[p[1]]] + list(p[2:])
                                        for p in filtered_pairs_old_indices])
            else:
                 # Handle case with no valid pairs left for the scene
                 final_pairs = np.empty((0, pairs.shape[1]), dtype=pairs.dtype)
                 if len(valid_selection) > 0: # Only print warning if there were images but no pairs
                     print(f"Scene {scene}: No valid pairs remaining after filtering missing images.")
        else: # Handle case where original pairs array was empty
            final_pairs = np.empty((0, 0), dtype=pairs.dtype) # Adjust shape if needed, or use pairs.shape[1] if known

        # --- Collect trajectories and intrinsics for valid images ONLY ---
        trajectories = []
        intrinsics = []
        # Use the main img_infos_dslr dict which should contain info for all processed images
        for imgname in valid_selection:
             if imgname.startswith('DSC'):
                 full_imgname = imgname + '.JPG'
                 # We assume img_idx_dslr and img_infos_dslr contain info for all *originally* found images
                 if full_imgname in img_idx_dslr:
                     imgidx = img_idx_dslr[full_imgname]
                     # Check if imgidx exists in img_infos_dslr (it should if it was processed successfully)
                     if imgidx in img_infos_dslr:
                         img_infos_idx = img_infos_dslr[imgidx]
                         # Ensure intrinsics and pose were updated during processing
                         if 'intrinsics' in img_infos_idx and 'cam_to_world' in img_infos_idx:
                             intrinsics.append(img_infos_idx['intrinsics'])
                             trajectories.append(img_infos_idx['cam_to_world'])
                         else:
                             print(f"Error: Missing processed data (intrinsics/pose) for image index {imgidx} ({full_imgname}) in scene {scene}")
                     else:
                          # This might happen if the image existed in images.txt but failed processing earlier?
                          print(f"Error: Could not find info dict for supposedly valid image index {imgidx} ({full_imgname}) in scene {scene}")
                 else:
                     # This should not happen because valid_selection is derived from images found in img_idx_dslr
                     print(f"Error: Could not find index for supposedly valid image {full_imgname} in scene {scene}")
             # Handle iphone case if uncommented later
             # elif imgname.startswith('frame_'):
             #     # Ensure similar logic for iphone data if used
             #     continue

        # Check consistency
        if len(trajectories) != len(valid_selection):
             print(f"Warning: Mismatch in trajectory count ({len(trajectories)}) and valid selection count ({len(valid_selection)}) for scene {scene}. Metadata might be incomplete.")
             # Decide if this should be a fatal error

        # Stack collected data, handling empty lists
        intrinsics_stack = np.stack(intrinsics, axis=0) if intrinsics else np.empty((0, 3, 3), dtype=np.float64) # Specify dtype if empty
        trajectories_stack = np.stack(trajectories, axis=0) if trajectories else np.empty((0, 4, 4), dtype=np.float64) # Specify dtype if empty

        # --- Save filtered metadata for this scene ---
        np.savez(scene_metadata_path,
                 trajectories=trajectories_stack,
                 intrinsics=intrinsics_stack,
                 images=np.array(valid_selection), # Save the filtered list as numpy array
                 pairs=final_pairs) # Save the filtered and remapped pairs

        # Clean up scene-specific data
        del img_idx_dslr, img_infos_dslr, points3D_dslr, observations_dslr # Clean up dslr sfm data
        # del img_idx_iphone, img_infos_iphone, points3D_iphone, observations_iphone # If iphone data is loaded
        del pyrender_scene

    # concat all scene_metadata.npz into a single file
    scene_data = {}
    for scene_subdir in scenes:
        scene_metadata_path = osp.join(output_dir, scene_subdir, 'scene_metadata.npz')
        with np.load(scene_metadata_path) as data:
            trajectories = data['trajectories']
            intrinsics = data['intrinsics']
            images = data['images']
            pairs = data['pairs']
        scene_data[scene_subdir] = {'trajectories': trajectories,
                                    'intrinsics': intrinsics,
                                    'images': images,
                                    'pairs': pairs}

    offset = 0
    counts = []
    scenes = []
    sceneids = []
    images = []
    intrinsics = []
    trajectories = []
    pairs = []
    for scene_idx, (scene_subdir, data) in enumerate(scene_data.items()):
        num_imgs = data['images'].shape[0]
        img_pairs = data['pairs']

        scenes.append(scene_subdir)
        sceneids.extend([scene_idx] * num_imgs)

        images.append(data['images'])

        intrinsics.append(data['intrinsics'])
        trajectories.append(data['trajectories'])

        # offset pairs
        img_pairs[:, 0:2] += offset
        pairs.append(img_pairs)
        counts.append(offset)

        offset += num_imgs

    images = np.concatenate(images, axis=0)
    intrinsics = np.concatenate(intrinsics, axis=0)
    trajectories = np.concatenate(trajectories, axis=0)
    pairs = np.concatenate(pairs, axis=0)
    np.savez(osp.join(output_dir, 'all_metadata.npz'),
             counts=counts,
             scenes=scenes,
             sceneids=sceneids,
             images=images,
             intrinsics=intrinsics,
             trajectories=trajectories,
             pairs=pairs)
    print('all done')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.pyopengl_platform.strip():
        os.environ['PYOPENGL_PLATFORM'] = args.pyopengl_platform
    process_scenes(args.scannetpp_dir, args.precomputed_pairs, args.output_dir, args.target_resolution)
