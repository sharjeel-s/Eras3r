# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo for comparing two models
# --------------------------------------------------------
import argparse
import math
import builtins
import datetime
import gradio
import os
import torch
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile # Added for main

from dust3r.inference import inference, load_model # Added load_model
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser(description="DUSt3R Model Comparison Demo")
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    
    # Model 1 arguments
    parser_model1 = parser.add_argument_group('Model 1 Specification (required)')
    model1_weights_group = parser_model1.add_mutually_exclusive_group(required=True)
    model1_weights_group.add_argument("--weights1", type=str, help="path to model 1 weights", default=None)
    model1_weights_group.add_argument("--model_name1", type=str, help="name of model 1 weights",
                                      choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                               "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                               "DUSt3R_ViTLarge_BaseDecoder_224_linear"], default=None)

    # Model 2 arguments
    parser_model2 = parser.add_argument_group('Model 2 Specification (required)')
    model2_weights_group = parser_model2.add_mutually_exclusive_group(required=True)
    model2_weights_group.add_argument("--weights2", type=str, help="path to model 2 weights", default=None)
    model2_weights_group.add_argument("--model_name2", type=str, help="name of model 2 weights",
                                      choices=["DUSt3R_ViTLarge_BaseDecoder_512_dpt",
                                               "DUSt3R_ViTLarge_BaseDecoder_512_linear",
                                               "DUSt3R_ViTLarge_BaseDecoder_224_linear"], default=None)
    
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def set_print_with_timestamp(time_format="%Y-%m-%d %H:%M:%S"):
    builtin_print = builtins.print

    def print_with_timestamp(*args, **kwargs):
        now = datetime.datetime.now()
        formatted_date_time = now.strftime(time_format)

        builtin_print(f'[{formatted_date_time}] ', end='')  # print with time stamp
        builtin_print(*args, **kwargs)

    builtins.print = print_with_timestamp


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, basename='scene.glb'): # Added basename
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, basename) # Use basename
    if not silent:
        print(f'(exporting 3D scene to {outfile})')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, basename='scene.glb'): # Added basename
    if scene is None:
        return None
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent, basename=basename) # Pass basename


def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, basename='scene.glb'): # Added basename
    if not filelist:
        return None, None, []
        
    imgs = load_images(filelist, size=image_size)
    if not imgs: # Handle case where no images could be loaded
        print("Warning: No images loaded.")
        return None, None, []

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        if not silent:
            print("Only one image provided, duplicating it for pair processing.")
            
    effective_scenegraph_type = scenegraph_type
    if scenegraph_type == "swin":
        effective_scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        effective_scenegraph_type = scenegraph_type + "-" + str(refid if refid < len(imgs) else 0)


    pairs = make_pairs(imgs, scene_graph=effective_scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1)

    mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
    scene = global_aligner(output, device=device, mode=mode)
    
    if scene is None: # global_aligner can return None if output is empty
        print("Warning: Global aligner did not produce a scene.")
        return None, None, []

    lr = 0.01
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)
        if not silent:
            pl.plot(to_numpy(loss))
            pl.show()


    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, basename=basename) # Pass basename

    rgbimg_out = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf])
    
    output_imgs = []
    if rgbimg_out is not None and len(rgbimg_out) > 0:
        cmap = pl.get_cmap('jet')
        
        valid_depths = [d for d in depths if d is not None and d.size > 0]
        depths_max = max([d.max() for d in valid_depths]) if valid_depths else 1.0
        if depths_max == 0: depths_max = 1.0 # Avoid division by zero

        valid_confs = [c for c in confs if c is not None and c.size > 0]
        confs_max = max([c.max() for c in valid_confs]) if valid_confs else 1.0
        if confs_max == 0: confs_max = 1.0


        for i in range(len(rgbimg_out)):
            output_imgs.append(rgbimg_out[i])
            current_depth = depths[i] if i < len(depths) and depths[i] is not None else np.zeros_like(rgbimg_out[i][:,:,0], dtype=np.float32)
            output_imgs.append(rgb(current_depth / depths_max))
            
            current_conf = confs[i] if i < len(confs) and confs[i] is not None else np.zeros_like(rgbimg_out[i][:,:,0], dtype=np.float32)
            output_imgs.append(rgb(cmap(current_conf / confs_max)))


    return scene, outfile, output_imgs


def set_scenegraph_options(inputfiles, winsize_val, refid_val, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 0
    max_winsize = max(1, math.ceil((num_files - 1) / 2)) if num_files > 1 else 1
    max_refid = max(0, num_files - 1)

    new_winsize_val = min(winsize_val, max_winsize)
    new_refid_val = min(refid_val, max_refid)

    winsize_visible = scenegraph_type == "swin"
    refid_visible = scenegraph_type == "oneref"

    return gradio.Slider(label="Scene Graph: Window Size", value=new_winsize_val,
                         minimum=1, maximum=max_winsize, step=1, visible=winsize_visible, interactive=True), \
           gradio.Slider(label="Scene Graph: Ref Id", value=new_refid_val, minimum=0,
                         maximum=max_refid, step=1, visible=refid_visible, interactive=True)


def main_demo(tmpdirname, model1, model_name1_str, model2, model_name2_str, device, image_size, server_name, server_port, silent=False):
    # Create separate temp subdirectories for each model's output files
    tmpdir_model1 = os.path.join(tmpdirname, "model1_out")
    tmpdir_model2 = os.path.join(tmpdirname, "model2_out")
    os.makedirs(tmpdir_model1, exist_ok=True)
    os.makedirs(tmpdir_model2, exist_ok=True)

    # Partial functions for reconstructing and generating 3D model for model 1
    # Note: 'basename' is hardcoded here for simplicity, as tmpdir is unique
    model_from_scene_fun1 = functools.partial(get_3D_model_from_scene, tmpdir_model1, silent, basename='scene_model1.glb')
    
    # Partial functions for model 2
    model_from_scene_fun2 = functools.partial(get_3D_model_from_scene, tmpdir_model2, silent, basename='scene_model2.glb')

    def run_reconstruction_for_both(inputfiles_paths, schedule_val, niter_val, 
                                    scenegraph_type_val, winsize_val, refid_val,
                                    # Model 1 post-proc params for initial GLB
                                    m1_min_conf, m1_as_pc, m1_mask_sky, m1_clean_depth, m1_trans_cams, m1_cam_size,
                                    # Model 2 post-proc params for initial GLB
                                    m2_min_conf, m2_as_pc, m2_mask_sky, m2_clean_depth, m2_trans_cams, m2_cam_size):
        
        gradio.Info("Processing Model 1...")
        scene_obj1, outfile1, imgs1 = get_reconstructed_scene(
            tmpdir_model1, model1, device, silent, image_size,
            inputfiles_paths, schedule_val, niter_val, 
            m1_min_conf, m1_as_pc, m1_mask_sky, m1_clean_depth, m1_trans_cams, m1_cam_size,
            scenegraph_type_val, winsize_val, refid_val, basename='scene_model1.glb'
        )
        gradio.Info("Processing Model 2...")
        scene_obj2, outfile2, imgs2 = get_reconstructed_scene(
            tmpdir_model2, model2, device, silent, image_size,
            inputfiles_paths, schedule_val, niter_val,
            m2_min_conf, m2_as_pc, m2_mask_sky, m2_clean_depth, m2_trans_cams, m2_cam_size,
            scenegraph_type_val, winsize_val, refid_val, basename='scene_model2.glb'
        )
        gradio.Info("Processing complete.")
        return scene_obj1, outfile1, imgs1, scene_obj2, outfile2, imgs2

    with gradio.Blocks(css=""".gradio-container {min-width: 100% !important;}""", title="DUSt3R Model Comparison Demo") as demo:
        scene1 = gradio.State(None)
        scene2 = gradio.State(None)

        gradio.HTML('<h1 style="text-align: center;">DUSt3R Model Comparison Demo</h1>')
        
        with gradio.Row():
            inputfiles = gradio.File(file_count="multiple", label="Upload Images (2 or more recommended)")
        
        with gradio.Accordion("Shared Reconstruction Parameters", open=True):
            with gradio.Row():
                schedule = gradio.Dropdown(["linear", "cosine"], value='linear', label="Alignment Schedule", info="For global alignment optimization.")
                niter = gradio.Number(value=100, precision=0, minimum=0, maximum=5000, label="Alignment Iterations", info="For global alignment.") # Reduced default for speed
            with gradio.Row():
                scenegraph_type = gradio.Dropdown([("complete: all pairs", "complete"),
                                                   ("swin: sliding window", "swin"),
                                                   ("oneref: one image to all others", "oneref")],
                                                  value='complete', label="Scene Graph Type",
                                                  info="Defines how image pairs are made for matching.",
                                                  interactive=True)
                winsize = gradio.Slider(label="Window Size (for swin)", value=3, minimum=1, maximum=10, step=1, visible=False, interactive=True)
                refid = gradio.Slider(label="Reference ID (for oneref)", value=0, minimum=0, maximum=0, step=1, visible=False, interactive=True)
        
        run_btn = gradio.Button("Run Reconstruction for Both Models", variant="primary")

        with gradio.Row():
            # --- Column for Model 1 ---
            with gradio.Column(scale=1):
                gradio.HTML(f'<h3 style="text-align: center;">Model 1: {model_name1_str}</h3>')
                with gradio.Accordion("Model 1 - Visualization Settings", open=False):
                    min_conf_thr1 = gradio.Slider(label="Min Confidence Thr.", value=3.0, minimum=1.0, maximum=20, step=0.1)
                    cam_size1 = gradio.Slider(label="Camera Size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                    as_pointcloud1 = gradio.Checkbox(value=False, label="Output as Pointcloud")
                    mask_sky1 = gradio.Checkbox(value=False, label="Mask Sky (post-process)")
                    clean_depth1 = gradio.Checkbox(value=True, label="Clean-up Depthmaps (post-process)")
                    transparent_cams1 = gradio.Checkbox(value=False, label="Transparent Cameras")
                
                outmodel1 = gradio.Model3D(label="Model 1 3D Output", height=400)
                outgallery1 = gradio.Gallery(label='Model 1: RGB, Depth, Confidence', columns=3, height="auto", object_fit="contain")

            # --- Column for Model 2 ---
            with gradio.Column(scale=1):
                gradio.HTML(f'<h3 style="text-align: center;">Model 2: {model_name2_str}</h3>')
                with gradio.Accordion("Model 2 - Visualization Settings", open=False):
                    min_conf_thr2 = gradio.Slider(label="Min Confidence Thr.", value=3.0, minimum=1.0, maximum=20, step=0.1)
                    cam_size2 = gradio.Slider(label="Camera Size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
                    as_pointcloud2 = gradio.Checkbox(value=False, label="Output as Pointcloud")
                    mask_sky2 = gradio.Checkbox(value=False, label="Mask Sky (post-process)")
                    clean_depth2 = gradio.Checkbox(value=True, label="Clean-up Depthmaps (post-process)")
                    transparent_cams2 = gradio.Checkbox(value=False, label="Transparent Cameras")

                outmodel2 = gradio.Model3D(label="Model 2 3D Output", height=400)
                outgallery2 = gradio.Gallery(label='Model 2: RGB, Depth, Confidence', columns=3, height="auto", object_fit="contain")

        # Events for scenegraph options
        scenegraph_type.change(set_scenegraph_options,
                               inputs=[inputfiles, winsize, refid, scenegraph_type],
                               outputs=[winsize, refid])
        inputfiles.change(set_scenegraph_options,
                          inputs=[inputfiles, winsize, refid, scenegraph_type],
                          outputs=[winsize, refid])

        # Run button click
        run_btn.click(fn=run_reconstruction_for_both,
                      inputs=[inputfiles, schedule, niter, scenegraph_type, winsize, refid,  # Shared
                              min_conf_thr1, as_pointcloud1, mask_sky1, clean_depth1, transparent_cams1, cam_size1, # M1
                              min_conf_thr2, as_pointcloud2, mask_sky2, clean_depth2, transparent_cams2, cam_size2  # M2
                             ],
                      outputs=[scene1, outmodel1, outgallery1, scene2, outmodel2, outgallery2])

        # Model 1 interactive controls
        model1_controls = [min_conf_thr1, as_pointcloud1, mask_sky1, clean_depth1, transparent_cams1, cam_size1]
        for ctrl in model1_controls:
            if isinstance(ctrl, gradio.Slider):
                ctrl.release(fn=model_from_scene_fun1,
                             inputs=[scene1, min_conf_thr1, as_pointcloud1, mask_sky1,
                                     clean_depth1, transparent_cams1, cam_size1],
                             outputs=outmodel1)
            else: # Checkbox
                ctrl.change(fn=model_from_scene_fun1,
                            inputs=[scene1, min_conf_thr1, as_pointcloud1, mask_sky1,
                                    clean_depth1, transparent_cams1, cam_size1],
                            outputs=outmodel1)
        
        # Model 2 interactive controls
        model2_controls = [min_conf_thr2, as_pointcloud2, mask_sky2, clean_depth2, transparent_cams2, cam_size2]
        for ctrl in model2_controls:
            if isinstance(ctrl, gradio.Slider):
                ctrl.release(fn=model_from_scene_fun2,
                             inputs=[scene2, min_conf_thr2, as_pointcloud2, mask_sky2,
                                     clean_depth2, transparent_cams2, cam_size2],
                             outputs=outmodel2)
            else: # Checkbox
                ctrl.change(fn=model_from_scene_fun2,
                            inputs=[scene2, min_conf_thr2, as_pointcloud2, mask_sky2,
                                    clean_depth2, transparent_cams2, cam_size2],
                            outputs=outmodel2)

    demo.queue().launch(share=True, server_name=server_name, server_port=server_port)


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    if not args.silent:
        set_print_with_timestamp()
        print('args', args)

    if args.tmp_dir is not None:
        tempfile.tempdir = args.tmp_dir

    # Determine model paths/names
    model_path1 = args.weights1 if args.weights1 else args.model_name1
    model_path2 = args.weights2 if args.weights2 else args.model_name2
    
    model_name1_str = os.path.basename(model_path1) if args.weights1 else args.model_name1
    model_name2_str = os.path.basename(model_path2) if args.weights2 else args.model_name2


    print(f"Loading Model 1: {model_name1_str}...")
    model1 = load_model(model_path1, args.device)
    print(f"Loading Model 2: {model_name2_str}...")
    model2 = load_model(model_path2, args.device)

    server_name = "0.0.0.0" if args.local_network else args.server_name

    with tempfile.TemporaryDirectory(prefix='dust3r_comparison_gradio_') as tmpdirname:
        print(f"Using temporary directory: {tmpdirname}")
        main_demo(tmpdirname, model1, model_name1_str, model2, model_name2_str, 
                  args.device, args.image_size, server_name, args.server_port, args.silent)