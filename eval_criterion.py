import argparse
import datetime
import json
import numpy as np
import os
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import AsymmetricCroCo3DStereo, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F401, needed when loading the model
from dust3r.inference import loss_of_one_batch  # noqa

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


# --------- User parameters ---------
DATASET_STR = "ScanNetpp(split='train', ROOT='/cluster/scratch/msharjeel/pair_list', resolution=224, aug_crop=16)"
MODEL_STR = "AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp_reparam', 0, float('inf')), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)"
PRETRAINED_PATH = "/cluster/scratch/msharjeel/checkpoints/dust3r_224_100000_mask/checkpoint-best.pth"
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Replace with your new criterion string or function
NEW_CRITERION_STR = "ExpConfLossNorm(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)"  
# -----------------------------------

def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print(f'Building {split} Data loader for dataset: ', dataset)
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))

    print(f"{split} dataset length: ", len(loader))
    return loader


# 1. Load dataset
dataset = get_data_loader(DATASET_STR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# 2. Load model
model = AsymmetricCroCo3DStereo(pos_embed='RoPE100', img_size=(224, 224), head_type='linear', output_mode='pts3d', depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp_reparam', 0, float('inf')), enc_embed_dim=1024, enc_depth=24, enc_num_heads=16, dec_embed_dim=768, dec_depth=12, dec_num_heads=12)
if PRETRAINED_PATH:
    checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
model = model.to(DEVICE)
model.eval()

# 3. Define criterion
#criterion = ExpConfLossNorm(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2).to(DEVICE)

criterion = ConfLossNorm(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2).to(DEVICE)

# 4. Evaluate loss
total_loss = 0
num_batches = 0

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


with torch.no_grad():
    for batch in dataset:
        # Move batch to device
        batch = move_to_device(batch, DEVICE)

        view1, view2 = batch
        pred1, pred2 = model(view1, view2)

        # Compute loss
        loss = criterion(view1, view2, pred1, pred2)
        loss_value, loss_details = loss
        print("bro please normal", loss_value)
        num_batches += 1

print(f"Average loss over dataset: {total_loss / num_batches if num_batches > 0 else float('nan')}")