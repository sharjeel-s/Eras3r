import torch
import torch.nn as nn
import torch.nn.functional as F
from dust3r.heads.postprocess import postprocess


class BiMonoHead(nn.Module):
    """ 
    predicts if each point is taken as a monocular or binocular point
    Each token outputs: - 16x16 3D points (+ confidence + bi-mono flag)
    """

    def __init__(self, net, has_conf=False, has_scene_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.scene_conf_mode = net.scene_conf_mode
        self.has_conf = has_conf
        self.has_scene_conf = has_scene_conf
        self.sig = nn.Sigmoid()

        self.bi_mono_flag = nn.Linear(net.dec_embed_dim, self.patch_size**2)  # B,S,16*16
        self.flag_proj = nn.Linear(net.dec_embed_dim + self.patch_size**2, (3 + has_conf + has_scene_conf) * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        bi_mono_flag = self.bi_mono_flag(tokens)
        bi_mono_flag = self.sig(bi_mono_flag)
        #bi_mono_flag = bi_mono_flag.sigmoid()
        #bi_mono_flag_out = bi_mono_flag.repeat(1, self.patch_size**2)  # B,S,16*16
        bi_mono_flag_out = bi_mono_flag.view(B, -1, H//self.patch_size, W//self.patch_size) 

        tokens = torch.cat([tokens, bi_mono_flag], dim=-1)

        feat = self.flag_proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)        
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W
        bi_mono_flag_out = F.pixel_shuffle(bi_mono_flag_out, self.patch_size)  # B,1,H,W
        feat = torch.cat([feat, bi_mono_flag_out], dim=1)
        return postprocess(feat, self.depth_mode, self.conf_mode, self.scene_conf_mode, bi_mono_flag=True)


    