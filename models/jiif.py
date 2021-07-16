import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import make_coord
from models.edsr import make_edsr_baseline

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class JIIF(nn.Module):

    def __init__(self, args, feat_dim=128, guide_dim=128, mlp_dim=[1024,512,256,128]):
        super().__init__()
        self.args = args
        self.feat_dim = feat_dim
        self.guide_dim = guide_dim
        self.mlp_dim = mlp_dim

        self.image_encoder = make_edsr_baseline(n_feats=self.guide_dim, n_colors=3)
        self.depth_encoder = make_edsr_baseline(n_feats=self.feat_dim, n_colors=1)

        imnet_in_dim = self.feat_dim + self.guide_dim * 2 + 2
        
        self.imnet = MLP(imnet_in_dim, out_dim=2, hidden_list=self.mlp_dim)
        
    def query(self, feat, coord, hr_guide, lr_guide, image):
        
        # feat: [B, C, h, w]
        # coord: [B, N, 2], N <= H * W
          
        b, c, h, w = feat.shape # lr
        B, N, _ = coord.shape

        # LR centers' coords
        feat_coord = make_coord((h, w), flatten=False).to(feat.device).permute(2, 0, 1).unsqueeze(0).expand(b, 2, h, w)

        q_guide_hr = F.grid_sample(hr_guide, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, C]

        rx = 1 / h
        ry = 1 / w

        preds = []
        
        k = 0
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()

                coord_[:, :, 0] += (vx) * rx 
                coord_[:, :, 1] += (vy) * ry 
                k += 1

                # feat: [B, c, h, w], coord_: [B, N, 2] --> [B, 1, N, 2], out: [B, c, 1, N] --> [B, c, N] --> [B, N, c]
                q_feat = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, c]
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= h
                rel_coord[:, :, 1] *= w

                q_guide_lr = F.grid_sample(lr_guide, coord_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1) # [B, N, C]
                q_guide = torch.cat([q_guide_hr, q_guide_hr - q_guide_lr], dim=-1)

                inp = torch.cat([q_feat, q_guide, rel_coord], dim=-1)

                pred = self.imnet(inp.view(B * N, -1)).view(B, N, -1) # [B, N, 2]
                preds.append(pred)

        preds = torch.stack(preds, dim=-1) # [B, N, 2, kk]
        weight = F.softmax(preds[:,:,1,:], dim=-1)

        ret = (preds[:,:,0,:] * weight).sum(-1, keepdim=True)

        return ret

    def forward(self, data):
        image, depth, coord, res, lr_image = data['image'], data['lr'], data['hr_coord'], data['lr_pixel'], data['lr_image']

        hr_guide = self.image_encoder(image)
        lr_guide = self.image_encoder(lr_image)

        feat = self.depth_encoder(depth)

        if self.training or not self.args.batched_eval:
            res = res + self.query(feat, coord, hr_guide, lr_guide, data['hr_depth'].repeat(1,3,1,1))

        # batched evaluation to avoid OOM
        else:
            N = coord.shape[1] # coord ~ [B, N, 2]
            n = 30720
            tmp = []
            for start in range(0, N, n):
                end = min(N, start + n)
                ans = self.query(feat, coord[:, start:end], hr_guide, lr_guide, data['hr_depth'].repeat(1,3,1,1)) # [B, N, 1]
                tmp.append(ans)
            res = res + torch.cat(tmp, dim=1)

        return res


