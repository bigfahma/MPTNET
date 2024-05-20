from models.Decoder import Decoder
from models.Encoder import Encoder
from models.PatchEmbedding import PatchEmbed
import torch.nn as nn
import torch
from models.utils import to_3tuple
import numpy as np
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from monai.data import decollate_batch

class MPTNet(nn.Module):
    def __init__(self, emb_dim, resolution, num_classes, depth, num_heads, window_size, n_layers = 3, pooling_size = [(1,2,2), (1,4,4), (1,8,8)], emb_ratio = [1, 4, 4]):
        super().__init__()
        
        D, H, W = resolution
        downD, downH, downW = emb_ratio
        print(downD)
        if downD == 1 : #CONFIG 2 ACDC
            out_proj1 = (4,2,2)
            out_proj2 = (4,2,2)
        elif downD == 2 :
            out_proj1 = (4,2,2) # CONFIG 1 
            out_proj2 = (2,2,2)
        elif downD == 3:
            out_proj1 = (2,2,2) #CONFIG 3
            out_proj2 = (2,2,2)

        window_size = to_3tuple(window_size)
        De, He, We = D//downD, H//downH, W//downW
        Dd, Hd, Wd = De//(2**n_layers), He//(2**n_layers), We//(2**n_layers)
        in_dim_encoder = [De, He, We]
        in_dim_decoder = [Dd, Hd, Wd]
        self.patch_embedding = PatchEmbed(patch_size=4, in_chans=1, embed_dim=emb_dim, 
                                 norm_layer=nn.LayerNorm, out_proj1 = out_proj1, out_proj2 = out_proj2)
        self.mrha_encoder = Encoder(resolution= in_dim_encoder, emb_dim= emb_dim, 
                                        num_heads = num_heads, window_size = window_size, n_layers=n_layers, pooling_size= pooling_size)
        self.mrha_decoder = Decoder(dim = (2**n_layers)*emb_dim, resolution = in_dim_decoder, num_classes=num_classes,
                                        depth = depth, num_heads = num_heads, 
                                         window_size = window_size, stride =(downD, downH, downW), n_layers=n_layers)

    def forward(self, x):
        #print("Input :",x.shape)
        x = self.patch_embedding(x)
        #print("After PE",x.shape)
        x, skips = self.mrha_encoder(x)
        #print("Output Encoder :",x.shape)
        x = self.mrha_decoder(x, skips)
        return x
