import torch.nn as nn
import torch.nn.functional as F
from models.utils import BasicLayerTr,SwinTransformerBlock_kv, get_attention_mask
import torch
from models.utils import to_3tuple, UpsampleConvUnit


class AttentionGate(nn.Module):

    def __init__(self,
                 dim,
                 resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                ):
        super().__init__()
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2)

        self.ag = SwinTransformerBlock_kv(
                    dim=dim,
                    in_dim = resolution,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=0 ,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[0] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                   
    def forward(self, skip, x_up):
        D, H, W = (self.resolution[0], self.resolution[1], self.resolution[2])
        x = x_up + skip
        attn_mask = get_attention_mask(D, H, W, window_size=self.window_size, shift_size=self.shift_size, device=x.device)
        x = self.ag(x, attn_mask, skip=skip, x_up=x_up)
        return x

    
class Dec_layer(nn.Module):
   
    def __init__(self,
                 dim,
                 resolution,
                 depth,
                 num_heads,
                 window_size=7,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.window_size = window_size
        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2)
        self.in_dim_kv  = (resolution[0]*2, resolution[1]*2, resolution[2]*2)

        self.basic_layer_tr = BasicLayerTr(dim=self.dim, resolution=self.resolution, depth=depth, 
                             num_heads=num_heads, window_size=window_size, norm_layer=norm_layer)
        self.inv_conv = UpsampleConvUnit(in_channels= self.dim, out_channels= self.dim //2)
        self.attention_gate = AttentionGate(dim = self.dim //2, resolution = self.in_dim_kv, num_heads = num_heads,
                                            window_size=self.window_size)

    def forward(self, x, skip):
        D, H, W = self.resolution
        x = self.basic_layer_tr(x)
        x = x.view(-1, D, H, W, self.dim).permute(0, 4, 1, 2, 3).contiguous()
        x = self.inv_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.attention_gate(skip, x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim, resolution, num_classes, depth, num_heads, window_size,stride = [4,2,2], n_layers=3):
        super(Decoder, self).__init__()
        self.dim = dim
        self.D_in, self.H_in, self.W_in = resolution

        # Dynamic creation of mrha_dec layers based on n_layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer_dim = dim // (2**i)
            layer_resolution = (self.D_in * (2**i), self.H_in * (2**i), self.W_in * (2**i))
            self.layers.append(Dec_layer(dim=layer_dim, resolution=layer_resolution, depth=depth, 
                                         num_heads=num_heads[n_layers-i-1], window_size=window_size))

        self.up_final = nn.ConvTranspose3d(dim // (2**n_layers), num_classes, kernel_size=stride, stride=stride)

    def forward(self, x, skips):
        skips = skips[::-1]
        for i, layer in enumerate(self.layers):
            x = layer(x, skips[i])
            #print(x.shape)

        # Final upsampling
        D, H, W = self.D_in * (2**len(self.layers)), self.H_in * (2**len(self.layers)), self.W_in * (2**len(self.layers))
        x = x.view(-1, D, H, W, self.dim // (2**len(self.layers))).permute(0, 4, 1, 2, 3).contiguous()
        out = self.up_final(x)
        return out
