import torch.nn as nn
import torch
import numpy as np
from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import torch.nn.functional as F


class ContiguousGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.contiguous()

class PatchMerging(nn.Module):
  

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=3,stride=2,padding=1)
       
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,2*C)
      
        return x
    
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads)) 

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w], indexing= "xy"))  
        coords_flatten = torch.flatten(coords, 1)  
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1) 
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None,pos_embed=None):

        B_, N, C = x.shape
        
        qkv = self.qkv(x)
        
        qkv=qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x+pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, in_dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(shift_size)
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        S, H, W = self.in_dim
        assert L == S * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))
        _, Sp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifts = (-self.shift_size[0], -self.shift_size[1], -self.shift_size[2])
            shifted_x = torch.roll(x, shifts=shifts, dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)

        # reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifts = (self.shift_size[0], self.shift_size[1], self.shift_size[2])
            x = torch.roll(shifted_x, shifts=shifts, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchMerging(nn.Module):
  

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Conv3d(dim,dim*2,kernel_size=3,stride=2,padding=1)
       
        self.norm = norm_layer(dim)

    def forward(self, x, S, H, W):

        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"
        x = x.view(B, S, H, W, C)
        
        x = F.gelu(x)
        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x=self.reduction(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,2*C)
      
        return x
class Patch_Expanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
       
        self.norm = norm_layer(dim)
        self.up=nn.ConvTranspose3d(dim,dim//2,2,2)
    def forward(self, x, S, H, W):
      
        
        B, L, C = x.shape
        assert L == H * W * S, "input feature has wrong size"

        x = x.view(B, S, H, W, C)

        x = self.norm(x)
        x=x.permute(0,4,1,2,3).contiguous()
        x = self.up(x)
        x = ContiguousGrad.apply(x)
        x=x.permute(0,2,3,4,1).contiguous().view(B,-1,C//2)
       
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    window_size_D, window_size_H, window_size_W = window_size
    x = x.view(B, D // window_size_D, window_size_D, H // window_size_H, window_size_H, W // window_size_W, window_size_W, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size_D, window_size_H, window_size_W, C)
    return windows


def window_reverse(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / (window_size[0] * window_size[1] * window_size[2])))
    window_size_D, window_size_H, window_size_W = window_size
    x = windows.view(B, D // window_size_D, H // window_size_H, W // window_size_W, window_size_D, window_size_H, window_size_W, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def get_attention_mask(D, H, W, window_size, shift_size, device):
    window_size_D, window_size_H, window_size_W = window_size
    shift_size_D, shift_size_H, shift_size_W = shift_size

    Dp = int(np.ceil(D / window_size_D)) * window_size_D
    Hp = int(np.ceil(H / window_size_H)) * window_size_H
    Wp = int(np.ceil(W / window_size_W)) * window_size_W

    img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)

    s_slices = (slice(0, -window_size_D),
                slice(-window_size_D, -shift_size_D),
                slice(-shift_size_D, None))
    h_slices = (slice(0, -window_size_H),
                slice(-window_size_H, -shift_size_H),
                slice(-shift_size_H, None))
    w_slices = (slice(0, -window_size_W),
                slice(-window_size_W, -shift_size_W),
                slice(-shift_size_W, None))

    cnt = 0
    for s in s_slices:
        for h in h_slices:
            for w in w_slices:
                img_mask[:, s, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size_D * window_size_H * window_size_W)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask

        
class BasicLayerTr(nn.Module):
   
    def __init__(self,
                 dim,
                 resolution,
                 depth,
                 num_heads,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = to_3tuple(window_size)
        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2)
        self.depth = depth
        self.D, self.H, self.W = resolution

        # Adjust shift_size for non-uniform window_size
        self.shift_size = [max(0, min(shift, w-1)) for shift, w in zip(self.shift_size, self.window_size)]

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                in_dim=resolution,
                num_heads=num_heads,
                window_size=self.window_size,
                shift_size= self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        attn_mask = get_attention_mask(self.D, self.H, self.W, self.window_size, self.shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return x


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention_kv(nn.Module):
   
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  

        # get pair-wise relative position index for each token inside the window
        coords_s = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_s, coords_h, coords_w], indexing = "xy")) 
        coords_flatten = torch.flatten(coords, 1) 
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() 
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= 3 * self.window_size[1] - 1
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)  
        self.register_buffer("relative_position_index", relative_position_index)

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        trunc_normal_(self.relative_position_bias_table, std=.02)


    def forward(self, skip,x_up,pos_embed=None, mask=None):

        B_, N, C = skip.shape
        
        kv = self.kv(skip)
        q = x_up

        kv=kv.reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q = q.reshape(B_,N,self.num_heads,C//self.num_heads).permute(0,2,1,3).contiguous()
        k,v = kv[0], kv[1]  
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()
        if pos_embed is not None:
            x = x + pos_embed
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock_kv(nn.Module):
    def __init__(self, dim, in_dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.num_heads = num_heads
        self.window_size = to_3tuple(window_size)
        self.shift_size = to_3tuple(shift_size)
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_kv(
                dim, window_size=self.window_size, num_heads=num_heads,
                qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       
    def forward(self, x, mask_matrix, skip=None, x_up=None):
        B, L, C = x.shape
        S, H, W = self.in_dim
        assert L == S * H * W, "input feature has wrong size"
        
        shortcut = x
        skip = self.norm1(skip)
        x_up = self.norm1(x_up)

        skip = skip.view(B, S, H, W, C)
        x_up = x_up.view(B, S, H, W, C)
        x = x.view(B, S, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        pad_b = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_g = (self.window_size[0] - S % self.window_size[0]) % self.window_size[0]

        skip = F.pad(skip, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        x_up = F.pad(x_up, (0, 0, 0, pad_r, 0, pad_b, 0, pad_g))  
        _, Sp, Hp, Wp, _ = skip.shape

        # cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifts = (-self.shift_size[0], -self.shift_size[1], -self.shift_size[2])
            skip = torch.roll(skip, shifts=shifts, dims=(1, 2, 3))
            x_up = torch.roll(x_up, shifts=shifts, dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            attn_mask = None

        # partition windows
        skip = window_partition(skip, self.window_size) 
        skip = skip.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  
        x_up = window_partition(x_up, self.window_size) 
        x_up = x_up.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  
        attn_windows = self.attn(skip, x_up, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Sp, Hp, Wp)  

        # reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            shifts = (self.shift_size[0], self.shift_size[1], self.shift_size[2])
            x = torch.roll(shifted_x, shifts=shifts, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0 or pad_g > 0:
            x = x[:, :S, :H, :W, :].contiguous()

        x = x.view(B, S * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x