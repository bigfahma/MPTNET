from timm.models.layers import to_3tuple
import torch.nn as nn
import torch.nn.functional as F


class project(nn.Module):
    def __init__(self,in_dim,out_dim,stride,padding,activate,norm,last=False):
        super().__init__()
        self.out_dim=out_dim
        self.conv1=nn.Conv3d(in_dim,out_dim,kernel_size=3,stride=stride,padding=padding)
        self.conv2=nn.Conv3d(out_dim,out_dim,kernel_size=3,stride=1,padding=1)
        self.activate=activate()
        self.norm1=norm(out_dim)
        self.last=last  
        if not last:
            self.norm2=norm(out_dim)
            
    def forward(self,x):
        x=self.conv1(x)
        x=self.activate(x)
        Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wd, Wh, Ww)
        x=self.conv2(x)
        if not self.last:
            x=self.activate(x)
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Wd, Wh, Ww)
        return x
        
class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, out_proj1= (4,2,2), out_proj2= (2,2,2)):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        stride1=[patch_size[0]//out_proj1[0], patch_size[1]//out_proj1[1], patch_size[2]//out_proj1[2]]
        stride2=[patch_size[0]//out_proj2[0], patch_size[1]//out_proj2[1], patch_size[2]//out_proj2[2]]
        self.proj1 = project(in_chans,embed_dim//2,stride1,1,nn.GELU,nn.LayerNorm,False)
        self.proj2 = project(embed_dim//2,embed_dim,stride2,1,nn.GELU,nn.LayerNorm,True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, self.patch_size[0] - D % self.patch_size[0]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[2] - W % self.patch_size[2]))
        x = self.proj1(x) 
        x = self.proj2(x) 

        if self.norm is not None:
            Wd, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2) 
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wd, Wh, Ww)

        return x
