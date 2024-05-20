import torch
import torch.nn as nn
from models.utils import BasicLayerTr
from monai.networks.blocks import ResidualUnit, ResBlock

class MultiPoolingTransformer(nn.Module):
    def __init__(self, resolution, in_channels, num_heads, window_size, pooling_size = [(1,2,2), (1,4,4), (1,8,8)]):
        super(MultiPoolingTransformer, self).__init__()

        self.D, self.H, self.W = resolution
        self.pooling_size = pooling_size 
        self.in_channels = in_channels

        self.pre_pooling_resblock = ResidualUnit(
            spatial_dims=3,
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=3,
            subunits=2, 
            adn_ordering="adn"
        )
        self.pool2 = nn.AvgPool3d(kernel_size= pooling_size[0])
        self.pool4 = nn.AvgPool3d(kernel_size= pooling_size[1])
        self.pool8 = nn.AvgPool3d(kernel_size= pooling_size[2])

        ## If Pool size 2 (1, 4, 4) stride & kernel 2 (1,2,2) (1,2,2)
        ## If pool size 3 (1, 8, 8) stride & kernel 3 (1,4,4) (1, 4, 4)
        ## If Pool size 2 (2, 4, 4) stride & kernel 2 (2,2,2) (2,2,2)
        ## If Pool size 3 (2, 8, 8) stride & kernel 3 (2,4,4) (2,4,4)

        self.sv_transformer2 = BasicLayerTr(dim = self.in_channels, 
                                            resolution=(self.D//pooling_size[0][0], self.H//pooling_size[0][1],self.W//pooling_size[0][2]), 
                                            depth=2, num_heads=num_heads, window_size= window_size)
        self.sv_transformer4 = BasicLayerTr(dim = self.in_channels, 
                                            resolution=(self.D//pooling_size[1][0], self.H//pooling_size[1][1],self.W//pooling_size[1][2]), 
                                            depth=2, num_heads=num_heads, window_size= window_size)
        self.sv_transformer8 = BasicLayerTr(dim = self.in_channels, 
                                            resolution=(self.D//pooling_size[2][0], self.H//pooling_size[2][1],self.W//pooling_size[2][2]), 
                                            depth=2, num_heads=num_heads, window_size= window_size)

        self.inv_conv2 = nn.ConvTranspose3d(self.in_channels, self.in_channels // 2, kernel_size=(pooling_size[1][0], 2, 2), stride=(pooling_size[1][0], 2, 2))
        self.inv_conv3 = nn.ConvTranspose3d(self.in_channels, self.in_channels // 2, kernel_size=(pooling_size[2][0], 4, 4), stride=(pooling_size[2][0], 4, 4))

        self.relu = nn.ReLU(inplace=True)
        self.bn_inv_conv1 = nn.BatchNorm3d(in_channels)
        self.bn_inv_conv2 = nn.BatchNorm3d(in_channels // 2)
        self.bn_inv_conv3 = nn.BatchNorm3d(in_channels // 2)

        self.res_unitf = ResidualUnit(
            spatial_dims=3,
            in_channels=2*self.in_channels,
            out_channels=2*self.in_channels,
            kernel_size=3,
            subunits=2, 
            strides = [2, 1, 1],
            adn_ordering="adn"
        )

    def forward(self, x):

        x = self.pre_pooling_resblock(x)
        # Apply pooling
        x2 = self.pool2(x)
        x4 = self.pool4(x)
        x8 = self.pool8(x)

        x2 = x2.flatten(2).transpose(1, 2).contiguous()
        x4 = x4.flatten(2).transpose(1, 2).contiguous()
        x8 = x8.flatten(2).transpose(1, 2).contiguous()
        # Apply Swin Volume Transformers
        x2 = self.sv_transformer2(x2)
        x4 = self.sv_transformer4(x4)
        x8 = self.sv_transformer8(x8)

        x2 = x2.view(-1, self.D//self.pooling_size[0][0], self.H//self.pooling_size[0][1],self.W//self.pooling_size[0][2], self.in_channels).permute(0, 4, 1, 2, 3).contiguous()
        x4 = x4.view(-1, self.D//self.pooling_size[1][0], self.H//self.pooling_size[1][1],self.W//self.pooling_size[1][2], self.in_channels).permute(0, 4, 1, 2, 3).contiguous()
        x8 = x8.view(-1, self.D//self.pooling_size[2][0], self.H//self.pooling_size[2][1],self.W//self.pooling_size[2][2], self.in_channels).permute(0, 4, 1, 2, 3).contiguous()

        # Inverse convolutions
        x4 = self.relu(self.inv_conv2(x4))
        x8 = self.relu(self.inv_conv3(x8))
        x2 = self.bn_inv_conv1(x2)
        x4 = self.bn_inv_conv2(x4)
        x8 = self.bn_inv_conv3(x8)
        out = torch.cat([x2, x4, x8], dim=1) 
        #out = self.convf(out)

        out = self.res_unitf(out)

        return out
