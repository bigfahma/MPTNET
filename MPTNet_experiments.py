from models.Decoder import Decoder
from models.Encoder import Encoder
from models.PatchEmbedding import PatchEmbed
import torch.nn as nn
import torch
from models.utils import to_3tuple
import numpy as np
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
import matplotlib.pyplot as plt
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch


import numpy as np
import matplotlib.pyplot as plt

def plot_class_overlays(tensor):
    """
    Plots the classes overlaid, highlighting voxels belonging to more than one class.
    
    Parameters:
    - tensor: A torch.Tensor of shape [B, C, H, W, D].
    """
    # Convert tensor to numpy for easy manipulation and plotting.
    tensor_np = tensor.numpy()
    
    # Assuming the first batch for simplicity.
    tensor_np = tensor_np[0]  # Shape: [C, H, W, D].
    
    # Select a slice for visualization (middle slice in D dimension for example).
    middle_slice = tensor_np[:, :, :, tensor_np.shape[-1] // 2]
    
    # Initialize plot.
    plt.figure(figsize=(10, 10))
    
    # Check for voxels that belong to multiple classes.
    multiple_classes = np.sum(middle_slice, axis=0) > 1
    
    # Plot each class with a unique color.
    for c in range(middle_slice.shape[0]):
        plt.contourf(middle_slice[c, :, :], levels=[0.5, 1], alpha=0.5)
    
    # Overlay voxels belonging to multiple classes in red.
    plt.contourf(multiple_classes, levels=[0.5, 1], colors=['red'], alpha=0.5)
    
    plt.title("Class Overlays with Conflicts Highlighted")
    plt.axis('off')
    plt.show()


def verify_mutual_exclusivity(tensor):
    """
    Verifies that each voxel belongs to one and only one class in the output tensor.
    
    Parameters:
    - tensor: A torch.Tensor of shape [B, C, H, W, D].
    
    Returns:
    - bool indicating whether the tensor satisfies the mutual exclusivity condition.
    """
    # Assuming the class dimension is along C, we sum along C.
    # If mutual exclusivity holds, the sum should be 1 for each voxel.
    sum_along_classes = tensor.sum(dim=1)
    
    # Check if each voxel's sum is equal to 1.
    is_exclusive = torch.all(sum_along_classes == 1)
    
    return is_exclusive.item()  # Convert to Python bool for readability.

class MPTNet(nn.Module):
    def __init__(self, emb_dim, resolution, num_classes, depth, num_heads, window_size, n_layers = 3, pooling_size = [(1,2,2), (2,4,4), (2,8,8)], emb_ratio = [1, 4, 4]):
        super().__init__()
        
        H, W, D = resolution
        downD, downH, downW = emb_ratio
        if downD == 1 : #CONFIG 2 ACDC
            out_proj1 = (4,2,2)
            out_proj2 = (4,2,2)
        elif downD == 2 :
            out_proj1 = (4,2,2) # CONFIG 1 SYNAPSE
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
        x = x.permute(0, 1, 4, 3, 2)
        #print("Input :",x.shape)
        x = self.patch_embedding(x)
        #print("After PE",x.shape)
        x, skips = self.mrha_encoder(x)
        #print("Output Encoder :",x.shape)
        x = self.mrha_decoder(x, skips)
        x = x.permute(0, 1, 4, 3, 2)
        return x
    
if __name__ == "__main__":
    # Example usage
    B, C, H, W, D = 1, 1, 128, 128, 32
                    # ACDC : (1, 4, 4), (128, 128, 8), Pooling : [1,2,2], [1,4,4], [1,8,8]
                    # SYNAPSE : (2, 4, 4), (256, 256, 32), Pooling : [1,2,2] [2,4,4]  [2,8,8]
    embed_dim = 96
    num_heads = [4, 8, 16, 32]
    window_size = (7,7,7)
    num_class = 9
    model = MPTNet(emb_dim=embed_dim, resolution=(H, W, D), num_classes= num_class, depth = 2, 
                   num_heads=num_heads, window_size=window_size, 
                   n_layers= 3, pooling_size = [(1,2,2), (1,4,4), (1,8,8)], emb_ratio = [1, 4, 4])
    input_tensor = torch.randn(B, C, H, W, D)
    output = model(input_tensor)
    print(np.unique(output.detach().numpy()))
    print(output.shape)  

    
    # model_unet = UNet(
    # spatial_dims=3,
    # in_channels=1,
    # out_channels=num_class,
    # channels=(16, 32, 64, 128, 256),
    # strides=(2, 2, 2, 2),
    # num_res_units=2,
    # norm=Norm.BATCH)

    # output = model_unet(input_tensor)
    # print(np.unique(output.detach().numpy()))
    # print(output.shape)  

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=num_class)])
    out = [post_pred(i) for i in decollate_batch(output)]
    out = torch.stack(out)


    # post_trans_images = Compose(
    #         [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    #     ) 
    # out = post_trans_images(output)
    
    exclusivity = verify_mutual_exclusivity(out)
    print("Exclusivity : ", exclusivity)
    #plot_class_overlays(out)
    print(out.shape)
    print(np.unique(out.detach().numpy()))

