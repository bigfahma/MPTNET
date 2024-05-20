import torch
import torch.nn as nn
from models.MultiPoolingTr import MultiPoolingTransformer  # Assuming MultiPoolingTransformer is in a separate file

class Encoder(nn.Module):
    def __init__(self, resolution, emb_dim, num_heads, window_size, n_layers = 3, pooling_size = [(1,2,2), (1,4,4), (1,8,8)]):
        super(Encoder, self).__init__()
        
        self.mp_transformers = nn.ModuleList()
        current_channels = emb_dim
        D, H, W = resolution

        for _ in range(n_layers):
            mp_transformer = MultiPoolingTransformer(
                resolution=(D, H, W), 
                in_channels=current_channels, 
                num_heads=num_heads[_], 
                window_size=window_size,
                pooling_size= pooling_size
            )
            self.mp_transformers.append(mp_transformer)
            current_channels *= 2
            D, H, W = D // 2, H // 2, W // 2

    def forward(self, x):
        skips = [x.flatten(2).transpose(1, 2).contiguous()]
        for mp_transformer in self.mp_transformers:
            x = mp_transformer(x)
            skips.append(x.flatten(2).transpose(1, 2).contiguous())
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x, skips[:-1]


if __name__ == "__main__":
    # Example usage
    B, C, H, W, D = 1, 8, 64, 64, 64
    n_heads = [4, 8, 16, 32]
    window_size = 8
    encoder = Encoder(resolution=(D, H, W), emb_dim = C, num_heads=n_heads, window_size=window_size, pooling_size = [(1,2,2), (2,4,4), (2,8,8)])
    input_tensor = torch.randn(B, C, D, H, W)
    output, skips = encoder(input_tensor)
    print(output.shape) 
