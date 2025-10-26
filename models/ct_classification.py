import torch
import torch.nn as nn
import torch.nn.functional as F

class Convnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, stride) -> None:
        super().__init__()
        convnd = nn.Conv3d if num_dims == 3 else nn.Conv2d
        normnd = nn.InstanceNorm3d if num_dims == 3 else nn.InstanceNorm2d
        self.conv = convnd(in_channels, out_channels, 3, stride, 1)
        self.norm = normnd(out_channels, affine=True)
        self.relu = nn.LeakyReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        return self.relu(self.norm(x))

class StackedConvnd(nn.Module):
    def __init__(self, num_dims, in_channels, out_channels, first_stride) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            Convnd(num_dims, in_channels, out_channels, first_stride),
            Convnd(num_dims, out_channels, out_channels, 1)
        )
    
    def forward(self, x):
        return self.blocks(x)


class UNet(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_dims = config.MODEL.NUM_DIMS
        assert self.num_dims in [2, 3], 'only 2d or 3d inputs are supported'
        self.num_classes = 6

        self.extra = config.MODEL.EXTRA
        self.enc_channels = self.extra.ENC_CHANNELS

        self.H, self.W, self.D = [64, 64, 64]
        
        # encoder
        self.enc = nn.ModuleList()
        prev_channels = 1
        for i, channels in enumerate(self.enc_channels):
            # we do not perform downsampling at first convolution layer
            first_stride = 2 if i != 0 else 1
            self.enc.append(StackedConvnd(self.num_dims, prev_channels, channels, first_stride))
            prev_channels = channels
        
        self.fc1 = nn.Linear(512 * (self.H//16) * (self.W//16) * (self.D//16), 512)
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):
        for layer in self.enc[:-1]:
            x = layer(x)
        x = self.enc[-1](x)
        x = x.view(-1, 512 * (self.H//16) * (self.W//16) * (self.D//16))
        x = F.leaky_relu(self.fc1(x), inplace=True)
        x = self.fc2(x)  
        return x
