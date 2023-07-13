import torch
from torch import nn

class ImageDepixelationModel(nn.Module):
    def __init__(self, layers_config: list):
        super(ImageDepixelationModel, self).__init__()

        layers = []
        for config in layers_config:
            layers.append(nn.Conv2d(config['in_channels'], config['out_channels'], config['kernel_size'], padding=config['kernel_size']//2))
            if config.get('batchnorm', False):
                layers.append(nn.BatchNorm2d(config['out_channels']))
            layers.append(config.get('activation', nn.ReLU())())
        
        self.layers = nn.Sequential(*layers)

    def forward(self, input_images: torch.Tensor):
        return self.layers(input_images)

