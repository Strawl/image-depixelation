import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batch_normalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: nn.Module = nn.ReLU()
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv2d(
            input_channels,
            hidden_channels,
            kernel_size,
            padding="same",
            padding_mode="zeros",
            bias=not use_batch_normalization
        ))
        # We already added one conv layer, so start the range from 1 instead of 0
        for i in range(1, num_hidden_layers):
            self.conv_layers.append(nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding="same",
                padding_mode="zeros",
                bias=not use_batch_normalization
            ))
        if self.use_batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(num_hidden_layers):
                self.batch_norm_layers.append(nn.BatchNorm2d(hidden_channels))
        self.output_layer = nn.Linear(self.hidden_channels * 64 * 64, self.num_classes)
    
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_hidden_layers):
            input_images = self.conv_layers[i](input_images)
            if self.use_batch_normalization:
                input_images = self.batch_norm_layers[i](input_images)
            input_images = self.activation_function(input_images)
        input_images = input_images.view(-1, self.hidden_channels * 64 * 64)
        input_images = self.output_layer(input_images)
        return input_images


if __name__ == "__main__":
    torch.random.manual_seed(0)
    network = SimpleCNN(3, 32, 3, True, 10, activation_function=nn.ELU())
    input = torch.randn(1, 3, 64, 64)
    output = network(input)
    print(output)
