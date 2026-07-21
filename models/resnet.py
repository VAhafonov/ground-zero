import torch.nn as nn
from layers.resblock import ResidualBlock, Bottleneck

"""
ResNet implemented in PyTorch. Based on:
https://arxiv.org/pdf/1512.03385
"""

class Resnet(nn.Module):
    def __init__(self, version: str, num_classes=10):
        super(Resnet, self).__init__()
        self.supported_versions = {'18', '34', '50', '101', '152'}
        if version not in self.supported_versions:
            raise AttributeError("Unsupported ResNet version: {}".format(version))
        self.name = f"resnet{version}"
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block_type = ResidualBlock if version in {'18', '34'} else Bottleneck
        num_channels = [64, 128, 256, 512] if version in {'18', '34'} else [256, 512, 1024, 2048]
        self.num_blocks_mapping = {
            '18': [2, 2, 2, 2],
            '34': [3, 4, 6, 3],
            '50': [3, 4, 6, 3],
            '101': [3, 4, 23, 3],
            '152': [3, 8, 36, 3],
        }

        # second_conv
        conv_idx = 0
        n_channels_before = 64
        n_channels = num_channels[conv_idx]
        self.conv2 = self.create_block(block_type, n_channels_before, n_channels,
                                       n_blocks=self.num_blocks_mapping[version][conv_idx], stride=1)

        # third_conv
        conv_idx = 1
        n_channels_before = n_channels
        n_channels = num_channels[conv_idx]
        self.conv3 = self.create_block(block_type, n_channels_before, n_channels,
                                       n_blocks=self.num_blocks_mapping[version][conv_idx])

        # fourth_conv
        conv_idx = 2
        n_channels_before = n_channels
        n_channels = num_channels[conv_idx]
        self.conv4 = self.create_block(block_type, n_channels_before, n_channels,
                                       n_blocks=self.num_blocks_mapping[version][conv_idx])

        # fifth_conv
        conv_idx = 3
        n_channels_before = n_channels
        n_channels = num_channels[conv_idx]
        self.conv5 = self.create_block(block_type, n_channels_before, n_channels,
                                       n_blocks=self.num_blocks_mapping[version][conv_idx])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(n_channels, num_classes)

    @staticmethod
    def create_block(block_type: nn.Module, n_channels_before: int, n_channels: int, n_blocks: int, stride: int = 2) -> nn.Sequential:
        block = [block_type(in_channels=n_channels_before, out_channels=n_channels, stride=stride)]
        for i in range(1, n_blocks):
            block.append(block_type(in_channels=n_channels))
        return nn.Sequential(*block)

    def forward(self, x):
        if isinstance(x, dict):
            x = x['input']
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.head(x)
        return x
