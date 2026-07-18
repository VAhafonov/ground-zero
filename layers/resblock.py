from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        if in_channels != out_channels or stride != 1:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.proj = nn.Identity()
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        x = self.proj(x)
        return self.last_relu(out + x)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.process_block = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.last_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.proj_in(x)
        out = self.process_block(out)
        out = self.proj_out(out)
        x = self.shortcut(x)
        return self.last_relu(out + x)