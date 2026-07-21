from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, stride=1):
        super(ResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
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
    def __init__(self, in_channels: int, out_channels: int | None = None, stride=1, inner_channels = None,
                 activation = nn.ReLU(inplace=True)):
        super(Bottleneck, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if inner_channels is None:
            inner_channels = out_channels // 4
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(inner_channels),
            activation
        )
        self.process_block = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            activation
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
        self.last_relu = activation

    def forward(self, x):
        out = self.proj_in(x)
        out = self.process_block(out)
        out = self.proj_out(out)
        x = self.shortcut(x)
        return self.last_relu(out + x)

class DarknetResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None, inner_channels: int | None = None,
                 activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)):
        super(DarknetResidualBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if inner_channels is None:
            inner_channels = out_channels // 2
        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            activation
        )
        self.proj_out = nn.Sequential(
            nn.Conv2d(inner_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        out = self.proj_in(x)
        out = self.proj_out(out)
        return out + x
