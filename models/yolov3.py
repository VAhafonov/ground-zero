import torch
from torch import nn
from torch.nn.functional import upsample

from layers.resblock import DarknetResidualBlock


"""
Yolo paper: https://arxiv.org/pdf/1506.02640
Yolov2 paper: https://arxiv.org/pdf/1612.08242
Yolov3 paper: https://arxiv.org/pdf/1804.02767
"""


def create_activation(leaky_coef: float = 0.1) -> nn.Module:
    return nn.LeakyReLU(negative_slope=leaky_coef, inplace=True)


class YOLOv3(nn.Module):
    def __init__(self, n_classes=80, leaky_coef=0.1, anchors_num=3, upsample_mode: str = 'nearest'):
        super(YOLOv3, self).__init__()
        self.upsample_mode = upsample_mode
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            create_activation(leaky_coef),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            create_activation(leaky_coef),
            DarknetResidualBlock(in_channels=64, activation=create_activation(leaky_coef)),
        )
        self.stage_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            create_activation(leaky_coef),
            DarknetResidualBlock(in_channels=128, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=128, activation=create_activation(leaky_coef)),
        )
        self.stage_2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=256, activation=create_activation(leaky_coef)),
        )
        self.stage_3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=512, activation=create_activation(leaky_coef)),
        )
        self.stage_4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            create_activation(leaky_coef),
            DarknetResidualBlock(in_channels=1024, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=1024, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=1024, activation=create_activation(leaky_coef)),
            DarknetResidualBlock(in_channels=1024, activation=create_activation(leaky_coef)),
        )
        last_c_num = anchors_num * (n_classes + 4 + 1)

        self.first_route = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            create_activation(leaky_coef),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            create_activation(leaky_coef),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef)
        )
        self.first_dt_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            create_activation(leaky_coef),
            nn.Conv2d(1024, last_c_num, kernel_size=1, stride=1, bias=True)
        )

        self.second_reduction = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
        )
        self.second_route = nn.Sequential(
            nn.Conv2d(256 + 512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
        )
        self.second_dt_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            create_activation(leaky_coef),
            nn.Conv2d(512, last_c_num, kernel_size=1, stride=1, bias=True),
        )

        self.third_reduction = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            create_activation(leaky_coef),
        )
        self.third_dt_head = nn.Sequential(
            nn.Conv2d(128 + 256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            create_activation(leaky_coef),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            create_activation(leaky_coef),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
            create_activation(leaky_coef),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            create_activation(leaky_coef),
            nn.Conv2d(256, last_c_num, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        x = self.conv1(x) # 1/1
        x = self.conv2(x) # 1/2
        x = self.stage_1(x) # 1/4
        x_2 = self.stage_2(x) # 1/8
        x_3 = self.stage_3(x_2) # 1/16
        x_4 = self.stage_4(x_3) # 1/32

        # first scale
        features_1 = self.first_route(x_4)
        first_d = self.first_dt_head(features_1) # 1/32

        # second scale
        features_2 = self.second_reduction(features_1) # 1/32
        features_2 = upsample(features_2, scale_factor=2, mode=self.upsample_mode) # 1/16
        features_2 = torch.cat([features_2, x_3], dim=1) # 1/16
        features_2 = self.second_route(features_2) # 1/16
        second_d = self.second_dt_head(features_2) # 1/16

        # third scale
        features_3 = self.third_reduction(features_2) # 1/16
        features_3 = upsample(features_3, scale_factor=2, mode=self.upsample_mode) # 1/8
        features_3 = torch.cat([features_3, x_2], dim=1) # 1/8
        third_d = self.third_dt_head(features_3)


        return {
            'first_d': first_d,
            'second_d': second_d,
            'third_d': third_d,
        }

if __name__ == "__main__":
    rand_tensor = torch.randn(1, 3, 256, 256)
    net = YOLOv3()
    out = net(rand_tensor)
    print()