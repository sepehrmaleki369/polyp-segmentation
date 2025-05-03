import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        base = resnet34(weights=None)
        self.layer0 = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.pool0 = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x):
        x0 = self.layer0(x)        # 64
        x1 = self.layer1(self.pool0(x0))  # 64
        x2 = self.layer2(x1)       # 128
        x3 = self.layer3(x2)       # 256
        x4 = self.layer4(x3)       # 512
        return x0, x1, x2, x3, x4


class NestedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NestedConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        return self.conv(torch.cat(inputs, dim=1))


class UNetPP(nn.Module):
    def __init__(self, num_classes=1, deep_supervision=False):
        super(UNetPP, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [64, 64, 128, 256, 512]

        self.encoder = ResNetEncoder()

        self.conv0_1 = NestedConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = NestedConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = NestedConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = NestedConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = NestedConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = NestedConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = NestedConvBlock(filters[2]*2 + filters[3], filters[2])

        self.conv0_3 = NestedConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = NestedConvBlock(filters[1]*3 + filters[2], filters[1])

        self.conv0_4 = NestedConvBlock(filters[0]*4 + filters[1], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:]
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.encoder(x)

        x0_1 = self.conv0_1([x0_0, F.interpolate(x1_0, size=x0_0.shape[2:], mode='bilinear', align_corners=True)])
        x1_1 = self.conv1_1([x1_0, F.interpolate(x2_0, size=x1_0.shape[2:], mode='bilinear', align_corners=True)])
        x0_2 = self.conv0_2([x0_0, x0_1, F.interpolate(x1_1, size=x0_0.shape[2:], mode='bilinear', align_corners=True)])

        x2_1 = self.conv2_1([x2_0, F.interpolate(x3_0, size=x2_0.shape[2:], mode='bilinear', align_corners=True)])
        x1_2 = self.conv1_2([x1_0, x1_1, F.interpolate(x2_1, size=x1_0.shape[2:], mode='bilinear', align_corners=True)])
        x0_3 = self.conv0_3([x0_0, x0_1, x0_2, F.interpolate(x1_2, size=x0_0.shape[2:], mode='bilinear', align_corners=True)])

        x3_1 = self.conv3_1([x3_0, F.interpolate(x4_0, size=x3_0.shape[2:], mode='bilinear', align_corners=True)])
        x2_2 = self.conv2_2([x2_0, x2_1, F.interpolate(x3_1, size=x2_0.shape[2:], mode='bilinear', align_corners=True)])
        x1_3 = self.conv1_3([x1_0, x1_1, x1_2, F.interpolate(x2_2, size=x1_0.shape[2:], mode='bilinear', align_corners=True)])
        x0_4 = self.conv0_4([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, size=x0_0.shape[2:], mode='bilinear', align_corners=True)])

        if self.deep_supervision:
            out1 = F.interpolate(torch.sigmoid(self.final1(x0_1)), size=input_size, mode='bilinear', align_corners=True)
            out2 = F.interpolate(torch.sigmoid(self.final2(x0_2)), size=input_size, mode='bilinear', align_corners=True)
            out3 = F.interpolate(torch.sigmoid(self.final3(x0_3)), size=input_size, mode='bilinear', align_corners=True)
            out4 = F.interpolate(torch.sigmoid(self.final4(x0_4)), size=input_size, mode='bilinear', align_corners=True)
            return [out1, out2, out3, out4]
        else:
            out = self.final(x0_4)
            out = torch.sigmoid(out)
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
            return out
