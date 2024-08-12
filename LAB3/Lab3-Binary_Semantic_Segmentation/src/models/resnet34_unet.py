# import torch
# import torch.nn as nn
# import torchvision
# import numpy as np


# def Conv1(in_planes, places, stride=2):
#     return nn.Sequential(
#         nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
#         nn.BatchNorm2d(places),
#         nn.ReLU(inplace=True),
#         nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#     )

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_places, places, stride=1, downsampling=False):
#         super(BasicBlock, self).__init__()
#         self.downsampling = downsampling

#         self.basic_block = nn.Sequential(
#             nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(places),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(places * self.expansion),
#         )

#         if self.downsampling:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(places * self.expansion),
#             )
        
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         residual = x
#         out = self.basic_block(x)

#         if self.downsampling:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)
#         return out

# class ResNet(nn.Module):
#     def __init__(self,blocks, num_classes=1, block=BasicBlock):
#         super(ResNet,self).__init__()
#         self.expansion = block.expansion

#         self.conv1 = Conv1(in_planes = 3, places= 64)

#         self.layer1 = self.make_layer(block, 64, 64, blocks[0], stride=1)
#         self.layer2 = self.make_layer(block, 64 * self.expansion, 128, blocks[1], stride=2)
#         self.layer3 = self.make_layer(block, 128 * self.expansion, 256, blocks[2], stride=2)
#         self.layer4 = self.make_layer(block, 256 * self.expansion, 512, blocks[3], stride=2)

#         self.avgpool = nn.AvgPool2d(7, stride=1)
#         self.fc = nn.Linear(512 * self.expansion, num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def make_layer(self, block, in_places, places, block_count, stride=1):
#         layers = []
#         layers.append(block(in_places, places, stride, downsampling=(stride != 1)))
#         for _ in range(1, block_count):
#             layers.append(block(places * self.expansion, places))

#         return nn.Sequential(*layers)


#     def forward(self, x):
#         x = self.conv1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         # x = self.avgpool(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc(x)
#         return x

# def double_convolution(in_channels, out_channels):
#     conv_op = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         nn.ReLU(inplace=True)
#     )
#     return conv_op

# class UNet(nn.Module):
#     def __init__(self, num_classes):
#         super(UNet, self).__init__()
#         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Expanding path.
#         self.up_transpose_1 = nn.ConvTranspose2d(
#             in_channels=512, out_channels=256, kernel_size=2, stride=2)
#         self.up_convolution_1 = double_convolution(256, 256)
#         self.up_transpose_2 = nn.ConvTranspose2d(
#             in_channels=256, out_channels=128, kernel_size=2, stride=2)
#         self.up_convolution_2 = double_convolution(128, 128)
#         self.up_transpose_3 = nn.ConvTranspose2d(
#             in_channels=128, out_channels=64, kernel_size=2, stride=2)
#         self.up_convolution_3 = double_convolution(64, 64)
#         self.up_transpose_4 = nn.ConvTranspose2d(
#             in_channels=64, out_channels=64, kernel_size=2, stride=2)
#         self.up_convolution_4 = double_convolution(64, 64)
#         # output => `out_channels` as per the number of classes.
#         self.out = nn.Conv2d(
#             in_channels=64, out_channels=num_classes, kernel_size=1
#         )
#     def forward(self, x):
#         # x = x.float() / 255.0
#         up_1 = self.up_transpose_1(x)
#         x = self.up_convolution_1(up_1)
#         up_2 = self.up_transpose_2(x)
#         x = self.up_convolution_2(up_2)
#         up_3 = self.up_transpose_3(x)
#         x = self.up_convolution_3(up_3)
#         up_4 = self.up_transpose_4(x)
#         x = self.up_convolution_4(up_4)
#         out = self.out(x)
#         return out

# class ResNet34Unet(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet34Unet, self).__init__()
#         self.encoder = ResNet(blocks=[3, 4, 6, 3])
#         self.decoder = UNet(num_classes)

#     def forward(self, x):
#         x = x.float() / 255.0
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
    
# # model = ResNet34Unet(num_classes=1)
# # input_tensor = torch.randn(1, 3, 512, 512)  # Example input tensor
# # output = model(input_tensor)
# # print(output.shape)
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(BasicBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1,
                      bias=False),  # need to change the stride
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1,
                      bias=False),  # retain the size of feature map
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # compute residual
        out += residual
        return nn.ReLU(inplace=True)(out)


class Conv2dReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
    ):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super(SegmentationHead, self).__init__(conv2d, upsampling)


class ResNet34Unet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet34Unet, self).__init__()
        inchannels = 3
        self.pre = nn.Sequential(
            nn.Conv2d(inchannels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.body = self.makelayers([3, 4, 6, 3])
        in_channels = [512, 256, 128, 128, 32]
        skip_channels = [256, 128, 64, 0, 0]
        out_channels = [256, 128, 64, 32, 16]
        blocks = [
            DecoderBlock(in_ch, skip_ch,
                         out_ch) for in_ch, skip_ch, out_ch in zip(
                             in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.seg = SegmentationHead(16, num_classes)

    def makelayers(self, blocklist):
        self.layers = []
        for index, blocknum in enumerate(blocklist):
            if index != 0:
                shortcut = nn.Sequential(
                    nn.Conv2d(64 * 2**(index - 1),
                              64 * 2**index,
                              1,
                              2,
                              bias=False),
                    nn.BatchNorm2d(64 * 2**index))  # make input and output consistent
                self.layers.append(
                    BasicBlock(64 * 2**(index - 1), 64 * 2**index, 2,
                                  shortcut))
            for i in range(0 if index == 0 else 1, blocknum):
                self.layers.append(
                    BasicBlock(64 * 2**index, 64 * 2**index, 1))
        return nn.Sequential(*self.layers)
    
    def forward(self, x):
        # make sure the input x is float
        x = x.float()

        self.features = []
        for i, l in enumerate(self.pre):
            x = l(x)
            if i == 2:
                self.features.append(x)

        for i, l in enumerate(self.body):
            if i == 3 or i == 7 or i == 13:
                self.features.append(x)
            x = l(x)
        skips = self.features[::-1]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        x = self.seg(x)
        return x