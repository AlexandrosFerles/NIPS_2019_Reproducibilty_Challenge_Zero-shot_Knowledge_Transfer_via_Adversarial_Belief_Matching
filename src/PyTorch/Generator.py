# based on documentation of page https://github.com/polo5/ZeroShotKnowledgeTransfer/blob/master/models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.noise_to_linear = nn.Linear(100, 128*64)

        self.batch_norm1 = nn.BatchNorm2d(128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.batch_norm2 = nn.BatchNorm2d(128)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(3)

    def forward(self, z):

        x = self.noise_to_linear(z).view(-1, 128, 64)
        x = self.batch_norm1(x)
        x = self.up1(x)
        x = self.conv1(x)

        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.up2(x)
        x = self.conv2(x)

        x = self.batch_norm3(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batch_norm4(x)

        return x