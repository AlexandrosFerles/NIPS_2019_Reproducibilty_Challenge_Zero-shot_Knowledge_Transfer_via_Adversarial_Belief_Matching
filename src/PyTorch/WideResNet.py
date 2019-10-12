import torch
import torch.nn as nn
from torchsummary import summary
import math


class IndividualBlock1(nn.Module):

    def __init__(self, input_features, output_features, stride, subsample_input=True, increase_filters=True):
        super(IndividualBlock1, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.batch_norm1 = nn.BatchNorm2d(input_features)
        self.batch_norm2 = nn.BatchNorm2d(output_features)

        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1, bias=False)

        self.subsample_input = subsample_input
        self.increase_filters = increase_filters
        if subsample_input:
            self.conv_inp = nn.Conv2d(input_features, output_features, kernel_size=1, stride=2, padding=0, bias=False)
        elif increase_filters:
            self.conv_inp = nn.Conv2d(input_features, output_features, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):

        if self.subsample_input or self.increase_filters:
            x = self.batch_norm1(x)
            x = self.activation(x)
            x1 = self.conv1(x)
        else:
            x1 = self.batch_norm1(x)
            x1 = self.activation(x1)
            x1 = self.conv1(x1)
        x1 = self.batch_norm2(x1)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)

        if self.subsample_input or self.increase_filters:
            return self.conv_inp(x) + x1
        else:
            return x + x1


class IndividualBlockN(nn.Module):

    def __init__(self, input_features, output_features, stride):
        super(IndividualBlockN, self).__init__()

        self.activation = nn.ReLU(inplace=True)

        self.batch_norm1 = nn.BatchNorm2d(input_features)
        self.batch_norm2 = nn.BatchNorm2d(output_features)

        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(output_features, output_features, kernel_size=3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        x1 = self.batch_norm1(x)
        x1 = self.activation(x1)
        x1 = self.conv1(x1)
        x1 = self.batch_norm2(x1)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)

        return x1 + x


class Nblock(nn.Module):

    def __init__(self, N, input_features, output_features, stride, subsample_input=True, increase_filters=True):
        super(Nblock, self).__init__()

        layers = []
        for i in range(N):
            if i == 0:
                layers.append(IndividualBlock1(input_features, output_features, stride, subsample_input, increase_filters))
            else:
                layers.append(IndividualBlockN(output_features, output_features, stride=1))

        self.nblockLayer = nn.Sequential(*layers)

    def forward(self, x):
        return self.nblockLayer(x)


class WideResNet(nn.Module):

    def __init__(self, d, k, n_classes, input_features, output_features, strides):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_features, output_features, kernel_size=3, stride=strides[0], padding=1, bias=False)

        filters = [16 * k, 32 * k, 64 * k]
        self.out_filters = filters[-1]
        N = (d - 4) // 6
        increase_filters = k > 1
        self.block1 = Nblock(N, input_features=output_features, output_features=filters[0], stride=strides[1], subsample_input=False, increase_filters=increase_filters)
        self.block2 = Nblock(N, input_features=filters[0], output_features=filters[1], stride=strides[2])
        self.block3 = Nblock(N, input_features=filters[1], output_features=filters[2], stride=strides[3])

        self.batch_norm = nn.BatchNorm2d(filters[-1])
        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=8)
        self.fc = nn.Linear(filters[-1], n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv1(x)
        attention1 = self.block1(x)
        attention2 = self.block2(attention1)
        attention3 = self.block3(attention2)
        out = self.batch_norm(attention3)
        out = self.activation(out)
        out = self.avg_pool(out)
        out = out.view(-1, self.out_filters)

        return self.fc(out), attention1, attention2, attention3


if __name__=='__main__':

    # change d and k if you want to check a model other than WRN-40-2
    d = 40
    k = 2
    strides = [1, 1, 2, 2]
    net = WideResNet(d=d, k=k, n_classes=10, input_features=3, output_features=16, strides=strides)

    # verify that an output is produced
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    net(sample_input)

    # Summarize model
    summary(net, input_size=(3, 32, 32))