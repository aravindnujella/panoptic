import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

import numpy as np


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class iconv(nn.Module):

    def __init__(self, new_in, old_out, d_out, *args, **kwargs):
        super(iconv, self).__init__()
        self.ignore_conv = nn.Conv2d(new_in, old_out, *args, **kwargs)
        self.copy_conv = nn.Conv2d(new_in, d_out, *args, **kwargs)

    def forward(self, x):
        ig = self.ignore_conv(x)
        cp = self.copy_conv(x)
        return torch.cat([ig, cp], 1)


class ibn(nn.Module):

    def __init__(self, old_in, d_in, *args, **kwargs):
        super(ibn, self).__init__()
        self.ignore_bn = nn.BatchNorm2d(old_in, *args, **kwargs)
        self.copy_bn = nn.BatchNorm2d(d_in, *args, **kwargs)
        self.old_in = old_in
        self.d_in = d_in

    def forward(self, x):
        ig, cp = x[:, :self.old_in, :, :], x[:, self.old_in:, :, :]
        ig, cp = self.ignore_bn(ig), self.copy_bn(cp)
        return torch.cat([ig, cp], 1)


class iBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, d, stride=1, downsample=None):
        super(iBottleneck, self).__init__()
        p, q, r, s = d
        self.conv1 = iconv(inplanes + p, planes, q, kernel_size=1)
        self.bn1 = ibn(planes, q)
        self.conv2 = iconv(planes + q, planes, r, kernel_size=3, stride=stride, padding=1)
        self.bn2 = ibn(planes, r)
        self.conv3 = iconv(planes + r, planes * self.expansion, s, kernel_size=1)
        self.bn3 = ibn(planes * self.expansion, s)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class iResNet(nn.Module):

    def __init__(self, block, layers, d=4):
        self.inplanes = 64
        super(iResNet, self).__init__()
        self.conv1 = iconv(3 + 1, 64, d, kernel_size=7, stride=2, padding=3,
                           bias=False)
        self.bn1 = ibn(64, d)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 1 * 64, 1 * d, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 2 * 64, 2 * d, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 4 * 64, 4 * d, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 8 * 64, 8 * d, layers[4], stride=2)

    def _make_layer(self, block, planes, d, block_count, stride):

        ds = [d, d // 2, d // 2, 2 * d]

        downsample = nn.Sequential(
            iconv(self.inplanes + ds[0], planes * block.expansion, ds[-1], kernel_size=1, stride=stride),
            ibn(planes * block.expansion, ds[-1]),
        )

        layers = []
        layers.append(block(self.inplanes, planes, ds, stride, downsample))
        self.inplanes = planes * block.expansion
        ds[0] = ds[-1]
        for i in range(1, block_count):
            layers.append(block(self.inplanes, planes, ds))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == '__main__':

    from PIL import Image
    img = Image.open("./data/0.png").convert('RGB')
    img = np.array(img, np.float32)

    mu = np.array(
        [0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, -1)
    sig = np.array(
        [0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, -1)

    img /= 255
    img -= mu
    img /= sig

    img = np.moveaxis(img, 2, 0)
    img = np.expand_dims(img, 0)

    img = torch.Tensor(img)

    impulse = torch.zeros(img.shape[-2:]).unsqueeze(0).unsqueeze(0)

    inp = torch.cat([img, impulse], 1)
    inp = inp.cuda()

    iresnet = iResNet(iBottleneck, [0, 3, 4, 23, 3], 4)
    iresnet = iresnet.cuda()

    print(iresnet(inp).shape)
