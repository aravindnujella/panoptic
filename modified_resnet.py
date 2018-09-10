import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# split conv across ignore filters and copy filters
# to free and freeze them separately


class split_conv(nn.Module):

    def __init__(self, cur_in, cur_out, d_out, kernel_size, stride=1, padding=0, bias=False):
        super(split_conv, self).__init__()
        self.ignore_conv = nn.Conv2d(cur_in, cur_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.copy_conv = nn.Conv2d(cur_in, d_out, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        ignore = self.ignore_conv(x)
        copy = self.copy_conv(x)
        return torch.cat([ignore, copy], 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, d_out, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = split_conv(inplanes, planes, d_out, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes + d_out)
        self.conv2 = split_conv(planes, planes, d_out, kernel_size=3, stride=stride,
                                padding=1)
        self.bn2 = nn.BatchNorm2d(planes + d_out)
        self.conv3 = split_conv(planes, planes * self.expansion, d_out * self.expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion + d_out * self.expansion)
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


class ModNet(nn.Module):

    def __init__(self, block, layers, d_in):
        self.inplanes = 64 + d_in
        super(ModNet, self).__init__()
        self.conv1 = nn.Conv2d(3 + 1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 1 * 64, 1 * d_in, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 2 * 64, 2 * d_in, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 4 * 64, 4 * d_in, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 8 * 64, 8 * d_in, layers[4], stride=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, d_out, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                split_conv(self.inplanes, planes * block.expansion, d_out * block.expansion,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion + d_out * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, d_out, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, d_out))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# modify resnet to include additional channels
def modify_resnet():
    layers = [0, 3, 4, 23, 3]
    resnet = models.resnet101(pretrained=True)
    modnet = ModNet(Bottleneck, layers, d_in=1)

    modnet.conv1 = modify_conv(resnet.conv1)
    modnet.bn1 = modify_bn(resnet.bn1)

    modnet.layer1 = modify_layer(resnet.layer1, layers[1], 1 * d_in)
    modnet.layer2 = modify_layer(resnet.layer2, layers[2], 2 * d_in)
    modnet.layer3 = modify_layer(resnet.layer3, layers[3], 4 * _din)
    modnet.layer4 = modify_layer(resnet.layer4, layers[4], 8 * d_in)

    return modnet


def modify_conv(conv, d_in, d_out):
    weight = conv.weight
    cur_out = weight.shape[0]
    cur_in = weight.shape[1]
    kernel_shape = weight.shape[2:]

    # ignore_filters: cur_out, cur_in + d_in, kernel_shape
    c = torch.zeros((cur_out, d_in) + kernel_shape)
    ig_filter_weight = torch.cat([weight, c], 1)

    # copy_filters: d_out, cur_in + d_in, kernel_shape
    fan_in = kernel_shape[0] * kernel_shape[1]
    a = torch.zeros((d_out, cur_in,) + kernel_shape)
    b = torch.eye(d_out, d_in).unsqueeze(-1).unsqueeze(-1)
    b = b.repeat([1, 1, kernel_shape[0], kernel_shape[1]]) / fan_in
    cp_filter_weight = torch.cat([a, b], 1)

    return ig_filter_weight, cp_filter_weight


def modify_bn(bn, d_in, d_out):

    return 0


def modify_bottleneck(bottleneck, d_in):
    modify_conv(block.conv1)
    modify_bn(block.bn1)
    modify_conv(block.conv2)
    modify_bn(block.bn2)
    modify_conv(block.conv3)
    modify_bn(block.bn3)

    if block.downsample != None:
        modify_conv(block.downsample)

    return stuff


def modify_layer(layer, count):
    for i in range(block_count):
        modify_bottleneck(layer[i])
    return stuff
