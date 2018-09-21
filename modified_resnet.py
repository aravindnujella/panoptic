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

    def __init__(self, cur_in, cur_out, d_out, *args, **kwargs):
        super(split_conv, self).__init__()
        self.ignore_conv = nn.Conv2d(cur_in, cur_out, *args, **kwargs)
        self.copy_conv = nn.Conv2d(cur_in, d_out, *args, **kwargs)

    def forward(self, x):
        ig = self.ignore_conv(x)
        cp = self.copy_conv(x)
        return torch.cat([ig, cp], 1)


class split_bn(nn.Module):

    def __init__(self, cur_in, d_in, *args, **kwargs):
        super(split_bn, self).__init__()
        self.ignore_bn = nn.BatchNorm2d(cur_in, *args, **kwargs)
        self.copy_bn = nn.BatchNorm2d(d_in, *args, **kwargs)
        self.cur_in = cur_in
        # self.d_in = d_in

    def forward(self, x):
        ig, cp = x[:, :self.cur_in, :, :], x[:, self.cur_in:, :, :]
        ig, cp = self.ignore_bn(ig), self.copy_bn(cp)
        return torch.cat([ig, cp], 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, d_out, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = split_conv(inplanes, planes, d_out, kernel_size=1)
        self.bn1 = split_bn(planes, d_out)
        self.conv2 = split_conv(planes, planes, d_out, kernel_size=3, stride=stride,
                                padding=1)
        self.bn2 = split_bn(planes, d_out)
        self.conv3 = split_conv(planes, planes * self.expansion, d_out * self.expansion, kernel_size=1)
        self.bn3 = split_bn(planes * self.expansion, d_out * self.expansion)
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
        self.conv1 = split_conv(3 + 1, 64, d_in, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = split_bn(self.inplanes, d_in)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 1 * 64, 1 * d_in, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 2 * 64, 2 * d_in, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 4 * 64, 4 * d_in, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 8 * 64, 8 * d_in, layers[4], stride=2)

    def _make_layer(self, block, planes, d_out, blocks, stride=1):
        # calling it downsample; technically it is not downsampling but a
        # linear resampling to adjust number of filter dimensions
        downsample = nn.Sequential(
            split_conv(self.inplanes, planes * block.expansion, d_out * block.expansion,
                       kernel_size=1, stride=stride),
            split_bn(planes * block.expansion, d_out * block.expansion),
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


# these functions are tailored towards modifying resnet weights only.
# so are not very generic

def init_conv(conv, d_in, d_out):
    weight = conv.weight
    old_out = weight.shape[0]
    old_in = weight.shape[1]
    kernel_size = weight.shape[2:]

    # ignore_filters: old_out, old_in + d_in, kernel_size
    c = torch.zeros((old_out, d_in) + kernel_size)
    ig_filter_weight = torch.cat([weight, c], 1)

    # copy_filters: d_out, old_in + d_in, kernel_size
    fan_in = kernel_size[0] * kernel_size[1]
    a = torch.zeros((d_out, old_in,) + kernel_size)
    b = torch.eye(d_out, d_in).unsqueeze(-1).unsqueeze(-1)
    b = b.repeat([1, 1, kernel_size[0], kernel_size[1]]) / fan_in
    cp_filter_weight = torch.cat([a, b], 1)

    mod_conv = split_conv(old_in + d_in, old_out, d_out, kernel_size=kernel_size, stride=conv.stride, padding=conv.padding)

    mod_conv.copy_filters.weight = cp_filter_weight
    mod_conv.ignore_filters.weight = ig_filter_weight
    return mod_conv


def init_bn(bn, d_in):

    old_in = bn.weight.shape[0]
    mod_bn = split_bn(old_in, d_in)
    
    mod_bn.ignore_bn.running_var = bn.running_var
    mod_bn.ignore_bn.running_mean = bn.running_mean
    mod_bn.ignore_bn.weight = bn.weight
    mod_bn.ignore_bn.bias = bn.bias

    mod_bn.copy_bn.running_var = nn.init.constant(1)
    mod_bn.copy_bn.running_mean = nn.init.constant(0)
    mod_bn.copy_bn.weight = nn.init.constant(1)
    mod_bn.copy_bn.bias = nn.init.constant(0)
    
    return mod_bn


def init_bottleneck(block, d_in):
    mod_block = Bottleneck()
    block.conv1 = init_conv(block.conv1, d_in)
    block.bn1 = init_bn(block.bn1, d_in)
    block.conv2 = init_conv(block.conv2, d_in)
    block.bn2 = init_bn(block.bn2, d_in)
    block.conv3 = init_conv(block.conv3, d_in)
    block.bn3 = init_bn(block.bn3, d_in)

    if block.downsample != None:
        block.downsample = init_conv(block.downsample, d_in)

    return block


def init_layer(layer, count):
    for i in range(block_count):
        init_bottleneck(layer[i])
    return stuff
