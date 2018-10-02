import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

import numpy as np
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# split conv across ignore filters and copy filters
# to free and freeze them separately

# new_ = new resnet dimensions
# old_ = old resnet dimensions
# new_ = old_ + d_


class split_conv(nn.Module):

    def __init__(self, new_in, old_out, d_out, *args, **kwargs):
        super(split_conv, self).__init__()
        self.ignore_conv = nn.Conv2d(new_in, old_out, *args, **kwargs)
        self.copy_conv = nn.Conv2d(new_in, d_out, *args, **kwargs)

    def forward(self, x):
        ig = self.ignore_conv(x)
        cp = self.copy_conv(x)
        return torch.cat([ig, cp], 1)


# separate batch normalization to old resnet
# channels and newly added channels

class split_bn(nn.Module):

    def __init__(self, old_in, d_in, *args, **kwargs):
        super(split_bn, self).__init__()
        self.ignore_bn = nn.BatchNorm2d(old_in, *args, **kwargs)
        self.copy_bn = nn.BatchNorm2d(d_in, *args, **kwargs)
        self.old_in = old_in
        # self.d_in = d_in

    def forward(self, x):
        ig, cp = x[:, :self.old_in, :, :], x[:, self.old_in:, :, :]
        ig, cp = self.ignore_bn(ig), self.copy_bn(cp)
        return torch.cat([ig, cp], 1)


class Bottleneck(nn.Module):
    expansion = 4

    # for the conv1
    # inplanes = new_in
    # planes = old_out
    # d_out = d
    def __init__(self, inplanes, planes, d, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = split_conv(new_in=inplanes, old_out=planes, d_out=d, kernel_size=1)
        self.bn1 = split_bn(old_in=planes, d_in=d)
        self.conv2 = split_conv(new_in=planes + d, old_out=planes, d_out=d, kernel_size=3, stride=stride, padding=1)
        self.bn2 = split_bn(old_in=planes, d_in=d)
        self.conv3 = split_conv(new_in=planes + d, old_out=planes * self.expansion, d_out=d, kernel_size=1)
        self.bn3 = split_bn(old_in=planes * self.expansion, d_in=d)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        print("x.shape: ", x.shape)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print(out.shape)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        print(out.shape)

        out = self.conv3(out)
        out = self.bn3(out)
        print(out.shape)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class iResNet(nn.Module):

    def __init__(self, block, layers, d):
        self.inplanes = 64 + d
        super(iResNet, self).__init__()
        self.conv1 = split_conv(3 + 1, 64, d, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = split_bn(self.inplanes, d)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 1 * 64, 1 * d, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 2 * 64, 2 * d, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 4 * 64, 4 * d, layers[3], stride=2)
        self.layer4 = self._make_layer(block, 8 * 64, 8 * d, layers[4], stride=2)

    def _make_layer(self, block, planes, d, block_count, stride):
        # calling it downsample; technically it is not downsampling but a
        # linear resampling to adjust number of filter dimensions
        downsample = nn.Sequential(
            split_conv(self.inplanes, planes * block.expansion, d, kernel_size=1, stride=stride),
            split_bn(planes * block.expansion, d),
        )

        layers = []
        layers.append(block(self.inplanes, planes, d, stride, downsample))
        self.inplanes = planes * block.expansion + d
        for i in range(1, block_count):
            layers.append(block(self.inplanes, planes, d))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        print("layer1")
        x = self.layer1(x)
        print("layer2")
        x = self.layer2(x)
        print("layer3")
        x = self.layer3(x)
        print("layer4")
        x = self.layer4(x)

        return x




# these functions are tailored towards modifying resnet weights only.
# so are not very generic

# init_conv takes conv module (old_in, old_out), d_in, d_out
# returns a new split_conv module which processes (old_in + d_in, old_out), (old_in + d_in, d_out) and returns old_out + d_out channeled output
# extra weight parameters are initialized with zeros



def init_conv(conv, mod_conv, d_in, d_out):
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

    # mod_conv = split_conv(old_in + d_in, old_out, d_out, kernel_size=kernel_size, stride=conv.stride, padding=conv.padding)

    assert(mod_conv.copy_conv.weight.shape == cp_filter_weight.shape)
    assert(mod_conv.ignore_conv.weight.shape == ig_filter_weight.shape)

    mod_conv.copy_conv.weight = nn.Parameter(cp_filter_weight)
    mod_conv.ignore_conv.weight = nn.Parameter(ig_filter_weight)
    # return mod_conv

# init_bn takes bn module and d_in
# returns split_bn module which normalizes old_in channels and d_in channels separately
# old_in channels normalization parameters are initialized with bn parameters
# d_in channels normalization parameters are initialized with mu = 0, var = 1, weight = 1, bias = 0

def init_bn(bn, mod_bn, d_in):

    old_in = bn.weight.shape[0]
    # mod_bn = split_bn(old_in, d_in)

    assert(old_in == mod_bn.ignore_bn.weight.shape[0])

    mod_bn.ignore_bn.running_var = bn.running_var
    mod_bn.ignore_bn.running_mean = bn.running_mean
    mod_bn.ignore_bn.weight = bn.weight
    mod_bn.ignore_bn.bias = bn.bias

    nn.init.constant_(mod_bn.copy_bn.running_var, 1)
    nn.init.constant_(mod_bn.copy_bn.running_mean, 0)
    nn.init.constant_(mod_bn.copy_bn.weight, 1)
    nn.init.constant_(mod_bn.copy_bn.bias, 0)

    # return mod_bn


# initialize a single bottleneck block
# takes a resnet block, returns iresnet block
# bottleneck block has three conv modules and 
# optionally one downsample conv
# the modification to bottleneck is defined as follows
# we take a current bottleneck module and parameters 
# p, q that determine how bottleneck is modified
# if downsample == none: p must be equal to q
# this is so that we can pure residual
# if downsample is not none: q determines how many extra channels are added at end of a bottleneck block
# in general modification is as follows
# conv1 -> in_planes, planes => in_planes + p, planes + p
# conv2 -> planes, planes => planes + p, planes + p
# conv3 -> planes, 4 * planes => planes + p, 4 * planes + q
# downsample (!none) -> in_planes, 4 * planes => in_planes + p, 4 * planes + q

def init_bottleneck(block, new_block, d_in, stride):

    inplanes = block.conv1.weight.shape[1]
    planes = block.conv1.weight.shape[0]

    # new_block = Bottleneck(inplanes, planes, d_in, stride)
    p,q,r,s = d_in
    if block.downsample != None:
        init_conv(block.downsample[0],downsample_conv, p, s)
        init_bn(block.downsample[1], downsample_bn, s)
        new_block.downsample = nn.Sequential(downsample_conv, downsample_bn,)
    else:
        # making sure that we can add pure residual
        assert(p == s)
        new_block.downsample = None
    
    init_conv(block.conv1, new_block.conv1, p, q)
    init_bn(block.bn1, new_block.bn1, q)
    init_conv(block.conv2, new_block.conv2, q, r)
    init_bn(block.bn2, new_block.bn2, r)
    init_conv(block.conv3, new_block.conv3, r, s)
    init_bn(block.bn3, new_block.bn3, s)


    # return new_block

def doub(d_in):
    for i in range(len(d_in)):
        d_in[i] = 2 * d_in[i]

# initialize a layer of blocks
def init_layer(layer, new_layer, block_count, d_in, stride):
    assert(len(layer) == len(new_layer))
    # middle numbers can be fiddled in anyway
    # without affecting correctness
    # d_in = [8,4,4,16]
    # d_in = [16,8,8,32]
    # d_in = [32,16,16,64]
    # d_in = [64,32,32,128]
    init_bottleneck(layer[0], new_layer[0], d_in, stride)
    for i in range(1, block_count):
        # d_in = [16,4,4,16]
        # d_in = [32,8,8,32]
        # d_in = [64,16,16,64]
        # d_in = [128,32,32,128]
        d_in[0] = d_in[-1]
        append(init_bottleneck(layer[i], new_layer[0], d_in, d_in, 1))

# modify resnet to include additional channels
def modify_resnet():
    d_in = 4
    layers = [0, 3, 4, 23, 3]
    resnet = models.resnet101(pretrained=True)
    iresnet = iResNet(Bottleneck, layers, d_in)

    init_conv(resnet.conv1, iresnet.conv1, 1, d_in)
    init_bn(resnet.bn1, iresnet.bn1, d_in)

    d_in = [4,2,2,8]

    doub(d_in)
    init_layer(resnet.layer1, iresnet.layer1, layers[1],d_in, stride=1)
    doub(d_in)
    init_layer(resnet.layer2, iresnet.layer2, layers[2], d_in, stride=2)
    doub(d_in)
    init_layer(resnet.layer3, iresnet.layer3, layers[3], d_in, stride=2)
    doub(d_in)
    init_layer(resnet.layer4, iresnet.layer4, layers[4], d_in, stride=2)

    return iresnet

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

    # net0 = resnet()
    # net0 = net0.cuda()

    net1 = modify_resnet()
    net1 = net1.cuda()
    print(net1(inp).shape)
    # assert(net0(x) == net1(x)[:128])