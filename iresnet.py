import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

import iResidualBlock as irb
import ResidualBlock as rb

import numpy as np

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class iResNet(nn.Module):

    def __init__(self, block_counts, d=4, ichannels=1):
        super(iResNet, self).__init__()
        inplanes = 64
        self.d = d
        self.block_counts = block_counts
        self.ichannels = ichannels

        # 0
        self.conv1 = irb.iConv2d(3 + ichannels, 64, d, kernel_size=7, stride=2, padding=3,
                                 bias=False)
        self.bn1 = irb.iBatchNorm2d(64, d)
        self.relu = nn.ReLU(inplace=True)
        # 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = irb._make_iresidual_layer(1 * inplanes, 1 * 64, 1 * d, block_counts[1], stride=1)
        # 2
        self.layer2 = irb._make_iresidual_layer(4 * inplanes, 2 * 64, 2 * d, block_counts[2], stride=2)
        # 3
        self.layer3 = irb._make_iresidual_layer(8 * inplanes, 4 * 64, 4 * d, block_counts[3], stride=2)
        # 4
        self.layer4 = irb._make_iresidual_layer(16 * inplanes, 8 * 64, 8 * d, block_counts[4], stride=2)

        self.wingi = rb._make_residual_layer(4, 8, 8, 3)
        self.wing0 = rb._make_residual_layer(64 + d, 16, 16, 1)
        self.wing1 = rb._make_residual_layer(256 + 2 * d, 32, 32, 1)
        self.wing2 = rb._make_residual_layer(512 + 4 * d, 64, 64, 1)
        self.wing3 = rb._make_residual_layer(1024 + 8 * d, 128, 128, 1)
        self.wing4 = rb._make_residual_layer(2048 + 16 * d, 256, 256, 1)

    def forward(self, x):
        # outs = [torch.cat([x[:,-1,:,:].unsqueeze(1) for i in range(8)], 1)]
        outs = [self.wingi(x)]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x); outs.append(self.wing0(x))

        x = self.maxpool(x)
        x = self.layer1(x); outs.append(self.wing1(x))

        x = self.layer2(x); outs.append(self.wing2(x))

        x = self.layer3(x); outs.append(self.wing3(x))

        x = self.layer4(x); outs.append(self.wing4(x))

        return x, outs


def init_conv(iconv, conv, d_in, d_out, val=0):
    weight = conv.weight
    old_out = weight.shape[0]
    old_in = weight.shape[1]
    kernel_size = weight.shape[2:]

    c = torch.zeros((old_out, d_in) + kernel_size)
    ig_filter_weight = torch.cat([weight, c], 1)

    fan_in = kernel_size[0] * kernel_size[1]
    a = torch.zeros((d_out, old_in,) + kernel_size)

    b = np.zeros((d_out, d_in,))
    if val == 0:
        b = np.zeros((d_out, d_in,))
    else:
        b = np.eye(d_out, d_in)
        if d_out > d_in:
            idx = np.array([i % d_in for i in range(d_out)])
            b[range(d_out), idx] = 1

    b = torch.from_numpy(b).unsqueeze(-1).unsqueeze(-1).float()
    b = b.repeat([1, 1, kernel_size[0], kernel_size[1]]) / fan_in

    cp_filter_weight = torch.cat([a, b], 1)

    # print(iconv.copy_conv.weight.shape, cp_filter_weight.shape)
    assert(iconv.copy_conv.weight.shape == cp_filter_weight.shape)
    assert(iconv.ignore_conv.weight.shape == ig_filter_weight.shape)

    iconv.copy_conv.weight = nn.Parameter(cp_filter_weight)
    iconv.ignore_conv.weight = nn.Parameter(ig_filter_weight)


def init_bn(ibn, bn, d_in):

    old_in = bn.weight.shape[0]
    assert(old_in == ibn.ignore_bn.weight.shape[0])

    ibn.ignore_bn.running_var = bn.running_var
    ibn.ignore_bn.running_mean = bn.running_mean
    ibn.ignore_bn.weight = nn.Parameter(bn.weight)
    ibn.ignore_bn.bias = nn.Parameter(bn.bias)


def init_bottleneck(iblock, block, ds):

    inplanes = block.conv1.weight.shape[1]
    planes = block.conv1.weight.shape[0]

    p, q, r, s = ds
    if block.downsample != None:
        init_conv(iblock.downsample[0], block.downsample[0], p, s, val=1)
        init_bn(iblock.downsample[1], block.downsample[1], s)
    else:
        assert(p == s)
        block.downsample = None

    init_conv(iblock.conv1, block.conv1, p, q)
    init_bn(iblock.bn1, block.bn1, q)
    init_conv(iblock.conv2, block.conv2, q, r)
    init_bn(iblock.bn2, block.bn2, r)
    init_conv(iblock.conv3, block.conv3, r, s)
    init_bn(iblock.bn3, block.bn3, s)


def init_layer(ilayer, layer, block_count, d):
    assert(len(ilayer) == len(layer))
    ds = [d, d // 2, d // 2, 2 * d]
    init_bottleneck(ilayer[0], layer[0], ds)
    ds[0] = ds[-1]
    for i in range(1, block_count):
        init_bottleneck(ilayer[i], layer[i], ds)


def init_pretrained(iresnet, resnet):

    d = iresnet.d
    block_counts = iresnet.block_counts
    ichannels = iresnet.ichannels

    init_conv(iresnet.conv1, resnet.conv1, ichannels, d, val=1)
    init_bn(iresnet.bn1, resnet.bn1, d)

    init_layer(iresnet.layer1, resnet.layer1, block_counts[1], 1 * d)
    init_layer(iresnet.layer2, resnet.layer2, block_counts[2], 2 * d)
    init_layer(iresnet.layer3, resnet.layer3, block_counts[3], 4 * d)
    init_layer(iresnet.layer4, resnet.layer4, block_counts[4], 8 * d)


def iresnet101(pretrained=False):
    block_counts = [0, 3, 4, 23, 3]
    iresnet = iResNet(block_counts, d=4)
    resnet = models.resnet101(pretrained=True)
    if pretrained:
        init_pretrained(iresnet, resnet)
    return iresnet


def iresnet50(pretrained=False):
    block_counts = [0, 3, 4, 6, 3]
    iresnet = iResNet(block_counts, d=8)
    resnet = models.resnet50(pretrained=True)
    if pretrained:
        init_pretrained(iresnet, resnet)
    return iresnet

if __name__ == '__main__':

    import os
    import time
    import torch
    import torchvision.models as models

    import numpy as np
    torch.set_printoptions(threshold=float('nan'))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    def resnet_conv(resnet, x):
        x = resnet.conv1(x)
        x = resnet.bn1(x)
        x = resnet.relu(x)
        x = resnet.maxpool(x)

        x = resnet.layer1(x)
        x = resnet.layer2(x)
        x = resnet.layer3(x)
        x = resnet.layer4(x)

        return x

    from PIL import Image
    img = Image.open("./data/1.png").convert('RGB')
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

    impulse = torch.zeros(img.shape[-2:])
    w, h = img.shape[-2:]
    impulse[w // 2 - 32:w // 2 + 32, h // 2 - 32:h // 2 + 32] = 1
    Image.fromarray((impulse.numpy() * 255).astype(np.uint8), "L").show()

    impulse.unsqueeze_(0).unsqueeze_(0)
    inp = torch.cat([img, impulse], 1)

    img = img.cuda()
    inp = inp.cuda()

    iresnet = iresnet50(pretrained=True)

    resnet = models.resnet50(pretrained=True)

    iresnet = iresnet.cuda()
    resnet = resnet.cuda()

    neq = (iresnet(inp)[0][:, :2048, :, :] != resnet_conv(resnet, img))
    print(torch.sum(neq))
    # print(iresnet(inp)[0].shape)
    print(iresnet(inp)[0][0, 2047, :, :])
    # print(resnet_conv(resnet, img)[0, 2047, :, :])
    print(iresnet(inp)[0][0, 2048:, :, :])
