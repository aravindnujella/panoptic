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
        self.conv1 = iconv(inplanes + p, planes, q, kernel_size=1, bias=False)
        self.bn1 = ibn(planes, q)
        self.conv2 = iconv(planes + q, planes, r, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn2 = ibn(planes, r)
        self.conv3 = iconv(planes + r, planes * self.expansion, s, kernel_size=1,bias=False)
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

    def __init__(self, block, block_counts, ichannels=1, d=4):
        super(iResNet, self).__init__()
        self.inplanes = 64
        self.d = d
        self.block_counts = block_counts
        self.ichannels = ichannels
        self.conv1 = iconv(3 + ichannels, 64, d, kernel_size=7, stride=2, padding=3,
                           bias=False)
        self.bn1 = ibn(64, d)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 1 * 64, 1 * d, block_counts[1], stride=1)
        self.layer2 = self._make_layer(block, 2 * 64, 2 * d, block_counts[2], stride=2)
        self.layer3 = self._make_layer(block, 4 * 64, 4 * d, block_counts[3], stride=2)
        self.layer4 = self._make_layer(block, 8 * 64, 8 * d, block_counts[4], stride=2)

    def _make_layer(self, block, planes, d, block_count, stride):

        ds = [d, d // 2, d // 2, 2 * d]

        downsample = nn.Sequential(
            iconv(self.inplanes + ds[0], planes * block.expansion, ds[-1], kernel_size=1, stride=stride, bias=False),
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


def init_conv(iconv, conv, d_in, d_out):
    weight = conv.weight
    old_out = weight.shape[0]
    old_in = weight.shape[1]
    kernel_size = weight.shape[2:]

    c = torch.zeros((old_out, d_in) + kernel_size)
    ig_filter_weight = torch.cat([weight, c], 1)

    fan_in = kernel_size[0] * kernel_size[1]
    a = torch.zeros((d_out, old_in,) + kernel_size)
    b = torch.zeros((d_out, d_in,) + kernel_size)
#     b = torch.eye(d_out, d_in).unsqueeze(-1).unsqueeze(-1)
#     b = b.repeat([1, 1, kernel_size[0], kernel_size[1]]) / fan_in
    cp_filter_weight = torch.cat([a, b], 1)

    assert(iconv.copy_conv.weight.shape == cp_filter_weight.shape)
    assert(iconv.ignore_conv.weight.shape == ig_filter_weight.shape)

    iconv.copy_conv.weight = nn.Parameter(cp_filter_weight)
    iconv.ignore_conv.weight = nn.Parameter(ig_filter_weight)

def init_bn(ibn, bn, d_in):

    old_in = bn.weight.shape[0]
    assert(old_in == ibn.ignore_bn.weight.shape[0])

    ibn.ignore_bn.running_var = bn.running_var
    ibn.ignore_bn.running_mean = bn.running_mean
    ibn.ignore_bn.weight = bn.weight
    ibn.ignore_bn.bias = bn.bias

    nn.init.constant_(ibn.copy_bn.running_var, 1)
    nn.init.constant_(ibn.copy_bn.running_mean, 0)
    nn.init.constant_(ibn.copy_bn.weight, 1)
    nn.init.constant_(ibn.copy_bn.bias, 0)


def init_bottleneck(iblock, block, ds):

    inplanes = block.conv1.weight.shape[1]
    planes = block.conv1.weight.shape[0]

    p,q,r,s = ds
    if block.downsample != None:
        init_conv(iblock.downsample[0], block.downsample[0], p, s)
        init_bn(iblock.downsample[1], block.downsample[1], s)
    else:
        assert(p == s)
        iblock.downsample = None

    init_conv(iblock.conv1, block.conv1, p, q)
    init_bn(iblock.bn1, block.bn1, q)
    init_conv(iblock.conv2, block.conv2, q, r)
    init_bn(iblock.bn2, block.bn2, r)
    init_conv(iblock.conv3, block.conv3, r, s)
    init_bn(iblock.bn3, block.bn3, s)


def init_layer(ilayer, layer, block_count, d):
    assert(len(ilayer) == len(layer))
    # !! repeated logic
    ds = [d, d // 2, d // 2, 2 * d]
    init_bottleneck(ilayer[0], layer[0], ds)
    ds[0] = ds[-1]
    for i in range(1, block_count):
        init_bottleneck(ilayer[i], layer[i], ds)


def init_pretrained(iresnet, resnet):

    d = iresnet.d
    block_counts = iresnet.block_counts
    ichannels = iresnet.ichannels

    init_conv(iresnet.conv1, resnet.conv1, ichannels, d)
    init_bn(iresnet.bn1, resnet.bn1, d)

    init_layer(iresnet.layer1, resnet.layer1, block_counts[1], 1 * d)
    init_layer(iresnet.layer2, resnet.layer2, block_counts[2], 2 * d)
    init_layer(iresnet.layer3, resnet.layer3, block_counts[3], 4 * d)
    init_layer(iresnet.layer4, resnet.layer4, block_counts[4], 8 * d)


# In[74]:


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

        return x
def resnet101():
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


# In[75]:


# main
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

img = img.cuda()
inp = inp.cuda()


iresnet = iResNet(iBottleneck, [0, 3, 4, 23, 3], 1, 4)

resnet = resnet101()
init_pretrained(iresnet, resnet)

iresnet = iresnet.cuda()
resnet = resnet.cuda()
neq = (iresnet(inp)[:,:2048,:,:] != resnet(img))
print(torch.sum(neq))
print(iresnet(inp).shape)


# In[45]:


x = iresnet(inp)
y = resnet(img)


# In[46]:


x[0,0,:,:]


# In[47]:


y[0,0,:,:]


# In[48]:


for i in range(64):
    print((x[0,i,:,:] != y[0,i,:,:]).sum())
