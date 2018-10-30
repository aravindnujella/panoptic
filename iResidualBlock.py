import torch
import torch.nn as nn
import torch.nn.functional as F


class iConv2d(nn.Module):

    def __init__(self, new_in, old_out, d_out, *args, **kwargs):
        super(iConv2d, self).__init__()
        self.ignore_conv = nn.Conv2d(new_in, old_out, *args, **kwargs)
        self.copy_conv = nn.Conv2d(new_in, d_out, *args, **kwargs)

    def forward(self, x):
        ig = self.ignore_conv(x)
        cp = self.copy_conv(x)
        return torch.cat([ig, cp], 1)


class iBatchNorm2d(nn.Module):

    def __init__(self, old_in, d_in, *args, **kwargs):
        super(iBatchNorm2d, self).__init__()
        self.ignore_bn = nn.BatchNorm2d(old_in, *args, **kwargs, momentum=0.1)
        self.copy_bn = nn.BatchNorm2d(d_in, *args, **kwargs, momentum=0.1)
        self.old_in = old_in
        self.d_in = d_in

    def forward(self, x):
        ig, cp = x[:, :self.old_in, :, :], x[:, self.old_in:, :, :]
        ig, cp = self.ignore_bn(ig), self.copy_bn(cp)
        return torch.cat([ig, cp], 1)


# residual block and iResidual are same except for impulse channels.
class iResidual(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, d, stride=1, downsample=None):
        super(iResidual, self).__init__()
        p, q, r, s = d
        self.conv1 = iConv2d(inplanes + p, planes, q, kernel_size=1, bias=False)
        self.bn1 = iBatchNorm2d(planes, q)
        self.conv2 = iConv2d(planes + q, planes, r, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = iBatchNorm2d(planes, r)
        self.conv3 = iConv2d(planes + r, planes * self.expansion, s, kernel_size=1, bias=False)
        self.bn3 = iBatchNorm2d(planes * self.expansion, s)
        self.relu = nn.ReLU(inplace=True)
        outplanes = inplanes * self.expansion
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


def _make_iresidual_layer(inplanes, planes, d, bc, stride=1):
    ds = [d, d // 2, d // 2, 2 * d]

    downsample = nn.Sequential(
        iConv2d(inplanes + ds[0], planes * iResidual.expansion, ds[-1], kernel_size=1, stride=stride, bias=False),
        iBatchNorm2d(planes * iResidual.expansion, ds[-1]),
    )

    layers = []
    layers.append(iResidual(inplanes, planes, ds, stride, downsample))
    inplanes = planes * iResidual.expansion
    ds[0] = ds[-1]
    for i in range(1, bc):
        layers.append(iResidual(inplanes, planes, ds))

    return nn.Sequential(*layers)
