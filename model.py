import torch
import torch.nn as nn
import torch.nn.functional as F
import iresnet
from ResidualBlock import *

import utils

class mask_branch(nn.Module):

    def __init__(self, block_counts, out_planes=1):
        super(mask_branch, self).__init__()

        bc4, bc3, bc2, bc1, bc0, bci = block_counts
        planes = 256
        d = 8
        e = 4
        self.layer4 = self._make_residual_layer(inplanes=256, planes=planes // 1, outplanes=(planes // 1)*e,bc=bc4)
        self.layer3 = self._make_residual_layer(inplanes=128 + e * planes // 1, planes=planes // 2, outplanes=(planes // 2)*e,bc=bc3)
        self.layer2 = self._make_residual_layer(inplanes=64 + e * planes // 2, planes=planes // 4, outplanes=(planes // 4)*e,bc=bc2)
        self.layer1 = self._make_residual_layer(inplanes=32 + e * planes // 4, planes=planes // 8, outplanes=(planes // 8)*e,bc=bc1)
        self.layer1 = self._make_residual_layer(inplanes=16 + e * planes // 8, planes=planes // 16, outplanes=(planes // 16)*e,bc=bc0)
        self.layer0 = self._make_residual_layer(inplanes=8 + e * planes // 16, planes=planes // 32, outplanes=(planes // 32)*e,bc=bci)

        self.layer0 = nn.Sequential(
            nn.Conv2d(planes//2, 1, kernel_size=(7, 7), bias=False, padding=(3,3)),
            nn.BatchNorm2d(1, momentum=0.1,track_running_stats=True),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        y, (li, l0, l1, l2, l3, l4) = x
        del(x)
        masks = []

        y = self.layer4(torch.cat([y, l4]), 1)
        y = F.interpolate(y, scale_factor=2)

        y = self.layer3(torch.cat([y, l3], 1))
        y = F.interpolate(y, scale_factor=2)

        y = self.layer2(torch.cat([y, l2], 1))
        y = F.interpolate(y, scale_factor=2)

        y = self.layer1(torch.cat([y, l1], 1))
        y = F.interpolate(y, scale_factor=2)

        y = self.layer0(torch.cat([y, l0], 1))
        y = F.interpolate(y, scale_factor=2)

        y = self.layeri(torch.cat([y, li], 1))

        return y

    def _make_layer(self, inplanes, planes, out_planes, bc):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, block.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(block.expansion * planes, momentum=0.1,track_running_stats=True),
        )
        layers = []
        layers.append(block(inplanes, planes, downsample=downsample))
        for i in range(1, bc):
            layers.append(block(block.expansion * planes, planes, downsample=None))

        return nn.Sequential(*layers)

class class_branch(nn.Module):

    def __init__(self):
        super(class_branch, self).__init__()
        self.cl1 = nn.Sequential(nn.Conv2d(2048+128, 512, kernel_size=(1,1), padding=(0,0), bias=False),
                                nn.ReLU(),
                                )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3,3), padding=(1,1), bias=False),
                                nn.ReLU(),
                                )
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512*7*7, 134)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(m.weight, 0)
    def forward(self, x):
        x = self.cl1(x)
        x = self.cl2(x)
        # x = self.pool1(x)
        # x = self.pool2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def _make_layer(self, inplanes, out_planes):
         return nn.Sequential(
            nn.Conv2d(inplanes, out_planes, kernel_size=(1,1), padding=(0,0),bias=False),
            nn.ReLU(),
            )            


class hgmodel(nn.Module):

    def __init__(self):
        super(hgmodel, self).__init__()
        self.iresnet0 = iresnet.iresnet50(pretrained=True)
        # self.iresnet1 = iresnet.iresnet50(pretrained=True)
        self.mb0 = mask_branch([1, 2, 3])
        # self.mb1 = mask_branch([3, 3, 3])
        self.cb = class_branch()

    def forward(self, x):
        img, impulse = x
        impulse.unsqueeze_(1)
        impulse -= 0.5
        # print(img.shape, impulse.shape)
        inp = torch.cat([img, impulse], 1)
        del(impulse)
        cf, mf = self.iresnet0(inp)
        m0 = self.mb0(mf)

        # del(mf); del(cf)

        # inp = torch.cat([img, m0], 1)
        # with torch.no_grad():
        #     cf, mf = self.iresnet1(inp)
        # m1 = self.mb1(mf)

        c = self.cb(cf)

        return m0, c
