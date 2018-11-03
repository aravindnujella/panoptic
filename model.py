import torch
import torch.nn as nn
import torch.nn.functional as F
import iresnet
import ResidualBlock as rb

import utils


class mask_branch(nn.Module):

    def __init__(self, block_counts, out_planes=1):
        super(mask_branch, self).__init__()

        bc4, bc3, bc2, bc1, bc0, bci = block_counts
        planes = 256
        d = 8
        e = 1
        self.layer4 = rb._make_residual_layer(inplanes=512, planes=planes // 1, outplanes=(planes // 1) * e, bc=bc4)
        self.layer3 = rb._make_residual_layer(inplanes=256 + e * planes // 1, planes=planes // 2, outplanes=(planes // 2) * e, bc=bc3)
        self.layer2 = rb._make_residual_layer(inplanes=128 + e * planes // 2, planes=planes // 4, outplanes=(planes // 4) * e, bc=bc2)
        self.layer1 = rb._make_residual_layer(inplanes=64 + e * planes // 4, planes=planes // 8, outplanes=(planes // 8) * e, bc=bc1)
        self.layer0 = rb._make_residual_layer(inplanes=32 + e * planes // 8, planes=planes // 16, outplanes=(planes // 16) * e, bc=bc0)
        self.layeri = rb._make_residual_layer(inplanes=16 + e * planes // 16, planes=planes // 32, outplanes=(planes // 32), bc=bci)

        self.mask_layer = nn.Sequential(
            nn.Conv2d((planes // 32), 1, kernel_size=(3, 3), padding=(1,1), bias=False),
            nn.BatchNorm2d(1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        (li, l0, l1, l2, l3, l4) = x
        del(x)
        masks = []

        y = self.layer4(l4)
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

        y = self.mask_layer(y)
        return y


class class_branch(nn.Module):

    def __init__(self):
        super(class_branch, self).__init__()
        self.cl1 = nn.Sequential(nn.Conv2d(2048 + 128, 512, kernel_size=(1, 1), padding=(0, 0), bias=False),
                                 nn.ReLU(),
                                 )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                 nn.ReLU(),
                                 )
        self.pool2 = nn.MaxPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * 7 * 7, 134)

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
            nn.Conv2d(inplanes, out_planes, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
        )


class hgmodel(nn.Module):

    def __init__(self):
        super(hgmodel, self).__init__()
        self.iresnet0 = iresnet.iresnet50(pretrained=True)
        # self.iresnet1 = iresnet.iresnet50(pretrained=True)
        self.mb0 = mask_branch([2, 2, 2, 2, 2, 2])
        # self.mb1 = mask_branch([1, 3, 3])
        self.cb = class_branch()

    def forward(self, x):
        img, impulse = x
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
