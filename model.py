import torch
import torch.nn as nn
import torch.nn.functional as F
import iresnet
import utils


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


class mask_branch(nn.Module):

    def __init__(self, block_counts, out_planes=1):
        super(mask_branch, self).__init__()

        bc5, bc4, bc3 = block_counts
        planes = 512
        d = 8
        e = Bottleneck.expansion
        self.layer4 = self._make_layer(Bottleneck, inplanes=4 * (8 * 64 + (8 * d) // 2), planes=planes // 1, bc=bc5)
        self.layer3 = self._make_layer(Bottleneck, inplanes=4 * (4 * 64 + (4 * d) // 2) + e * planes // 1, planes=planes // 2, bc=bc4)
        self.layer2 = self._make_layer(Bottleneck, inplanes=4 * (2 * 64 + (2 * d) // 2) + e * planes // 2, planes=planes // 4, bc=bc3)
        self.layer1 = self._make_layer(Bottleneck, inplanes=4 * (1 * 64 + (1 * d) // 2) + e * planes // 4, planes=planes // 8, bc=bc3)

        self.mask_layer = nn.Sequential(
            nn.Conv2d(planes//2, 1, kernel_size=(3, 3), bias=False, padding=(1,1)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        l1, l2, l3, l4 = x
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

        y = F.interpolate(y, scale_factor=2)
        y = self.mask_layer(y)
        return y

    def _make_layer(self, block, inplanes, planes, bc):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, block.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(block.expansion * planes),
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
        x = self.pool1(x)
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
        self.mb0 = mask_branch([3, 3, 3])
        # self.mb1 = mask_branch([3, 3, 3])
        self.cb = class_branch()

    def forward(self, x):
        img, impulse = x
        impulse.unsqueeze_(1)
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
