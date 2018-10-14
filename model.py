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
        planes = 256
        e = Bottleneck.expansion
        self.layer4 = self._make_layer(Bottleneck, inplanes=2048 + 64, planes=planes // 1, bc=bc5)
        self.layer3 = self._make_layer(Bottleneck, inplanes=1024 + 32 + e * planes // 1, planes=planes // 2, bc=bc4)
        self.layer2 = self._make_layer(Bottleneck, inplanes=512 + 16 + e * planes // 2, planes=planes // 4, bc=bc3)

    def forward(self, x):
        l2, l3, l4 = x
        print(l2.shape, l3.shape, l4.shape)
        masks = []

        y = self.layer4(l4)
        y = F.interpolate(y,scale_factor=2)

        y = self.layer3(torch.cat([y, l3], 1))
        y = F.interpolate(y,scale_factor=2)

        y = self.layer2(torch.cat([y, l2], 1))
        y = F.interpolate(y,scale_factor=2)

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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.cl1 = self._make_layer()
        self.cl2 = self._make_layer()
        self.avg = nn.AvgPool2d(kernel_size=16,stride=1)
        self.fc = nn.Linear(2048,121)

    def forward(self, x):
        x = self.cl1(x)
        x = self.pool(x)
        x = self.cl2(x)
        x = self.avg(x)
        x = x.view(-1, 121)
        x = self.fc(x)
        return x

    def _make_layer(self):
        return None

class hgmodel(nn.Module):

    def __init__(self):
        super(hgmodel, self).__init__()
        self.iresnet1 = iresnet.iresnet101(pretrained=False)
        self.iresnet2 = iresnet.iresnet101(pretrained=False)
        self.mb0 = mask_branch([3,3,3])
        self.mb1 = mask_branch([3,3,3])
        self.cb = class_branch()

    def forward(self, x):
        # print(x[0][0].shape, x[1][0].shape)
        # print(len(x[0]), len(x[1]))
        img, impulse = self.unpack_imgs(x)
        print(img.shape, impulse.shape)
        inp = torch.cat([img, impulse], 1)
        resnet_feats = self.iresnet1(inp)
        m0 = self.mb0(resnet_feats)

        del(resnet_feats)
        m0 = F.interpolate(m0, scale_factor=4)

        inp = torch.cat([img, m0], 1)
        resnet_feats = self.iresnet2(inp)
        m1 = self.mb1(resnet_feats)

        c = self.cb(resnet_feats[-1])
        return c, m1


    def unpack_imgs(self, x):
        imgs, impulses = x
        new_imgs = []
        
        for i,img in enumerate(imgs,0):
            rep = impulses[i].shape[0]
            img = torch.cat([img.unsqueeze(0)]*rep, 0)
            new_imgs.append(img)
        
        new_impulses = [it.unsqueeze(1) for it in impulses]

        return torch.cat(new_imgs,0).float(), torch.cat(new_impulses,0).float()

