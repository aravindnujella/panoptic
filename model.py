import torch.nn as nn
import torch

import iresnet


class mask_branch(nn.Module):

    def __init__(self, block_counts, out_planes=1):
        super(mask_branch, self).__init__()
        bc5, bc4, bc3 = block_counts
        block = iresnet.iBottleNeck
        self.layer5 = self._make_layer(block, bc5, inplanes=2048+64, planes=512)
        self.layer4 = self._make_layer(block, bc4, inplanes=, planes=)
        self.layer3 = self._make_layer(block, bc3, inplanes=, planes=)

    def forward(self, x):
        l3, l4, l5 = m
        masks = []

        # y = self.layer5(torch.cat([c, l5], 1))
        y = l5
        y = self.upsample(y)

        y = self.layer4(torch.cat([y, l4], 1))
        y = self.upsample(y)

        y = self.layer3(torch.cat([y, l3], 1))
        y = self.mask_layer3(y)

        return y


class class_branch(nn.Module):

    def __init__(self):
        super(class_branch, self).__init__()

    def forward(self, x):
        
        return x


class hgmodel(nn.Module):

    def __init__(self):
        super(hgmodel, self).__init__()
        self.iresnet = iresnet.iresnet101(pretrained=True)
        self.mb0 = mask_branch()
        self.mb1 = mask_branch()
        self.cb = class_branch()

    # TOTO: Copy from single-object-detection
    def forward(self, x):
        img, impulse = x
        inp = torch.cat([img, impulse], 1)

        class_feats, mask_feats = self.iresnet(x)

        m0 = self.mb0([mask_feats, class_feats])
        m1 = 0

        c = self.cb(class_feats)
