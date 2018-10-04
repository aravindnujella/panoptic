import torch.nn as nn
import torch

import iresnet

class mask_branch(nn.Module):

    def __init__(self):
        super(mask_branch, self).__init__()

    def forward(self, x):
        return x


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
