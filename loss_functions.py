import torch
import torch.nn as nn

# 
def loss_criterion1(pred, gt):
    pred_masks, pred_class = pred
    
    gt_masks, gt_class = gt
    # gt_masks = [g.unsqueeze(1) for g in gt_masks]
    gt_masks = torch.stack(gt_masks, 1).float()
    gt_class = torch.stack(gt_class).float()
    print(gt_masks.shape, pred_masks.shape, gt_class.shape, pred_class.shape)

    mask_loss = soft_iou(pred_masks, gt_masks)
    mask_loss = mask_loss.mean()

    class_loss = ce_class_loss(pred_class, gt_class)
    class_loss = class_loss.mean()

    return mask_loss + class_loss

# Classifcation losses available: crossentropy

# inputs: one hot repn of gt labels, prob of pred labels
# gt_shape: (B, N), pred_shape: (B, N)
# loss_shape: (B,)

def ce_class_loss(pred, gt):
    _loss = nn.CrossEntropyLoss(reduction='none')
    labels = gt.nonzero()[:, 1]
    l = _loss(pred, labels)
    return l

# Mask losses available: soft_iou, balanced_bce
# input pred_masks and gt_masks of same shape
# pred_masks are point wise probabilities .i.e,
# after sigmoid application
# both must be of same shape: (B, C, w, h)
# output loss shape: same as input shape

def soft_iou(pred_masks, gt_masks):
    i = (pred_masks * gt_masks).sum(-1).sum(-1)
    u = (pred_masks + gt_masks - pred_masks * gt_masks).sum(-1).sum(-1)
    l = 1 - i / u
    return l


def balanced_bce(pred, gt):
    return 0


# Easy example selection: takes tensor of losses and returns least n% of the losses


def select_easy(l, frac=1):
    return l
    # sort(L)
    # l = math.ceil(len(L) * frac)
    # return L[:l]
