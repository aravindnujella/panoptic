import torch
import torch.nn as nn


def loss_criterion(pred, gt):
    pred_masks, pred_scores = pred

    gt_masks, gt_labels = gt
    gt_masks.unsqueeze_(1)
    gt_labels = gt_labels.long()

    mask_loss = soft_iou(pred_masks, gt_masks)
    class_loss = ce_class_loss(pred_scores, gt_labels)

    return mask_loss

# Classifcation losses available: crossentropy

# inputs: one hot repn of gt labels, prob of pred labels
# gt_shape: (B), pred_shape: (B, N)
# loss_shape: (B,)


def ce_class_loss(pred_scores, gt_labels):
    _loss = nn.CrossEntropyLoss(reduction='none')
    l = _loss(pred_scores, gt_labels)
    return l.mean()

# Mask losses available: soft_iou, balanced_bce
# input pred_masks and gt_masks of same shape
# pred_masks are point wise probabilities .i.e,
# after sigmoid application
# both must be of same shape: (B, C, w, h)
# output loss shape: (B,)


def soft_iou(pred_masks, gt_masks):
    pred_masks = pred_masks.sigmoid()
    i = (pred_masks * gt_masks).sum(-1).sum(-1)
    u = (pred_masks + gt_masks - pred_masks * gt_masks).sum(-1).sum(-1)
    l = 1 - i / u
    return l.mean()


def balanced_bce(pred_masks, gt_masks):
    fg_size = gt_masks.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    bg_size = (1 - gt_masks).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)

    _loss = nn.BCEWithLogitsLoss(reduction='none')

    # loss where gt is +ve, -ve
    fg_loss = _loss(pred_masks * gt_masks, gt_masks)
    bg_loss = _loss(pred_masks * (1 - gt_masks), gt_masks)

    l = (bg_size * fg_loss + fg_size * bg_loss) / (fg_size + bg_size)
    return l.mean()


# Easy example selection: takes tensor of losses and returns least n% of the losses


def select_easy(l, frac=1):
    return l
    # sort(L)
    # l = math.ceil(len(L) * frac)
    # return L[:l]
