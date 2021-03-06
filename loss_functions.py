import torch
import torch.nn as nn
import torch.nn.functional as F


def loss_criterion(pred, gt):
    mask_logits, cat_scores = pred

    gt_masks, gt_labels = gt
    gt_masks.unsqueeze_(1)
    gt_labels = gt_labels.long()

    mask_loss = balanced_bce(mask_logits, gt_masks)
    class_loss = ce_class_loss(cat_scores, gt_labels)

    return mask_loss


# Classifcation losses available: crossentropy

# inputs: one hot repn of gt labels, prob of pred labels
# gt_shape: (B), pred_shape: (B, N)
# loss_shape: (B,)


def ce_class_loss(cat_scores, gt_labels):
    _loss = nn.CrossEntropyLoss(reduction='none')
    l = _loss(cat_scores, gt_labels)
    return l.mean()


# Mask losses available: soft_iou, balanced_bce;
# multi_scale_mask_loss which can inturn use above two losses
# input pred_masks and gt_masks of same shape
# pred_masks are (pre-sigmoid) logit scores .i.e,
# both must be of same shape: (B, C, w, h)
# output loss shape: (B,)


def soft_iou(mask_logits, gt_masks):
    assert(mask_logits.shape == gt_masks.shape)
    pred_masks = mask_logits.sigmoid()
    i = (pred_masks * gt_masks).sum(-1).sum(-1)
    u = (pred_masks + gt_masks - pred_masks * gt_masks).sum(-1).sum(-1)
    l = 1 - i / u
    return l.mean()


def balanced_bce(mask_logits, gt_masks):
    assert(mask_logits.shape == gt_masks.shape)
    fg_size = gt_masks.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    bg_size = (1 - gt_masks).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)

    _loss = nn.BCEWithLogitsLoss(reduction='none')

    # loss where gt is +ve, -ve
    fg_loss = _loss(mask_logits * gt_masks, gt_masks)
    bg_loss = _loss(mask_logits * (1 - gt_masks), gt_masks)

    l = (bg_size * fg_loss + fg_size * bg_loss) / (fg_size + bg_size)
    return l.mean()

# list of masks. each in format (B,1,w,h)
# =>[(B,1,w,h)]


def msm_loss(multi_mask_logits, multi_gt_masks, _loss_fn):
    n = len(multi_mask_logits)
    _losses = []

    for i in range(n):
        _loss = _loss_fn(multi_mask_logits[i], multi_gt_masks[i])
        _losses.append(_loss)

    return torch.stack(_losses)

# Easy example selection: takes tensor of losses and returns least n% of the losses


def select_easy(l, frac=1):
    return l
    # sort(L)
    # l = math.ceil(len(L) * frac)
    # return L[:l]


# Accuracy metrics
#
def mean_iou(mask_logits, gt_masks, cutoff=0.5):
    assert(mask_logits.shape == gt_masks.shape)
    with torch.no_grad():
        gt_masks = gt_masks.float()
        pred_masks = mask_logits.sigmoid()
        pred_masks = (pred_masks > cutoff).float()
        i = ((pred_masks * gt_masks) > 0).sum(-1).sum(-1).float()
        u = ((pred_masks + gt_masks) > 0).sum(-1).sum(-1).float()
        iou = (i / u).squeeze()
        return iou.mean()


def segmentation_quality(mask_logits, gt_masks, cutoff=0.5):
    assert(mask_logits.shape == gt_masks.shape)
    with torch.no_grad():
        gt_masks = gt_masks.float()
        pred_masks = mask_logits.sigmoid()
        pred_masks = (pred_masks > cutoff).float()
        i = ((pred_masks * gt_masks) > 0).sum(-1).sum(-1).float()
        u = ((pred_masks + gt_masks) > 0).sum(-1).sum(-1).float()
        iou = (i / u).squeeze()
        tp_count = (iou > 0.5).sum().float()
        total_iou = (iou * (iou > 0.5).float()).sum().float()
        return total_iou / tp_count, tp_count, iou.shape[0]
