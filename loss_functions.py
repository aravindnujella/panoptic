import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_criterion(pred, gt):
    mask_logits, cat_scores = pred

    gt_masks, gt_labels = gt
    gt_masks.unsqueeze_(1)
    print(gt_masks.type())
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


# Mask losses available: soft_iou, balanced_bce
# input pred_masks and gt_masks of same shape
# pred_masks are (pre-sigmoid) logit scores .i.e,
# both must be of same shape: (B, C, w, h)
# output loss shape: (B,)


def soft_iou(mask_logits, gt_masks):
    pred_masks = mask_logits.sigmoid()
    i = (pred_masks * gt_masks).sum(-1).sum(-1)
    u = (pred_masks + gt_masks - pred_masks * gt_masks).sum(-1).sum(-1)
    l = 1 - i / u
    # print(l)
    return l.mean()


def balanced_bce(mask_logits, gt_masks):
    fg_size = gt_masks.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
    bg_size = (1 - gt_masks).sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)

    _loss = nn.BCEWithLogitsLoss(reduction='none')

    # loss where gt is +ve, -ve
    fg_loss = _loss(mask_logits * gt_masks, gt_masks)
    bg_loss = _loss(mask_logits * (1 - gt_masks), gt_masks)

    l = (bg_size * fg_loss + fg_size * bg_loss) / (fg_size + bg_size)
    return l.mean()


# Easy example selection: takes tensor of losses and returns least n% of the losses


def select_easy(l, frac=1):
    return l
    # sort(L)
    # l = math.ceil(len(L) * frac)
    # return L[:l]


# accuracy metrics
# 
def mean_iou(mask_logits, gt_masks, cutoff=0.5):
    with torch.no_grad():
        gt_masks = gt_masks.float()
        pred_masks = mask_logits.sigmoid()
        # pred_masks = F.threshold(pred_masks, 0.5, 0)
        pred_masks = (pred_masks>0.5).float()
        i = ((pred_masks*gt_masks)>0).sum(-1).sum(-1).float()
        u = ((pred_masks + gt_masks)>0).sum(-1).sum(-1).float()

        iou = (i/u).squeeze()
        return torch.stack([iou.mean(), i.mean(), u.mean()])
