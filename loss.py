# TOTO: write loss functions
#


# Classifcation losses available: crossentropy
#

def class_loss(pred, gt):

    # Mask losses available: soft_iou, balanced_bce
    #


def soft_iou(pred, gt):
    return 0


def balanced_bce(pred, gt):
    return 0

_losses = {'soft_iou': soft_iou, 'balanced_bce': balanced_bce}


# class specific mask loss
# in this the loss of pred[idx] wrt to gt will be returned


def cs_mask_loss(pred, gt, idx, loss_name):
    return 0

# class agnostic mask loss


def ca_mask_loss(pred, gt, loss_name):
    return 0


# Easy example selection: takes tensor of losses and returns least n% of the losses


def select_easy(L, frac=1):
    sort(L)
    l = math.ceil(len(L) * frac)
    return L[:l]
