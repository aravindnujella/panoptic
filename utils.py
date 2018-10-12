import torch
import numpy as np


# takes list of tensors xD
def cudify_list(l):
    return [torch.from_numpy(it).cuda() for it in l]

def cudify_data(d):
    return [cudify_list(it) for it in d]
    # images, impulses, instance_masks, cat_ids = d
    # return cudify_list(images), 

# rudimentary checkpointing
# TODO: add load checkpoint??, add validation stats at the end of checkpoint
# 
class Checkpoint:

    def __init__(self, iters_per_epoch, model_dir, model_name):
        self.iters_per_epoch = iters_per_epoch
        self.model_dir = model_dir
        self.model_name = model_name

        self.step = 0
        self.running_loss = None
        # self.val_fn = val_fn

    # loss must be cpu torch tensor
    def update(self, loss, net):
        if self.running_loss is None:
            self.running_loss = torch.zeros_like(loss)

        self.running_loss += loss
        self.step += 1

        if step % iters_per_epoch == 0:
            self.display_loss()
            torch.save(net.state_dict(), "%s%s_%d.pt" % (self.model_dir, self.model_name, self.step))
            self.running_loss = None

    def display_loss(self):
        l = self.running_loss.shape[0]
        out = ""
        for i in range(l):
            out += "Loss%d\t%0.5f" % (i, self.running_loss / self.step)
        print(out)


