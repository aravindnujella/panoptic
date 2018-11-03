import torch
import torch.nn as nn
import numpy as np


def cudify_data(d):
    # n = d[0].shape[0]
    # l = np.arange(n)
    # np.random.shuffle(l)
    # idx = l[:min(n, 8)]
    return [it.float().cuda() for it in d]

MEAN_PIXEL = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, -1)
STD_PIXEL = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, -1)

# single image(wh3) to input format required by pretrained models


def imgToInp(img):
    inp = img.copy() / 255
    inp -= MEAN_PIXEL
    inp /= STD_PIXEL
    inp = np.moveaxis(inp, 2, 0)
    return inp

# single input(3wh) for pretrained models to RGB image for PIL.Image


def inpToImg(inp):
    img = np.moveaxis(inp.copy(), 0, 2)
    img *= STD_PIXEL
    img += MEAN_PIXEL
    img *= 255
    return img.astype(np.uint8)

# TODO: add load checkpoint??, add validation stats at the end of checkpoint
#


class Checkpoint:

    def __init__(self, iters_per_epoch, model_dir, model_name):
        self.iters_per_epoch = iters_per_epoch
        self.model_dir = model_dir
        self.model_name = model_name

        self.step = 0
        self.run = {}
        # self.val_fn = val_fn

    # metrics must be a dict of form {'name': torch.Tensor()}
    def update(self, metrics, net):
        if not any(self.run):
            self.reset_run(metrics)

        for key in metrics.keys():
            self.run[key] += metrics[key]

        self.step += 1

        if self.step % self.iters_per_epoch == 0:
            state_dict = None
            if isinstance(net, nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save(state_dict, "%s%s_%d.pt" % (self.model_dir, self.model_name, self.step - self.step))

            self.display_loss()
            self.reset_run(metrics)

    def reset_run(self, metrics):
        for key in metrics.keys():
            self.run[key] = torch.zeros_like(metrics[key])

    def display_loss(self):
        torch.set_printoptions(precision=4)
        print("\nStep: %d" % (self.step))

        for key in self.run.keys():
            print("%s\t" % key, self.run[key] / self.iters_per_epoch)
