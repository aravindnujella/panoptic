import torch
import numpy as np

def cudify_data(d):
    n = d[0].shape[0]
    l = np.arange(n)
    np.random.shuffle(l)
    idx = l[:min(n, 16)]
    return [it[idx].float().cuda() for it in d]

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
        self.running_loss = None
        # self.val_fn = val_fn

    # loss must be cpu torch tensor
    def update(self, loss, net):
        if self.running_loss is None:
            self.running_loss = torch.zeros_like(loss)

        self.running_loss += loss
        self.step += 1

        if self.step % self.iters_per_epoch == 0:
            self.display_loss()
            torch.save(net.state_dict(), "%s%s_%d.pt" % (self.model_dir, self.model_name, self.step - self.step))
            self.running_loss = None

    def display_loss(self):
        print("Step: %d\t%0.5f" % (self.step, self.running_loss.data / self.iters_per_epoch))
