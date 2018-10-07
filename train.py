import model
import pan_loader
import base_config
import loss


import torch
import torch.optim as optim

from optparse import OptionParser

def main():
    # train network
    iters_per_checkpoint = 60
    for epoch in range(10000):  # loop over the dataset multiple times
        running_loss = 0.0
        loss1, loss2 = 0.0, 0.0
        for i, data in enumerate(train_loader, 0):
            batch_images, batch_gt_responses, batch_class_ids, batch_impulses, batch_fn = [
                v.cuda(non_blocking=True) for v in data
            ]
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            pred_class, pred_masks = net([batch_images, batch_impulses])
            # we are giving no weighting for classes...
            class_loss, mask_loss = loss_criterion(
                pred_class, batch_class_ids, pred_masks, batch_gt_responses)
            loss = class_loss + mask_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss1 += class_loss.item()
            loss2 += mask_loss.item()
            if i % iters_per_checkpoint == iters_per_checkpoint - 1:
                # scheduler.step((loss1+loss2)/iters_per_checkpoint)
                print("batch: ", i, "epoch: ", epoch,
                      "loss: %0.5f" % (running_loss / iters_per_checkpoint))
                print("class_loss: %0.5f \t mask_loss: %0.5f" %
                      (loss1 / iters_per_checkpoint, loss2 / iters_per_checkpoint))
                torch.save(net.state_dict(), model_dir + ("multi.pt"))
                running_loss = 0.0
                loss1 = 0.0
                loss2 = 0.0
    print('Finished Training')
if __name__ == '__main__':
    
    
    dataset =  


    main()