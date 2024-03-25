import torch
import torch.nn as nn
from utils.score import ssim, cls_loss_bce, l1_loss


class GanModel(nn.Module):
    def __init__(self, generator, discriminator):
        super(GanModel, self).__init__()
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
    
    def process(self, x, gen_cls_onehot, loss_ep, optimizer = None, train = True):
        output = self(x)
        cls_output = self.discriminator(output)
        ssim_loss = 1 - ssim(output, x)
        img_loss = l1_loss(cls_output, gen_cls_onehot)
        loss = img_loss + ssim_loss
        loss_ep += float(loss.data.cpu().numpy())

        if train:
            optimizer.zero_grad()
            loss.backward()
            loss_ep += float(loss.data.cpu().numpy())
            optimizer.step()
            return loss, loss_ep, optimizer
        
        return loss, loss_ep

    def forward(self, x):
        output = self.generator(x)
        return output