import torch
import torch.nn as nn
from utils.score import ssim, cls_loss_bce, l1_loss, mse
from utils.optimizer import adam, sgd


class GanModel(nn.Module):
    def __init__(self, generator, discriminator):
        super(GanModel, self).__init__()
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)
        self.gen_optimizer, self.gen_lr_scheduler, self.gen_warmup = None, None, None
        self.dis_optimizer, self.dis_lr_scheduler, self.dis_warmup = None, None, None
    
    def set_gen_optimizer(self, lr_rate, momentum = None, milestones = None, step = None, warmup_milestones = None, warm_step = None):
        self.gen_optimizer, self.gen_lr_scheduler, self.gen_warmup = sgd([p for p in self.generator.parameters() if p.requires_grad], lr_rate, momentum, milestones, step, warmup_milestones, warm_step)

    def set_dis_optimizer(self, lr_rate, momentum = None, milestones = None, step = None, warmup_milestones = None, warm_step = None):
        self.dis_optimizer, self.dis_lr_scheduler, self.dis_warmup = sgd([p for p in self.discriminator.parameters() if p.requires_grad], lr_rate, momentum, milestones, step, warmup_milestones, warm_step)
    
    def process(self, x, cls_real_onehot, cls_fake_onehot, gen_fake_onehot, gen_loss_ep, dis_loss_ep=0, train_dis = False, train_gen = False, ssim_ratio = 1):
        '''
        discriminator train:
            1. dis_real_loss: input x is a clean image
            2. dis_fake_loss: generator's output is a broken image
        '''
        output = self(x)
        cls_fake_output = self.discriminator(output.detach())
        # img_real_loss = l1_loss(cls_real_output, cls_real_onehot)
        img_fake_loss = l1_loss(cls_fake_output, cls_fake_onehot)
        # dis_loss = img_fake_loss + img_real_loss
        dis_loss = img_fake_loss
        dis_loss_ep = dis_loss_ep + float(dis_loss.detach().data.cpu().numpy())

        gen_fake_output = self.discriminator(output)
        gen_gan_loss = l1_loss(gen_fake_output, gen_fake_onehot)
        ssim_loss = 1 - ssim(output, x)
        gen_loss = gen_gan_loss + ssim_ratio * ssim_loss
        # mse_loss = mse(output, x) # aaaa
        # gen_loss = gen_gan_loss + mse_loss # aaaa
        gen_loss_ep = gen_loss_ep + float(gen_loss.detach().data.cpu().numpy())

        if train_dis:
            self.dis_optimizer.zero_grad()
            dis_loss.backward()
            self.dis_optimizer.step()
            
        if train_gen:
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()
        
        if train_dis or train_gen:
            return gen_loss, gen_loss_ep, dis_loss, dis_loss_ep

        else:
            val_loss = gen_loss_ep
            return val_loss

    def forward(self, x):
        output = self.generator(x)
        return output
    
    def dis_forward(self, x):
        output = self.discriminator(x)
        return output
