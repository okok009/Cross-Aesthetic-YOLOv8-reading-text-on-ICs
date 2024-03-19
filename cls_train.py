import torch
import torch.nn as nn
import torchvision
import cv2
import os
import numpy as np
import datetime
from nets.classify import vgg
from tqdm import tqdm
from utils.callbacks import LossHistory
from utils.fit_one_epoch import cls_fit_one_epoch
from data.dataset import cls_dataloader
from torchvision.transforms import v2
torchvision.disable_beta_transforms_warning()

# -----------------------------------
# device
# -----------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

if __name__=='__main__':
    # -----------------------------------
    # model
    # -----------------------------------
    n_classes = 2
    model_name = 'vgg19'
    model = vgg(model_name, n_classes)
    model = model.to(device=device)

    # -----------------------------------
    # optimizer
    # -----------------------------------
    lr_rate = 0.001
    milestones = [15000, 20000, 45000]
    warmup_milestones = [3000, 6000, 9000]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=0.2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
    warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, 2)

    # -----------------------------------
    # data_loader
    # -----------------------------------
    cls_dir = 'D:/Datasets/ICText_cls/train/'
    batch_size = 20
    shuffle = True
    epochs = 400
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    train_data_loader = cls_dataloader(cls_dir, transform, batch_size=batch_size, shuffle=True, num_workers=2)
    train_iter = len(train_data_loader.dataset)//batch_size

    cls_dir = 'D:/Datasets/ICText_cls/test/'
    batch_size = 1
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    val_data_loader = cls_dataloader(cls_dir, transform, batch_size=batch_size, shuffle=False, num_workers=1, device=device)
    val_iter = len(val_data_loader.dataset)//batch_size

    # -----------------------------------
    # Log
    # -----------------------------------
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H.%M')
    log_dir         = os.path.join('logs', f"{model}_", "loss_" + str(time_str))

    # -----------------------------------
    # fit one epoch (train & validation)
    # -----------------------------------
    best_top1 = 0
    best_epoch = 1
    for epoch in range(1, epochs+1):
        best_top1, best_epoch = cls_fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, warmup, train_iter, val_iter, train_data_loader, val_data_loader, save_period=1, save_dir='checkpoints/'+model_name, device=device, best_top1=best_top1, best_epoch=best_epoch )
