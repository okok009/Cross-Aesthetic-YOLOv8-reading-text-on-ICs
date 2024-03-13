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
from data.dataset import gen_dataloader
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
    model_name = 'vgg16'
    model = vgg(model_name, n_classes, freezing = True)

    model = model.to(device=device)

    # -----------------------------------
    # optimizer
    # -----------------------------------
    lr_rate = 0.01
    milestones = [5500, 11000, 22000]
    warmup_milestones = [100, 300, 600]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr_rate, momentum=0.2)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, 0.1)
    warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer, warmup_milestones, 2)

    # -----------------------------------
    # data_loader
    # -----------------------------------
    img_txt = 'E:/ray_workspace/CrossAestheticYOLOv8/data/broken_clean_img.txt'
    img_path = 'E:/Datasets/ICText/train2021/'
    json_path = 'E:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
    batch_size = 20
    shuffle = True
    epochs = 20
    with open(img_txt) as f:
        img_ids = f.readlines()
    train_iter = len(img_ids)//batch_size
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )

    train_data_loader = gen_dataloader(img_txt, img_path, json_path, transform, batch_size, shuffle)

    val_iter = None
    val_data_loader = None

    # -----------------------------------
    # Log
    # -----------------------------------
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H.%M')
    log_dir         = os.path.join('logs', f"{model}_", "loss_" + str(time_str))

    # -----------------------------------
    # fit one epoch (train & validation)
    # -----------------------------------
    for epoch in range(1, epochs+1):
        cls_fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, warmup, train_iter, val_iter, train_data_loader, val_data_loader, n_classes+1, save_period=1, device=device)
