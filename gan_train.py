import torch
import torch.nn as nn
import torchvision
import cv2
import os
import numpy as np
import datetime
from nets.classify import vgg
from nets.unet_def import unt_rdefnet18, unt_rdefnet152, unt_rdefnet101
from nets.spnet import SPNet
from nets.gan import GanModel
from tqdm import tqdm
from utils.callbacks import LossHistory
from utils.fit_one_epoch import cls_fit_one_epoch, gen_fit_one_epoch, gen_fit_one_epoch_1
from utils.optimizer import adam, sgd
from data.dataset import cls_dataloader, gen_dataloader
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
    # list of which epoch should change training model (gen or dis)
    # train gen when epoch < change_list[0], train dis when change_list[0] < epoch < change_list[1], and so on.
    # -----------------------------------
    # change_list = [50, 55, 100, 105, 150, 155]
    change_list = [200, 205, 250, 255, 300, 305]

    # -----------------------------------
    # model
    # 記得換生成器時要去gan.py修改forward
    # -----------------------------------
    n_classes = 2

    # gen_model_name = 'unt_rdefnet152'
    # gen_weight = None
    # gen_model = unt_rdefnet152(3, gen_weight, input_size=120)
    # gen_model = gen_model.to(device=device)
    gen_model_name = 'spnet'
    seg_model = 'unt_rdefnet152'
    gen_model = SPNet(64, input_size=120, num_palette=16)
    gen_model = gen_model.to(device=device)

    dis_model_name = 'vgg19'
    dis_weight = f'checkpoints/{dis_model_name}/best.pth'
    dis_weight = torch.load(dis_weight)
    dis_model = vgg(dis_model_name, n_classes, freezing=True)
    dis_model.load_state_dict(dis_weight)
    dis_model = dis_model.to(device=device)

    gan_model_name = 'GanModel'
    weight = f'checkpoints/{gan_model_name}/{gen_model_name}/last.pth'
    weight = torch.load(weight)
    model = GanModel(gen_model, dis_model)
    model_dict = model.state_dict()
    # no_load_key, temp_dict = [], {}
    # for k, v in weight.items():

    #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
    #         temp_dict[k] = v
    #     else:
    #         no_load_key.append(k)

    # print('no_load_key:\n', no_load_key)
    # model_dict.update(temp_dict)
    model.load_state_dict(weight)
    for param in model.generator.seg.parameters():
        param.requires_grad = False

    # -----------------------------------
    # optimizer
    # -----------------------------------
    lr_rate = 0.01
    milestones = [25000, 45000, 80000]
    warmup_milestones = [2000, 4000, 6000]
    momentum = 0.9
    step = 0.1
    warm_step = 4
    model.set_gen_optimizer(lr_rate, momentum, milestones, step, warmup_milestones, warm_step)

    lr_rate = 0.001
    milestones = [15000, 20000, 45000]
    warmup_milestones = [3000, 6000]
    momentum = 0.9
    step = 0.1
    warm_step = 2
    model.set_dis_optimizer(lr_rate, momentum, milestones, step, warmup_milestones, warm_step)

    # -----------------------------------
    # data_loader
    # -----------------------------------
    img_dir = 'D:/Datasets/ICText_cls/train/clean_img/'
    batch_size = 40
    shuffle = True
    epochs = 200
    transform = v2.Compose([
        v2.ToDtype(torch.float32)]
    )
    train_data_loader = gen_dataloader(img_dir, transform, batch_size=batch_size, shuffle=True, num_workers=2)
    train_iter = len(train_data_loader.dataset)//batch_size

    img_dir = 'D:/Datasets/ICText_cls/test/clean_img'
    batch_size = 1
    transform = v2.Compose([
        v2.ToDtype(torch.float32)]
    )
    val_data_loader = gen_dataloader(img_dir, transform, batch_size=batch_size, shuffle=False, num_workers=1, device=device)
    val_iter = len(val_data_loader.dataset)//batch_size

    # -----------------------------------
    # Log
    # -----------------------------------
    time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H.%M')
    log_dir         = os.path.join('logs', f"{gen_model}_", "loss_" + str(time_str))

    # -----------------------------------
    # fit one epoch (train & validation)
    # -----------------------------------
    best_epoch = 1
    best_val = 0
    gen_loss_ep, dis_loss_ep = 0, 0
    for epoch in range(1, epochs+1):
        best_val, best_epoch, gen_loss_ep, dis_loss_ep = gen_fit_one_epoch_1(epoch,
                                        epochs,
                                        model,
                                        train_iter,
                                        val_iter,
                                        train_data_loader,
                                        val_data_loader,
                                        save_period=4,
                                        save_dir='checkpoints/'+gan_model_name+'/'+gen_model_name,
                                        device=device,
                                        best_epoch=best_epoch,
                                        best_val=best_val,
                                        gen_loss_ep=gen_loss_ep,
                                        dis_loss_ep=dis_loss_ep,
                                        change_list=change_list)
# for epoch in range(1, epochs+1):
#         best_ssim, best_epoch = gen_fit_one_epoch(epoch,
#                                         epochs,
#                                         optimizer,
#                                         gen_model,
#                                         dis_model,
#                                         lr_scheduler,
#                                         warmup,
#                                         train_iter,
#                                         val_iter,
#                                         train_data_loader,
#                                         val_data_loader,
#                                         save_period=1,
#                                         save_dir='checkpoints/'+gen_model_name,
#                                         device=device,
#                                         best_epoch=best_epoch,
#                                         best_ssim=best_ssim)