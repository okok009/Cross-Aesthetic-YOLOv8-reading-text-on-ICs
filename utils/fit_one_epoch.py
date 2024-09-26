import datetime
import torch
import os
import numpy as np
import torchvision
from tqdm import tqdm
from utils.score import cls_loss_bce, seg_loss_bce, seg_loss_class, seg_miou, ssim
from gan_pred import gan_pred
torchvision.disable_beta_transforms_warning()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cls_fit_one_epoch(epoch,
                       epochs,
                       optimizer,
                       model,
                       lr_scheduler,
                       warmup,
                       train_iter,
                       val_iter,
                       train_data_loader,
                       val_data_loader,
                       save_period=2,
                       save_dir='checkpoints',
                       device='cpu',
                       best_top1=0,
                       best_epoch=0):
    print('---------------start training---------------')
    loss_ep = 0
    model.train()
    with tqdm(total=train_iter,desc=f'Epoch {epoch}/{epochs}') as pbar:
        for img, cls_onehot in train_data_loader:
            img = img.to(device)
            cls_onehot = cls_onehot.to(device)
            output = model(img)
            loss = cls_loss_bce(output, target=cls_onehot)

            optimizer.zero_grad()
            loss.backward()
            loss_ep += float(loss.data.cpu().numpy())
            optimizer.step()

            pbar.set_postfix(**{'batch_loss'    : loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            if lr_scheduler is not None: lr_scheduler.step()
            if warmup is not None: warmup.step()
    loss_ep /= train_iter
    print('\n---------------start validate---------------')
    val_loss = 0
    model.eval()
    top1_acc = 0
    with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        with torch.no_grad():
            for img, cls_onehot in val_data_loader:
                img = img.to(device)
                cls_onehot = cls_onehot.to(device)
                output = model(img)
                loss = cls_loss_bce(output, target=cls_onehot)
                val_loss += float(loss.data.cpu().numpy())
                if (output.round() == cls_onehot).sum() == len(cls_onehot[0]):
                    top1_acc += 1
                pbar.update(1)       
    val_loss /= val_iter
    top1_acc /= val_iter

   

    if epoch == 1:
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'last.pth'))
        best_top1 = top1_acc
        best_epoch = epoch


    elif epoch % save_period == 0 or epoch == epochs:
        if top1_acc > best_top1:
            torch.save(model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
            best_top1 = top1_acc
            best_epoch = epoch
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'last.pth'))
    
    print(f'\ntrain_loss:{loss_ep} || val_loss:{val_loss} || top1_acc:{top1_acc} || best_top1_acc:{best_top1} || best_epoch:{best_epoch}\n')
   
    return best_top1, best_epoch

def seg_fit_one_epoch(epoch,
                       epochs,
                       optimizer,
                       model,
                       lr_scheduler,
                       warmup,
                       train_iter,
                       val_iter,
                       train_data_loader,
                       val_data_loader,
                       loss_history,
                       num_classes,
                       save_period=2,
                       save_dir='checkpoints',
                       device= 'cpu'):
    print('---------------start training---------------')
    loss_ep = 0
    model.train()
    with tqdm(total=train_iter,desc=f'Epoch {epoch}/{epochs}') as pbar:
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            output = model(img)
            loss = seg_loss_bce(output=output, target=label, mode='train')

            optimizer.zero_grad()
            loss.backward()
            loss_ep += float(loss.data.cpu().numpy())
            optimizer.step()

            pbar.set_postfix(**{'batch_loss'    : loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            if lr_scheduler is not None: lr_scheduler.step()
            if warmup is not None: warmup.step()
    loss_ep = loss_ep / train_iter
    print('---------------start validate---------------')
    val_loss = 0
    total_iters, total_unions, mious = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    model.eval()
    with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        with torch.no_grad():
            for img, label in val_data_loader:
                img = img.to(device)
                label = label.to(device)

                output = model(img)
                loss = seg_loss_bce(output=output, target=label, mode='val')
                val_loss += loss.data.cpu().numpy()
                
                total_iters, total_unions = seg_miou(output=output, target=label, total_iters=total_iters, total_unions=total_unions)

                pbar.update(1)

    val_loss = val_loss / val_iter
    for i in range(num_classes):
        mious[i] = 1.0*total_iters[i]/(total_unions[i]+2.220446049250313e-16)
    mious = format(mious.mean(), '.6f')

    print(f'\ntrain_loss:{loss_ep} || val_loss:{val_loss}, val_miou:{mious}\n')
    loss_history.append_loss(epoch + 1, loss=loss_ep, val_loss=val_loss)

    if epoch % save_period == 0 or epoch == epochs:
        torch.save(model.state_dict(), os.path.join('E:/ray_workspace/defnet', save_dir, f'ep{epoch}-val_loss{val_loss}-miou{mious}.pth'))

def gen_fit_one_epoch(epoch,
                       epochs,
                       optimizer,
                       gen_model,
                       dis_model,
                       lr_scheduler,
                       warmup,
                       train_iter,
                       val_iter,
                       train_data_loader,
                       val_data_loader,
                       save_period=2,
                       save_dir='checkpoints',
                       device='cpu',
                       best_ssim=0,
                       best_epoch=0):
    '''
    gen_cls_onehot = [0, 1] ---> input is clean imgs and want to generate Only_broken imgs
    '''
    print('---------------start training---------------')
    loss_ep = 0
    gen_model.train()
    dis_model.train()
    with tqdm(total=train_iter, desc=f'Epoch {epoch}/{epochs}') as pbar:
        for img in train_data_loader:
            img = img.to(device)
            gen_cls_onehot = torch.zeros([img.shape[0], 2]) 
            gen_cls_onehot[:, 1] = 1 # if img_dir is Only_broken_img then onehot should be [1, 0]
            gen_cls_onehot = gen_cls_onehot.to(device)
            output = gen_model(img)
            print('output: ', output)
            ssim_loss = 1 - ssim(output, img)
            if epoch > 0:
                cls_output = dis_model(output)
                print('cls_output: ', cls_output)
                img_loss = cls_loss_bce(cls_output, target=gen_cls_onehot)
                # loss = 0.02 * ssim_loss + 0.08 * img_loss
                loss = img_loss
            else:
                loss = ssim_loss

            optimizer.zero_grad()
            loss.backward()
            loss_ep += float(loss.data.cpu().numpy())
            optimizer.step()

            pbar.set_postfix(**{'batch_loss'    : loss, 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            if lr_scheduler is not None: lr_scheduler.step()
            if warmup is not None: warmup.step()
    loss_ep /= train_iter
    print('\n---------------start validate---------------')
    val_loss = 0
    gen_model.eval()
    with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
        with torch.no_grad():
            for img in val_data_loader:
                img = img.to(device)
                gen_cls_onehot = torch.zeros([img.shape[0], 2]) 
                gen_cls_onehot[:, 1] = 1 # if img_dir is Only_broken_img then onehot should be [1, 0]
                gen_cls_onehot = gen_cls_onehot.to(device)
                output = gen_model(img)
                ssim_loss = 1 - ssim(output, img)
                loss = ssim_loss
                if epoch > 0:
                    cls_output = dis_model(output)
                    img_loss = cls_loss_bce(cls_output, target=gen_cls_onehot)
                    loss = 0.2 * loss + 0.8 * img_loss
                val_loss += float(loss.data.cpu().numpy())
                pbar.update(1)       
    val_loss /= val_iter

   

    if epoch == 1:
        torch.save(gen_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
        torch.save(gen_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'last.pth'))
        best_ssim = val_loss
        best_epoch = epoch


    elif epoch % save_period == 0 or epoch == epochs:
        if val_loss < best_ssim:
            torch.save(gen_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
            best_ssim = val_loss
            best_epoch = epoch
        torch.save(gen_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'last.pth'))
    
    print(f'\ntrain_loss:{loss_ep} || val_loss:{val_loss} || best_val_ssim:{best_ssim} || best_epoch:{best_epoch}\n')
    if epoch == 400:
        print(f'Epoch {epoch + 1} is coming. The loss will become mix loss (ssim and cls).')
    return best_ssim, best_epoch

def gen_fit_one_epoch_1(epoch,
                       epochs,
                       gan_model,
                       train_iter,
                       val_iter,
                       train_data_loader,
                       val_data_loader,
                       save_period=2,
                       save_dir='checkpoints',
                       device='cpu',
                       best_val=0,
                       best_epoch=0,
                       gen_loss_ep = 0,
                       dis_loss_ep = 0,
                       change_list = [50, 55, 95, 100, 155, 195],
                       gen_model_name = 'spnet_new_version'):
    '''
    gen_cls_onehot = [0, 1] ---> input is clean imgs and want to generate Only_broken imgs
    '''
    print('---------------start training---------------')
    if gen_loss_ep > 0.5:
        train_gen = True
        train_dis = False
    elif dis_loss_ep > 0.8:
        train_gen = False
        train_dis = True
    else:
        train_gen = True
        train_dis = False
    gen_loss_ep = 0
    dis_loss_ep = 0
    gan_model.train()
    with tqdm(total=train_iter, desc=f'Epoch {epoch}/{epochs}') as pbar:
        for img in train_data_loader:
            img = img.to(device)

            '''
            cls_real_onehot: clean_image [1, 0].
            cls_fake_onehot: generate broken_image [1, 0].
            gen_fake_onehot: generate broken_image [0, 1].
            '''
            cls_real_onehot, cls_fake_onehot, gen_fake_onehot = torch.zeros([img.shape[0], 2]), torch.zeros([img.shape[0], 2]), torch.zeros([img.shape[0], 2])
            cls_real_onehot[:, 0], cls_fake_onehot[:, 0], gen_fake_onehot[:, 1] = 1, 1, 1
            cls_real_onehot, cls_fake_onehot, gen_fake_onehot = cls_real_onehot.to(device), cls_fake_onehot.to(device), gen_fake_onehot.to(device)

            if epoch < change_list[0] or change_list[1] <= epoch < change_list[2] or change_list[3] <= epoch < change_list[4]:
                train_dis = False
                train_gen= True
                gen_loss, gen_loss_ep, dis_loss, dis_loss_ep = gan_model.process(img, cls_real_onehot, cls_fake_onehot, gen_fake_onehot, gen_loss_ep, dis_loss_ep, train_dis, train_gen)
            
            elif change_list[0] <= epoch < change_list[1] or change_list[2] <= epoch < change_list[3] or change_list[4] <= epoch < change_list[5]:
                train_dis = True
                train_gen= False
                gen_loss, gen_loss_ep, dis_loss, dis_loss_ep = gan_model.process(img, cls_real_onehot, cls_fake_onehot, gen_fake_onehot, gen_loss_ep, dis_loss_ep, train_dis, train_gen)

            else:
                train_dis = True
                train_gen= True
                gen_loss, gen_loss_ep, dis_loss, dis_loss_ep = gan_model.process(img, cls_real_onehot, cls_fake_onehot, gen_fake_onehot, gen_loss_ep, dis_loss_ep, train_dis, train_gen)

            pbar.set_postfix(**{'gen_batch_loss'    : gen_loss.data.cpu().numpy(), 
                                'gen_lr'            : get_lr(gan_model.gen_optimizer),
                                'dis_batch_loss'    : dis_loss.data.cpu().numpy(), 
                                'dis_lr'            : get_lr(gan_model.dis_optimizer)})
            pbar.update(1)

            del gen_loss, dis_loss
            if train_gen and gan_model.gen_lr_scheduler is not None: gan_model.gen_lr_scheduler.step()
            if train_gen and gan_model.gen_warmup is not None: gan_model.gen_warmup.step()
            if train_dis and gan_model.dis_lr_scheduler is not None: gan_model.dis_lr_scheduler.step()
            if train_dis and gan_model.dis_warmup is not None: gan_model.dis_warmup.step()

    gen_loss_ep /= train_iter
    dis_loss_ep /= train_iter
    # print('\n---------------start validate---------------')
    # val_loss = 0
    # gan_model.eval()
    # with tqdm(total=val_iter,desc=f'Epoch {epoch}/{epochs}',postfix=dict) as pbar:
    #     with torch.no_grad():
    #         for img in val_data_loader:
    #             img = img.to(device)
    #             gen_cls_onehot = torch.zeros([img.shape[0], 2]) 
    #             gen_cls_onehot[:, 1] = 1 # if img_dir is Only_broken_img then onehot should be [1, 0]
    #             gen_cls_onehot = gen_cls_onehot.to(device)
    #             val_loss = gan_model.process(img, cls_real_onehot, cls_fake_onehot, gen_fake_onehot, val_loss, train_gen=False, train_dis=False)
    #             pbar.update(1)       
    # val_loss /= val_iter
   
    torch.save(gan_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'last.pth'))
    # if epoch == 1:
    #     torch.save(gan_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
    #     best_val = val_loss
    #     best_epoch = epoch

    # elif val_loss < best_val:
    #     torch.save(gan_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'best.pth'))
    #     best_val = val_loss
    #     best_epoch = epoch

    img_list = ['56746_1', '57990_1', '88100_1']
    if epoch%save_period == 0 and epoch != 9:
        torch.save(gan_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'ep{epoch}.pth'))
        '''just for sample'''
        gan_pred(model_=gan_model, image_id_list=img_list, write=True, weight=f'ep{epoch}.pth', gen_model_name=gen_model_name)
    else:
        for i in range(len(change_list)):
            if epoch == change_list[i]-1 or epoch == change_list[i]:
                torch.save(gan_model.state_dict(), os.path.join('E:/ray_workspace/CrossAestheticYOLOv8/', save_dir, f'ep{epoch}.pth'))
                '''just for sample'''
                gan_pred(model_=gan_model, image_id_list=img_list, write=True, weight=f'ep{epoch}.pth', gen_model_name=gen_model_name)
            
    '''just for sample'''
    gan_pred(model_=gan_model, image_id_list=img_list, write=True, weight='last.pth', gen_model_name=gen_model_name)

    # print(f'\ntrain_gen_loss:{gen_loss_ep} || train_dis_loss:{dis_loss_ep} || val_loss:{val_loss} || best_val_loss:{best_val} || best_epoch:{best_epoch}\n')

    return best_val, best_epoch, gen_loss_ep, dis_loss_ep