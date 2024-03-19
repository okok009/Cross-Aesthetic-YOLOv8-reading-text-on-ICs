import datetime
import torch
import os
import numpy as np
import torchvision
from tqdm import tqdm
from utils.score import cls_loss_bce, seg_loss_bce, seg_loss_class, seg_miou
torchvision.disable_beta_transforms_warning()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def cls_fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, warmup, train_iter, val_iter, train_data_loader, val_data_loader, save_period=2, save_dir='checkpoints', device='cpu', best_top1=0, best_epoch=0):
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
            lr_scheduler.step()
            warmup.step()
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

def seg_fit_one_epoch(epoch, epochs, optimizer, model, lr_scheduler, train_iter, val_iter, train_data_loader, val_data_loader, loss_history, num_classes, save_period=2, save_dir='checkpoints', device= 'cpu'):
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
            lr_scheduler.step()
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
