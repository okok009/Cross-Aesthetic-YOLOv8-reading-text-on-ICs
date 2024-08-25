from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import ignite.metrics as metrics
import numpy as np
from collections import OrderedDict
# from ignite.engine import Engine
# from ignite.metrics import SSIM
from utils.pytorch_ssim_master.pytorch_ssim import SSIM

# create default evaluator for doctests

def seg_loss_bce(output, target, mode='train'):
    m = nn.Softmax(dim=1)
    output = m(output)
    device = output.device

    if mode == 'val':
        target_t = torch.zeros(1, output.shape[2], output.shape[3], device=device)
        target_t = target_t == target
        for i in range(1, output.shape[1]):
            class_filter = torch.ones(1, output.shape[2], output.shape[3], device=device)*i
            class_filter = class_filter == target
            target_t = torch.cat((target_t, class_filter), 1)
        target = target_t * 1.0
        target = target.to(device=device)
        
    loss_weight = torch.ones_like(output, device=device) * 10
    loss_weight[0, 0] = loss_weight[0, 0] * (1/10)
    loss_fn = nn.BCELoss(loss_weight)
    loss = loss_fn(output, target)

    return loss

def seg_loss_class(output, target, mode='train'):
    m = nn.Softmax(dim=1)
    output = m(output)
    device = target.device
    output = output.to(device)

    if mode == 'val':
        target_t = torch.zeros(1, output.shape[2], output.shape[3], device=device)
        target_t = target_t == target
        for i in range(1, output.shape[1]):
            class_filter = torch.ones(1, output.shape[2], output.shape[3], device=device)*i
            class_filter = class_filter == target
            target_t = torch.cat((target_t, class_filter), 1)
        target = target_t * 1.0
    target = target.to(device=device)

    lambda1 = torch.tensor([1], device=device)
    lambda2 = torch.tensor([2], device=device)
    loss_tp = 0
    loss_tn = 0
    loss_fn = nn.BCELoss()
    for i in range(output.shape[1]):
        tp = target == i
        tn = target != i
        loss_tp += loss_fn(output * tp, target * tp)
        loss_tn += loss_fn(output * tn, target * tn)

    loss = lambda1 * loss_tp + lambda2 * loss_tn

    return loss

def seg_miou(output, target, total_iters=0, total_unions=0):
    num_classes = output.shape[1]
    m = nn.Softmax(dim=1)
    output = m(output)
    _, output = torch.max(output, 1)
    output = output.unsqueeze(1)
    for i in range(num_classes):
        p = target == i
        pred_p = output == i
        if p.sum() == 0:
            total_iters[i] += 1
            total_unions[i] += 1
        total_iters[i] += (pred_p * p).sum()
        total_unions[i] += (p + pred_p).sum()
    
    return total_iters, total_unions

def cls_loss_bce(output, target):
    device = output.device
    target = target.to(device)
    loss_fn = nn.BCELoss()
    loss = loss_fn(output, target)

    return loss

def ssim(output, target):
    '''
    Cite by https://github.com/Po-Hsun-Su/pytorch-ssim.git
    '''
    device = output.device
    target = target.to(device)
    metric = SSIM()
    ssim = metric(output, target)

    return ssim

def l1_loss(output, target):
    device = output.device
    target = target.to(device)
    metric = nn.L1Loss()
    l1_loss = metric(output, target)

    return l1_loss

def ahash(output, target, kernel_size=11, sigma=1.5):
    device = output.device
    target = target.to(device)

    ksize_half = (kernel_size - 1) * 0.5
    kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=device)
    gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
    gauss = (gauss / gauss.sum()).unsqueeze(dim=0)
    weight = torch.matmul(gauss.t(), gauss)
    if weight.shape[0] != output.shape[1]:
        weight = weight.expand(output.shape[1], output.shape[1], -1, -1)
    weight = weight.to(device=device, dtype=output.dtype)
    gauss_output = F.conv2d(output, weight)
    gauss_target = F.conv2d(target, weight)
    average_output = torch.sum(gauss_output, (-1, -2)) / (gauss_output.shape[-1] * gauss_output.shape[-2])
    average_target = torch.sum(gauss_target, (-1, -2)) / (gauss_target.shape[-1] * gauss_target.shape[-2])
    for i in range(gauss_output.shape[0]):
        for j in range(gauss_output.shape[1]):
            gauss_output[i, j] = gauss_output[i, j] > average_output[i, j]
            gauss_target[i, j] = gauss_target[i, j] > average_target[i, j]
    ahash = gauss_output != gauss_target
    ahash = ahash.to(dtype=torch.float)
    ahash = ahash.sum() / (gauss_target.shape[-1] * gauss_target.shape[-2])
    return(ahash)

def isw(ori_output, trans_output, sigma=0.8):
    '''
    Instance Selective Whitening(ISW)

    trans_output.shape = [B, C, H, W] => [B, C, HW]
    ori_output.shape = [B, C, H, W] => [B, C, HW]
    trans_cov[B, C, C]:  Transform feature's Covariance Matrix
    ori_cov[B, C, C]:    Original feature's Covariance Matrix
    variance[1, C, C]:      Variance matrix of trans_cov and ori_cov after instance whitening
    isw[B, C, C]
    '''
    trans_output = trans_output.reshape([trans_output.shape[0], trans_output.shape[1], -1])
    ori_output = ori_output.reshape([ori_output.shape[0], ori_output.shape[1], -1])
    trans_cov = cov_matrix(trans_output)
    ori_cov = cov_matrix(ori_output)
    trans_output = iw(trans_output, trans_cov)
    ori_output = iw(ori_output, ori_cov)
    trans_cov = cov_matrix(trans_output)
    ori_cov = cov_matrix(ori_output)
    variance = var_matrix(trans_cov, ori_cov)
    mask = variance > (variance.mean()*sigma)
    num = mask.sum()
    isw = ori_cov * mask
    isw = torch.norm(isw, p=1) / num
    return isw

def cov_matrix(feature):
    '''
    Covariance Matrix
    cov_matrix: convariance matrix
    mean: mean of each channel

    feature.shape = [B, C, HW]
    cov_matrix.shape = [B, C, C]
    '''
    mean = torch.mean(feature, -1, True)
    feature = feature - mean.expand(-1, -1, feature.shape[-1])
    cov_matrix = torch.matmul(feature, torch.transpose(feature, -2, -1)) / feature.shape[-1]
    return cov_matrix

def iw(feature, cov_matrix):
    '''
    IW(Instance Whitening):
    A feature after IW transform, that will have zero mean and that's diag(covariance matrix) will be 1.
    (This idea is from Whitening Transform which's output have a identity matrix.)

    feature.shape = [B, C, HW]
    cov_matrix.shape = [B, C, C]
    iw_feature.shape = [B, C, HW]

    tips:
        We don't need to worry what if batch size is odd, cause torch.diagonal can handle this issue.
    '''
    dia_cov = None
    for i in range(0, cov_matrix.shape[0], 2):
        if dia_cov is not None:
            dia = torch.diagonal(cov_matrix[i:i+2], dim1=1, dim2=2).unsqueeze(-1)
            dia_cov = torch.cat((dia_cov, dia))
        else:
            dia_cov = torch.diagonal(cov_matrix[i:i+2], dim1=1, dim2=2).unsqueeze(-1)
    dia_cov = torch.sqrt(dia_cov)
    mean = torch.mean(feature, -1, True)
    feature = feature - mean.expand(-1, -1, feature.shape[-1])
    iw_feature = feature / dia_cov.expand(-1, -1, feature.shape[-1])
    return iw_feature

def var_matrix(cov_1, cov_2):
    '''
    Variance Matrix
    This variance is to think all stochastic variable is a pixel, then do something like covariance matrix.
    That why the term cov_1*cov_1 and cov_2*cov_2 is element-wise multiplier.
    And variance need to calculate a mean of all sample in one batch.

    cov_1.shape, cov_2.shape = [B, C, C]
    variance.shape = [B, C, C] => [1, C, C]
    '''
    mean = (cov_1 + cov_2) / 2
    cov_1 = cov_1 - mean
    cov_2 = cov_2 - mean
    variance = (cov_1 * cov_1 + cov_2 * cov_2) / 2
    variance = torch.mean(variance, 0, True)

    return variance

def mse(output, target):
    loss = nn.MSELoss()
    mse = loss(output, target)

    return mse

def mae(output, target):
    loss = nn.L1Loss()
    mae = loss(output, target)

    return mae
