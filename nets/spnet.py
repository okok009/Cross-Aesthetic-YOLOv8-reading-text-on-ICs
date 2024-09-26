'''
SPNet(Segmentation Palette Net)
Assume the input size is 120*120 (H*W), then the palette model could generate a output is 2*2 (H*W).
'''
from typing import Mapping
import torch
import torch.nn as nn
import torchvision
import numpy as np
from utils.score import ssim
from nets.unet_def import unt_rdefnet18, unt_rdefnet34, unt_rdefnet50, unt_rdefnet101, unt_rdefnet152

def palette(original_image, binarize_result, num_palette=3):
    '''
    c: color feature
    目前測試使用mode會全部取到0
    所以改成使用mean
    '''
    H = original_image.shape[-2]
    W = original_image.shape[-1]

    # m = nn.Softmax(dim=1)
    # binarize_result = m(binarize_result)
    # _, indx = torch.max(binarize_result, 1)
    # mask = binarize_result - (_.unsqueeze(1)/2)
    # mask = mask == (_.unsqueeze(1)/2)
    # binarize_result = binarize_result * mask
    # binarize_result = binarize_result / _.unsqueeze(1)

    c = original_image[:, 0].unsqueeze(1) * binarize_result
    for i in range(1, original_image.shape[1]):
        c_ = original_image[:, i].unsqueeze(1) * binarize_result
        c = torch.cat((c, c_), dim=1)
    
    c = c.reshape(c.shape[0], c.shape[1], -1)
    # print('c.shape: ', c.shape)
    # v, idx = torch.mode(c, dim=2)
    # print('v:', v)
    # v = v.unsqueeze(-1)
    v = c.mean(dim=2).unsqueeze(-1)

    if num_palette == 3:
        c_1 = torch.cat((v[:, 0], v[:, 3], v[:, 6]), dim=1)
        c_2 = torch.cat((v[:, 1], v[:, 4], v[:, 7]), dim=1)
        c_3 = torch.cat((v[:, 2], v[:, 5], v[:, 8]), dim=1)

        c_1 = binarize_result[:, 0].unsqueeze(1).expand([-1, 3, -1, -1]) * c_1.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_2 = binarize_result[:, 1].unsqueeze(1).expand([-1, 3, -1, -1]) * c_2.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_3 = binarize_result[:, 2].unsqueeze(1).expand([-1, 3, -1, -1]) * c_3.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])

        result = c_1 + c_2 + c_3

    elif num_palette == 16:
        c_1 = torch.cat((v[:, 0], v[:, 16], v[:, 32]), dim=1)
        c_2 = torch.cat((v[:, 1], v[:, 17], v[:, 33]), dim=1)
        c_3 = torch.cat((v[:, 2], v[:, 18], v[:, 34]), dim=1)
        c_4 = torch.cat((v[:, 3], v[:, 19], v[:, 35]), dim=1)
        c_5 = torch.cat((v[:, 4], v[:, 20], v[:, 36]), dim=1)
        c_6 = torch.cat((v[:, 5], v[:, 21], v[:, 37]), dim=1)
        c_7 = torch.cat((v[:, 6], v[:, 22], v[:, 38]), dim=1)
        c_8 = torch.cat((v[:, 7], v[:, 23], v[:, 39]), dim=1)
        c_9 = torch.cat((v[:, 8], v[:, 24], v[:, 40]), dim=1)
        c_10 = torch.cat((v[:, 9], v[:, 25], v[:, 41]), dim=1)
        c_11 = torch.cat((v[:, 10], v[:, 26], v[:, 42]), dim=1)
        c_12 = torch.cat((v[:, 11], v[:, 27], v[:, 43]), dim=1)
        c_13 = torch.cat((v[:, 12], v[:, 28], v[:, 44]), dim=1)
        c_14 = torch.cat((v[:, 13], v[:, 29], v[:, 45]), dim=1)
        c_15 = torch.cat((v[:, 14], v[:, 30], v[:, 46]), dim=1)
        c_16 = torch.cat((v[:, 15], v[:, 31], v[:, 47]), dim=1)

        c_1 = binarize_result[:, 0].unsqueeze(1).expand([-1, 3, -1, -1]) * c_1.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_2 = binarize_result[:, 1].unsqueeze(1).expand([-1, 3, -1, -1]) * c_2.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_3 = binarize_result[:, 2].unsqueeze(1).expand([-1, 3, -1, -1]) * c_3.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_4 = binarize_result[:, 3].unsqueeze(1).expand([-1, 3, -1, -1]) * c_4.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_5= binarize_result[:, 4].unsqueeze(1).expand([-1, 3, -1, -1]) * c_5.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_6 = binarize_result[:, 5].unsqueeze(1).expand([-1, 3, -1, -1]) * c_6.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_7 = binarize_result[:, 6].unsqueeze(1).expand([-1, 3, -1, -1]) * c_7.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_8 = binarize_result[:, 7].unsqueeze(1).expand([-1, 3, -1, -1]) * c_8.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_9 = binarize_result[:, 8].unsqueeze(1).expand([-1, 3, -1, -1]) * c_9.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_10 = binarize_result[:, 9].unsqueeze(1).expand([-1, 3, -1, -1]) * c_10.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_11 = binarize_result[:, 10].unsqueeze(1).expand([-1, 3, -1, -1]) * c_11.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_12 = binarize_result[:, 11].unsqueeze(1).expand([-1, 3, -1, -1]) * c_12.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_13 = binarize_result[:, 12].unsqueeze(1).expand([-1, 3, -1, -1]) * c_13.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_14 = binarize_result[:, 13].unsqueeze(1).expand([-1, 3, -1, -1]) * c_14.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_15 = binarize_result[:, 14].unsqueeze(1).expand([-1, 3, -1, -1]) * c_15.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])
        c_16 = binarize_result[:, 15].unsqueeze(1).expand([-1, 3, -1, -1]) * c_16.unsqueeze(-1).unsqueeze(-1).expand([-1, -1, H, W])

        result = c_1 + c_2 + c_3 + c_4 + c_5 + c_6 + c_7 + c_8 + c_9 + c_10 + c_11 + c_12 + c_13 + c_14 + c_15 + c_16

    return result

class Palette_old_version_2(nn.Module):
    def __init__(self, in_channel, num_palette=4):
        super().__init__()

        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.offset1 = nn.Conv2d(64, 2*9, 3, 1, 1)
        self.dconv1 = torchvision.ops.DeformConv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.offset2 = nn.Conv2d(16, 2*9, 3, 1, 1)
        self.dconv2 = torchvision.ops.DeformConv2d(16, 16, 3, 1, 1)
        if num_palette == 4:
            self.conv6 = nn.Conv2d(16, 3, kernel_size=3, stride=2, padding=1)
        elif num_palette == 16:
            self.conv6 = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.conv64 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)


    def forward(self, x):

        out = self.conv1(x) # [B, 64, 60, 60]
        out = self.leakyrelu(out)
        out = self.conv2(out) # [B, 12, 30, 30]
        out = self.leakyrelu(out)
        outs = self.conv64(out)
        out = out + outs
        out = self.leakyrelu(out)
        outs = self.conv64(out)
        out = out + outs
        out = self.leakyrelu(out)
        out = self.conv3(out) # [B, 24, 15, 15]
        out = self.leakyrelu(out)

        # offset = self.offset1(out)
        # out = self.dconv1(out, offset) # [B, 24, 15, 15]

        out = self.conv4(out) # [B, 12, 8, 8]
        out = self.leakyrelu(out)
        outs = self.conv32(out)
        out = out + outs
        out = self.leakyrelu(out)
        outs = self.conv32(out)
        out = out + outs
        out = self.leakyrelu(out)
        out = self.conv5(out) # [B, 6, 4, 4]
        out = self.leakyrelu(out)

        # offset = self.offset2(out)
        # out = self.dconv2(out, offset) # [B, 6, 4, 4]

        out = self.conv6(out) # [B, 3, 4, 4] or [B, 3, 2, 2]
        out = self.leakyrelu(out)

        return out
    
class SPNet(nn.Module):
    def __init__(self, in_channel=None, seg_model='unt_rdefnet152', input_size=400, num_palette=4, palette_name='new_version'):
        super().__init__()
        self.in_channel = in_channel
        self.num_palette = num_palette
        self.palette_name = palette_name
        if palette_name == 'new_version':
            model_dict = {'unt_rdefnet18': unt_rdefnet18(self.num_palette, input_size=input_size, b_out=False), 
                        'unt_rdefnet101': unt_rdefnet101(self.num_palette, input_size=input_size, b_out=False), 
                        'unt_rdefnet152': unt_rdefnet152(self.num_palette, input_size=input_size, b_out=False)}
            self.seg = model_dict[seg_model]
            self.palette = palette

        elif palette_name == 'old_version_2':
            model_dict = {'unt_rdefnet18': unt_rdefnet18(self.num_palette, input_size=input_size, b_out=True), 
                        'unt_rdefnet101': unt_rdefnet101(self.num_palette, input_size=input_size, b_out=True), 
                        'unt_rdefnet152': unt_rdefnet152(self.num_palette, input_size=input_size, b_out=True)}
            self.seg = model_dict[seg_model]
            self.palette = Palette_old_version_2(in_channel=self.in_channel, num_palette=self.num_palette)

        elif palette_name == 'old_version':
            model_dict = {'unt_rdefnet18': unt_rdefnet18(self.num_palette, input_size=input_size, b_out=True), 
                        'unt_rdefnet101': unt_rdefnet101(self.num_palette, input_size=input_size, b_out=True), 
                        'unt_rdefnet152': unt_rdefnet152(self.num_palette, input_size=input_size, b_out=True)}
            self.seg = model_dict[seg_model]
            self.palette = nn.Sequential(
                nn.Conv2d(self.in_channel, 6, kernel_size=3, stride=2, padding=1), # [B, 6, 60, 60]
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 30, 30]
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 15, 15]
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 8, 8]
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(12, 6, kernel_size=3, stride=2, padding=1), # [B, 6, 4, 4]
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
                # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # [B, 128, 2, 2]
                # nn.LeakyReLU(negative_slope=0.01, inplace=True),
                nn.Conv2d(6, 3, kernel_size=1), # [B, 3, 4, 4]
                nn.ReLU(inplace=True)
            )
        else:
            TypeError

    def process(self, x):
        output = self(x)
        ssim_loss = 1 - ssim(output, x)
        return ssim_loss
    
    def forward(self, x):
        if self.palette_name == 'new_version':
            '''
            b: binarize result
            result: generated image
            '''
            b = self.seg(x)
            m = nn.Softmax(dim=1)
            b = m(b)
            _, indx = torch.max(b, 1)
            mask = b - (_.unsqueeze(1)/2)
            mask = mask == (_.unsqueeze(1)/2)
            b = b * mask
            b = b / _.unsqueeze(1)
            result = self.palette(x, b, self.num_palette)

            return b, result
        
        elif self.palette_name == 'old_version_2' or self.palette_name == 'old_version' :
            '''
            f1: feature1 from seg backbone
            b: binarize result
            c: color result
            '''
            f1, b = self.seg(x)
            c = self.palette(f1)
            m = nn.Softmax(dim=1)
            b = m(b)
            _, indx = torch.max(b, 1)
            b = b == _.unsqueeze(1)

            return b, c
        