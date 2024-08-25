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

class Palette(nn.Module):
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
    def __init__(self, in_channel, seg_model='unt_rdefnet152', input_size=400, num_palette=4):
        super().__init__()
        self.in_channel = in_channel
        self.num_palette = num_palette
        model_dict = {'unt_rdefnet18': unt_rdefnet18(self.num_palette, input_size=input_size, b_out=True), 
                      'unt_rdefnet101': unt_rdefnet101(self.num_palette, input_size=input_size, b_out=True), 
                      'unt_rdefnet152': unt_rdefnet152(self.num_palette, input_size=input_size, b_out=True)}
        self.seg = model_dict[seg_model]
        '''new_version'''
        self.palette = Palette(in_channel=self.in_channel, num_palette=self.num_palette)
        '''old_version'''
        # self.palette = nn.Sequential(
        #     nn.Conv2d(self.in_channel, 6, kernel_size=3, stride=2, padding=1), # [B, 6, 60, 60]
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 30, 30]
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 15, 15]
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Conv2d(12, 12, kernel_size=3, stride=2, padding=1), # [B, 12, 8, 8]
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Conv2d(12, 6, kernel_size=3, stride=2, padding=1), # [B, 6, 4, 4]
        #     nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # [B, 128, 2, 2]
        #     # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        #     nn.Conv2d(6, 3, kernel_size=1), # [B, 3, 4, 4]
        #     nn.ReLU(inplace=True)
        # )

    def process(self, x):
        output = self(x)
        ssim_loss = 1 - ssim(output, x)
        return ssim_loss
    
    def forward(self, x):
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
        b = b==_.unsqueeze(1)

        return b, c