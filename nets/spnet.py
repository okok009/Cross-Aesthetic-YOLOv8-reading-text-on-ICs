'''
SPNet(Segmentation Palette Net)
Assume the input size is 120*120 (H*W), then the palette model could generate a output is 2*2 (H*W).
'''
import torch
import torch.nn as nn
from utils.score import ssim
from nets.unet_def import unt_rdefnet18, unt_rdefnet34, unt_rdefnet50, unt_rdefnet101, unt_rdefnet152

class SPNet(nn.Module):
    def __init__(self, in_channel, seg_model='unt_rdefnet152', input_size=400):
        super().__init__()
        self.in_channel = in_channel
        self.num_palette = 16
        model_dict = {'unt_rdefnet18': unt_rdefnet18(self.num_palette, input_size=input_size), 
                      'unt_rdefnet101': unt_rdefnet101(self.num_palette, input_size=input_size), 
                      'unt_rdefnet152': unt_rdefnet152(self.num_palette, input_size=input_size)}
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

    def process(self, x):
        output = self(x)
        ssim_loss = 1 - ssim(output, x)
        return ssim_loss
    
    def forward(self, x):
        '''
        b: binarize result
        c: color result
        c_1: color 1 result
        c_2: color 2 result
        '''
        b = self.seg(x)
        c = self.palette(x)
        m = nn.Softmax(dim=1)
        b = m(b)
        _, indx = torch.max(b, 1)
        b = b==_.unsqueeze(1)
        c_1 = b[:, 0].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_2 = b[:, 1].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_3 = b[:, 2].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 2].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_4 = b[:, 3].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 3].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_5 = b[:, 4].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_6 = b[:, 5].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_7 = b[:, 6].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 2].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_8 = b[:, 7].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 3].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_9 = b[:, 8].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 2, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_10 = b[:, 9].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 2, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_11 = b[:, 10].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 2, 2].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_12 = b[:, 11].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 2, 3].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_13 = b[:, 12].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 3, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_14 = b[:, 13].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 3, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_15 = b[:, 14].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 3, 2].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        c_16 = b[:, 15].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 3, 3].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
        result = c_1 + c_2 + c_3 + c_4 + c_5 + c_6 + c_7 + c_8 + c_9 + c_10 + c_11 + c_12 + c_13 + c_14 + c_15 + c_16
        return result