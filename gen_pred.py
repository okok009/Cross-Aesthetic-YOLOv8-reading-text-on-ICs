import torch
import cv2
from utils.score import ssim
from nets.unet_def import unt_rdefnet18
from torchvision.io import read_image

if __name__ == "__main__":
    
    a = read_image('D:/Datasets/ICText_cls/train/clean_img/77477.jpg').unsqueeze(0).to(dtype=torch.float) / 255
    # weight = torch.load('checkpoints/unt_rdefnet18/last.pth')
    model = unt_rdefnet18(3, 'checkpoints/unt_rdefnet18/best.pth',  input_size=120)
    model.eval()
    out = model(a)
    ssim_n = ssim(out, a)
    print('ssim: ', ssim_n)
    out = out[0].permute((1, 2, 0))
    out = out.data.cpu().numpy()
    out[:, :, ::1] = out[:, :, ::-1]
    cv2.imshow('a', out)
    cv2.waitKey(0)