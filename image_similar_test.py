from torchvision.io import read_image
from utils.score import ssim, ahash
import torch


if __name__ == '__main__':

    preds = read_image('D:/Datasets/ICText_cls/train/clean_img/1652.jpg').unsqueeze(0).to(dtype=torch.float)
    target = read_image('D:/Datasets/ICText_cls/train/Only_broken_img/1651.jpg').unsqueeze(0).to(dtype=torch.float)
    # ssim_num = ssim(preds, target)
    # print(ssim_num)

    ahash_num = ahash(preds, target)
    print(ahash_num)