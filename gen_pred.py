import torch
import cv2
from utils.score import ssim, cls_loss_bce
from nets.unet_def import unt_rdefnet18
from nets.classify import vgg
from torchvision.io import read_image

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    a = read_image('D:/Datasets/ICText_cls/test/clean_img/57990.jpg').unsqueeze(0).to(dtype=torch.float) / 255
    a = a.to(device=device)
    # weight = torch.load('checkpoints/unt_rdefnet18/last.pth')

    model = unt_rdefnet18(3, 'checkpoints/unt_rdefnet18/best.pth',  input_size=120)
    model = model.to(device=device)
    model.eval()

    dis_model_name = 'vgg19'
    dis_weight = f'checkpoints/{dis_model_name}/best.pth'
    dis_weight = torch.load(dis_weight)
    dis_model = vgg(dis_model_name, 2)
    dis_model.load_state_dict(dis_weight)
    dis_model = dis_model.to(device=device)
    dis_model.eval()

    out     = model(a)
    cls_out = dis_model(out)
    print('cls_onehot: ', cls_out.data)

    out_onehot = torch.tensor([[0., 1.]])
    cls_loss = cls_loss_bce(cls_out, out_onehot)
    ssim_n = ssim(out, a)

    print('cls_loss: ', cls_loss.data)
    print('ssim: ', 1 - ssim_n.data)
    
    out = out[0].permute((1, 2, 0))
    out = out.data.cpu().numpy()
    out[:, :, ::1] = out[:, :, ::-1]
    cv2.imshow('a', out)
    cv2.waitKey(0)