import torch
import cv2
from utils.score import ssim, cls_loss_bce
from nets.unet_def import unt_rdefnet18
from nets.gan import GanModel
from nets.classify import vgg
from torchvision.io import read_image

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    a = read_image('D:/Datasets/ICText_cls/test/clean_img/389836.jpg').unsqueeze(0).to(dtype=torch.float) / 255
    a = a.to(device=device)
    # weight = torch.load('checkpoints/unt_rdefnet18/last.pth')

    n_classes = 2

    gen_model_name = 'unt_rdefnet18'
    gen_model = unt_rdefnet18(3, input_size=120)
    gen_model = gen_model.to(device=device)

    dis_model_name = 'vgg19'
    dis_model = vgg(dis_model_name, n_classes, freezing=True)
    dis_model = dis_model.to(device=device)

    gen_model_name = 'GanModel'
    weight = f'checkpoints/{gen_model_name}/best.pth'
    weight = torch.load(weight)
    model = GanModel(gen_model, dis_model)
    model.load_state_dict(weight)
    model = model.to(device=device)
    model.eval()

    out     = model(a)
    
    out = out[0].permute((1, 2, 0))
    out = out.data.cpu().numpy()
    out[:, :, ::1] = out[:, :, ::-1]
    cv2.imshow('a', out)
    cv2.waitKey(0)