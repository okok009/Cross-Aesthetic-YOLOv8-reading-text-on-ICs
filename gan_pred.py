import torch
import cv2
import os
from utils.score import ssim, cls_loss_bce
from nets.unet_def import unt_rdefnet18, unt_rdefnet152
from nets.gan import GanModel
from nets.classify import vgg
from torchvision.io import read_image

def gan_pred():
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    image_id = '81058_1'
    a = read_image(f'D:/Datasets/ICText_cls/train/Not_broken_img/{image_id}.jpg').unsqueeze(0).to(dtype=torch.float) / 255
    a = a.to(device=device)
    # weight = torch.load('checkpoints/unt_rdefnet18/last.pth')

    n_classes = 2

    gen_model_name = 'unt_rdefnet152'
    gen_model = unt_rdefnet152(3, input_size=120)
    gen_model = gen_model.to(device=device)

    dis_model_name = 'vgg19'
    dis_model = vgg(dis_model_name, n_classes, freezing=True)
    dis_model = dis_model.to(device=device)

    gan_model_name = 'GanModel'
    weight_ep = os.listdir(f'checkpoints/{gan_model_name}/{gen_model_name}/')
    weight_ep.remove('old_version')
    weight_ep.remove('best.pth')
    weight_ep.remove('last.pth')
    weight_ep.sort(key = lambda x:int(x[2:-4]))
    # weight_ep = ['best.pth', 'last.pth']
    print(weight_ep)

    for w in weight_ep:
        w = w[:-4]
        weight = f'checkpoints/{gan_model_name}/{gen_model_name}/{w}.pth'
        weight = f'checkpoints/{gan_model_name}/{gen_model_name}/old_version/fulldata/best.pth'
        weight = torch.load(weight)
        model = GanModel(gen_model, dis_model)
        model.load_state_dict(weight)
        model = model.to(device=device)
        model.eval()

        out     = model(a)
        dis_out = model.dis_forward(out)

        ssim_loss = ssim(out, a)
        print(f'{w}_ssim: ', ssim_loss.data.cpu())

        print(f'{w}_dis: ', dis_out.data.cpu())
        out = out[0].permute((1, 2, 0))
        out = out.data.cpu().numpy()
        cv2.imwrite(f'checkdata/{image_id}_{w}.jpg', out*255)
        out[:, :, ::1] = out[:, :, ::-1]
        cv2.imshow('a', out)
        cv2.waitKey(0)

if __name__ == "__main__":
    
    gan_pred()