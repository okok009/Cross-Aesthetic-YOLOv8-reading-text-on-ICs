import torch
import cv2
import os
import numpy as np
from utils.score import ssim, cls_loss_bce
from nets.unet_def import unt_rdefnet18, unt_rdefnet152
from nets.spnet import SPNet
from nets.gan import GanModel
from nets.classify import vgg
from torchvision.io import read_image

def gan_pred(image_id_list=None, weight=None):
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if image_id_list is None:
        image_id_list = ['88100_1']

    for image_id in image_id_list:
        a = read_image(f'D:/Datasets/ICText_cls/test/clean_img/{image_id}.jpg').unsqueeze(0).to(dtype=torch.float) / 255
        a = a.to(device=device)
        # weight = torch.load('checkpoints/unt_rdefnet18/last.pth')

        n_classes = 2

        gen_model_name = 'spnet'
        gen_model = SPNet(64, input_size=120, num_palette=16)
        # gen_model = unt_rdefnet152(3, input_size=120)
        gen_model = gen_model.to(device=device)

        dis_model_name = 'vgg19'
        dis_model = vgg(dis_model_name, n_classes, freezing=True)
        dis_model = dis_model.to(device=device)

        gan_model_name = 'GanModel'
        
        if weight is None:
            weight_ep = os.listdir(f'checkpoints/{gan_model_name}/{gen_model_name}/')
            # weight_ep.remove('using_ssim')
            # weight_ep.remove('using_mse')
            weight_ep.remove('old_version')
            # weight_ep.remove('best.pth')
            weight_ep.remove('last.pth')
            weight_ep.sort(key = lambda x:int(x[2:-4]))
            weight_ep.append('last.pth')
        else:
            weight_ep = [weight]

        # print(weight_ep)
        for w in weight_ep:
            print(w)
            w = w[:-4]
            weight_ = f'checkpoints/{gan_model_name}/{gen_model_name}/{w}.pth'
            weight_ = torch.load(weight_)
            model = GanModel(gen_model, dis_model)
            model.load_state_dict(weight_)
            model = model.to(device=device)
            model.eval()
            # for i in range(16):
            #     print(f'-------------\nchannel: {i}')
            b, c     = model(a, pred=True)
            # b[:, i] = b[:, i] * 0
            if c.shape[-1] == 2:
                c_1 = b[:, 0].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
                c_2 = b[:, 1].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 0, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
                c_3 = b[:, 2].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 0].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
                c_4 = b[:, 3].unsqueeze(1).expand([-1, 3, -1, -1]) * c[:, :, 1, 1].unsqueeze(-1).unsqueeze(-1).expand([-1, -1, b.shape[-2], b.shape[-1]])
                out = c_1 + c_2 + c_3 + c_4
            elif c.shape[-1] == 4:
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
                out = c_1 + c_2 + c_3 + c_4 + c_5 + c_6 + c_7 + c_8 + c_9 + c_10 + c_11 + c_12 + c_13 + c_14 + c_15 + c_16
            dis_out = model.dis_forward(out)

            ssim_loss = ssim(out, a)
            print(f'{w}_ssim: ', ssim_loss.data.cpu())

            print(f'{w}_dis: ', dis_out.data.cpu())
            out = out[0].permute((1, 2, 0))
            out = out.data.cpu().numpy()
            out[:, :, ::1] = out[:, :, ::-1]
            cv2.imwrite(f'checkdata/{image_id}_{w}.jpg', out*255)
            # out[:, :, ::1] = out[:, :, ::-1]
            # cv2.imshow('a', out)
            # cv2.waitKey(0)
            b = b.data.cpu().numpy()
            b = b.astype(np.uint8)
            for i in range(16):
                jj=b[0, i]*255
                # jj=cv2.applyColorMap(b[0, i], cv2.COLORMAP_JET)
                print(f'ch{i}')
                cv2.imshow(f'ch{i}', jj)
                cv2.waitKey(0)

if __name__ == "__main__":
    
    img_list = ['56746_1', '57990_1', '88100_1', '59998_2', '57820_1']
    gan_pred(image_id_list=img_list, weight='last.pth')