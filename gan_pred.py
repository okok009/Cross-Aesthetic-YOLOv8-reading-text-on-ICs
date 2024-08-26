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

def normalize(activations):
    # transform activations so that all the values be in range [0, 1]
    activations = activations - np.min(activations[:])
    activations = activations / np.max(activations[:])
    return activations

def visualize_activations(image, activations):
    activations = normalize(activations)
 
    activations = np.stack([activations, activations, activations], axis=2)
    masked_image = np.multiply(image.permute(1, 2, 0).detach().numpy(), activations)
     
    return masked_image

def click(event, x, y, flags, param):
    global xy
    if flags == 1:
        if event == 1:
            image = read_image(f'D:/Datasets/ICText_cls/test/clean_img/{image_id}.jpg').unsqueeze(0).to(dtype=torch.float) / 255
            image = image.to(device=device)
            image.requires_grad=True
            out = model(image)
            xy = [x, y]
            print(f"x:{xy[0]}, y:{xy[1]}")
            out[0, 0, xy[1], xy[0]].backward(retain_graph=True)
            gradient_of_input = image.grad[0, 0].data.cpu().numpy()
            gradient_of_input_mean = np.mean(gradient_of_input)
            gradient_of_input_mask = (gradient_of_input > (gradient_of_input_mean * 10)) * gradient_of_input_mean * 0.4
            gradient_of_input_mask = gradient_of_input_mask + (gradient_of_input > (gradient_of_input_mean * 5)) * gradient_of_input_mean * 0.2
            gradient_of_input_mask = gradient_of_input_mask + (gradient_of_input > (gradient_of_input_mean * 1)) * gradient_of_input_mean * 0.2
            gradient_of_input_mask = gradient_of_input_mask + gradient_of_input_mean * 0.01
            gradient_of_input_mask[0, 0] = 0
            networkcare_mask = visualize_activations(image.data.cpu()[0], gradient_of_input_mask)
            networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 0] = 255
            networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 1] = 0
            networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 2] = 0
            networkcare_mask = cv2.cvtColor(networkcare_mask, cv2.COLOR_RGB2BGR)
            cv2.imshow("networkcare_mask", networkcare_mask)

def gan_pred(image_id_list=None, weight=None, gen_model_name='spnet', write=False,
              check_output=False, check_feature=False, check_receptive_field=False,
                check_networkcare=False):
    global out
    global image
    global model
    global image_id
    global device

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if image_id_list is None:
        image_id_list = ['88100_1']

    for id in image_id_list:
        image_id = id
        image = read_image(f'D:/Datasets/ICText_cls/test/clean_img/{image_id}.jpg').unsqueeze(0).to(dtype=torch.float) / 255
        image = image.to(device=device)
        if check_networkcare:
            image.requires_grad=True

        n_classes = 2

        if gen_model_name == 'spnet':
            gen_model = SPNet(64, input_size=120, num_palette=16)
        else:
            gen_model = unt_rdefnet152(3, input_size=120)   
        gen_model = gen_model.to(device=device)

        dis_model_name = 'vgg19'
        dis_model = vgg(dis_model_name, n_classes, freezing=True)
        dis_model = dis_model.to(device=device)

        gan_model_name = 'GanModel'
        
        if weight is None:
            weight_ep = os.listdir(f'checkpoints/{gan_model_name}/{gen_model_name}/')
            if 'best.pth' in weight_ep:
                weight_ep.remove('best.pth')
            if 'old_version' in weight_ep:
                weight_ep.remove('old_version')
            if 'last.pth' in weight_ep:
                weight_ep.remove('last.pth')
            weight_ep.sort(key = lambda x:int(x[2:-4]))
            weight_ep.append('best.pth')
        else:
            weight_ep = [weight]

        for w in weight_ep:
            print(w)
            w = w[:-4]
            weight_ = f'checkpoints/{gan_model_name}/{gen_model_name}/{w}.pth'
            weight_ = torch.load(weight_)
            model = GanModel(gen_model, dis_model, gen_model=gen_model_name)
            model.load_state_dict(weight_)
            model = model.to(device=device)
            model.eval()
            # for i in range(16):
            #     print(f'-------------\nchannel: {i}')
            if gen_model_name == 'spnet':
                b, c     = model(image, pred=True)
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
            else:
                out = model(image)
            dis_out = model.dis_forward(out)

            ssim_loss = ssim(out, image)
            print(f'{w}_ssim: ', ssim_loss.data.cpu())

            print(f'{w}_dis: ', dis_out.data.cpu())
            if write:
                out_w = out[0].permute((1, 2, 0))
                out_w = out_w.data.cpu().numpy()
                out_w[:, :, ::1] = out_w[:, :, ::-1]
                cv2.imwrite(f'checkdata/{image_id}_{w}.jpg', out_w*255)
            if check_output:
                out_o = out[0].permute((1, 2, 0))
                out_o = out_o.data.cpu().numpy()
                out_o[:, :, ::1] = out_o[:, :, ::-1]
                cv2.imshow('a', out_o)
                cv2.waitKey(0)
                cv2.destroyWindow('a')
            if check_feature:
                b = b.data.cpu().numpy()
                b = b.astype(np.uint8)
                for i in range(16):
                    feature = b[0, i]*255
                    print(f'ch{i}')
                    cv2.imshow(f'ch{i}', feature)
                    cv2.waitKey(0)
                    cv2.destroyWindow('ch{i}')
            if check_receptive_field:
                input = torch.ones_like(image, requires_grad=True)
                one_out = model(input)
                grad = torch.zeros_like(one_out, requires_grad=True)
                grad.data[0, 0, 90, 40] = 1.
                one_out.backward(gradient=grad)
                # one_out[0, 0, max_row_id, max_col_id].backward()
                gradient_of_input = input.grad[0, 0].data.cpu().numpy()
                receptive_field_mask = visualize_activations(image.data.cpu()[0], gradient_of_input)
                receptive_field_mask = cv2.cvtColor(receptive_field_mask, cv2.COLOR_RGB2BGR)
                cv2.imshow("receiptive_field_max_activation", receptive_field_mask)
                cv2.waitKey(0)
                cv2.destroyWindow("receiptive_field_max_activation")
            if check_networkcare:
                xy = [50, 50]
                out[0, 0, xy[1], xy[0]].backward()
                gradient_of_input = image.grad[0, 0].data.cpu().numpy()
                gradient_of_input_mean = np.mean(gradient_of_input)
                gradient_of_input_mask = (gradient_of_input > (gradient_of_input_mean * 10)) * gradient_of_input_mean * 0.4
                gradient_of_input_mask = gradient_of_input_mask + (gradient_of_input > (gradient_of_input_mean * 5)) * gradient_of_input_mean * 0.2
                gradient_of_input_mask = gradient_of_input_mask + (gradient_of_input > (gradient_of_input_mean * 1)) * gradient_of_input_mean * 0.2
                gradient_of_input_mask = gradient_of_input_mask + gradient_of_input_mean * 0.01
                gradient_of_input_mask[0, 0] = 0
                networkcare_mask = visualize_activations(image.data.cpu()[0], gradient_of_input_mask)
                networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 0] = 255
                networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 1] = 0
                networkcare_mask[xy[1]-2:xy[1]+2, xy[0]-2:xy[0]+2, 2] = 0
                networkcare_mask = cv2.cvtColor(networkcare_mask, cv2.COLOR_RGB2BGR)
                cv2.imshow("networkcare_mask", networkcare_mask)
                cv2.setMouseCallback("networkcare_mask", click)
                
                cv2.waitKey(0)
                cv2.destroyWindow("networkcare_mask")

if __name__ == "__main__":
    
    img_list = ['56746_1', '57990_1', '88100_1', '59998_2', '57820_1']
    img_list = ['88100_1']
    gan_pred(image_id_list=img_list, weight='best.pth', gen_model_name='unt_rdefnet152', write=False, check_output=True, check_receptive_field=False, check_networkcare=True)