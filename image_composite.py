import torch
import cv2
import os
import torch.nn as nn
from nets.unet_def import unt_rdefnet18, unt_rdefnet152
from nets.gan import GanModel
from nets.classify import vgg
from data.cut_bbx import cut_bbx
from torchvision.io import read_image
from torchvision.transforms import v2

def pad_resize(image):
    if image.shape[-1] != image.shape[-2]:
        pad_size = abs(image.shape[-1]-image.shape[-2])
        pad = nn.ZeroPad2d((0, pad_size, 0, 0)) if image.shape[-1] < image.shape[-2] else nn.ZeroPad2d((0, 0, 0, pad_size))
        image = pad(image)
    transform = v2.Compose([
        v2.Resize((1024, 1024), antialias=True)]
    )
    image = transform(image)
    image = image.permute((1, 2, 0))
    image = image.data.cpu().numpy()
    image[:, :, ::1] = image[:, :, ::-1]
    return image

def dataload(cut, image_id, device, box_path):
    box, box_annotation = cut(img_id = image_id)
    box = read_image(box_path + "/" + image_id + '_1.jpg')
    box = box.unsqueeze(0)
    box = box / 255
    box = box.to(device)
    return box, box_annotation

def composite(cut, image_id, new_box, box_annotation):
    new_image = cut(reverse = True, img_id = image_id, bbx = new_box, bbox = box_annotation)

    return new_image

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    '''
    data
    '''
    json_path = 'D:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
    data_path = 'D:/Datasets/ICText/train2021/'
    box_path = 'D:/Datasets/ICText_cls/train/Not_broken_img'
    new_data_path = 'D:/Datasets/ICText_pair/images/train/'
    image_ids = os.listdir(box_path)
    image_ids_ = []

    '''
    model
    '''
    gen_model_name = 'unt_rdefnet152'
    gen_model = unt_rdefnet152(3, input_size=120)
    gen_model = gen_model.to(device=device)

    dis_model_name = 'vgg19'
    dis_model = vgg(dis_model_name, 2, freezing=True)
    dis_model = dis_model.to(device=device)

    gan_model_name = 'GanModel'
    weight = torch.load(f'checkpoints/{gan_model_name}/{gen_model_name}/old_version/fulldata/best.pth')
    model = GanModel(gen_model, dis_model)
    model.load_state_dict(weight)
    model = model.to(device=device)
    model.eval()

    '''
    cut
    '''
    cut = cut_bbx(data_path, json_path, 'gen')

    for image_id in image_ids:
        if image_id[-7] == '_':
            image_id = image_id[:-7]
        elif image_id[-8] == '_':
            image_id = image_id[:-8]
        else:
            image_id = image_id[:-6]

        if image_id not in image_ids_:
            box, box_annotation = dataload(cut, image_id, device, box_path)
            new_box = model(box)
            new_box = new_box[0].permute((1, 2, 0))
            new_box = new_box.data.cpu().numpy()
            new_box[:, :, ::1] = new_box[:, :, ::-1]
            new_image = composite(cut, image_id, new_box, box_annotation)
            cv2.imwrite(new_data_path+f'new_{image_id}.jpg', new_image)
            image_ids_.append(image_id)

            image = read_image(data_path + image_id + '.jpg')
            
            new_image = read_image(new_data_path+f'new_{image_id}.jpg')
            
            image = pad_resize(image)
            new_image = pad_resize(new_image)
            
            cv2.imwrite(new_data_path+f'{image_id}.jpg', image)
            cv2.imwrite(new_data_path+f'new_{image_id}.jpg', new_image)