import torch
import cv2
import os
import torch.nn as nn
from nets.unet_def import unt_rdefnet18, unt_rdefnet152
from nets.spnet import SPNet
from nets.gan import GanModel
from nets.classify import vgg
from data.cut_bbx import cut_bbx
from torchvision.io import read_image
from torchvision.transforms import v2

'''
將圖片中所有的bouding box都做生成或添加雜訊
'''

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

def box_sort(name):
    if name[-6] == '_':
        num = name[-5]
    elif name[-7] == '_':
        num = name[-6:-4]
    elif name[-8] == '_':
        num = name[-7:-4]
    else:
        raise ValueError
    return int(num)

def dataload(cut, image_id, names, device, box_path):
    box, box_annotation = cut(img_id = image_id, img_txt = 'not_only_broken')
    boxs = None
    names = sorted(names, key=box_sort)
    for name in names:
        box = read_image(box_path + "/" + name)
        box = box.unsqueeze(0)
        box = box / 255
        box = box.to(device)
        if boxs is None:
            boxs = box
        else:
            boxs = torch.cat((boxs, box), 0)
    return boxs, box_annotation

def composite(model_name=None, write=False, show=False, new_folder='', train_val='train'):
    '''
    data
    '''
    if train_val == 'train':
        json_path = 'D:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
        data_path = 'D:/Datasets/ICText/train2021/'
        box_path = 'D:/Datasets/ICText_cls/train/not_only_broken_img'
        new_data_path = 'D:/Datasets/ICText_pair/images/'
        image_ids = sorted(os.listdir(box_path))
        image_ids_ = []

    elif train_val == 'val':
        json_path = 'D:/Datasets/ICText/annotation/GOLD_REF_VAL_FINAL.json'
        data_path = 'D:/Datasets/ICText/val2021/'
        box_path = 'D:/Datasets/ICText_cls/test/not_only_broken_img'
        new_data_path = 'D:/Datasets/ICText_pair/images/'
        image_ids = sorted(os.listdir(box_path))
        image_ids_ = []

    '''
    model
    (只限定unt_rdefnet152或SPNet)
    '''
    if model_name is not None:
        gen_model_name = model_name
        gen_model = unt_rdefnet152(3, input_size=120) if gen_model_name == 'unt_rdefnet152' else SPNet(3, input_size=120)
        gen_model = gen_model.to(device=device)

        dis_model_name = 'vgg19'
        dis_model = vgg(dis_model_name, 2, freezing=True)
        dis_model = dis_model.to(device=device)

        gan_model_name = 'GanModel'
        weight = torch.load(f'checkpoints/{gan_model_name}/{gen_model_name}/ep150.pth')
        model = GanModel(gen_model, dis_model)
        model.load_state_dict(weight)
        model = model.to(device=device)
        model.eval()
    else:
        model = None

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
            image_ids_.append(image_id)
            names = []
            for name in image_ids:
                if image_id == name[:len(image_id)] and name[len(image_id):len(image_id)+1] == '_':
                    names.append(name)
            box, box_annotation = dataload(cut, image_id, names, device, box_path)
            
            if model is not None:
                if model_name == 'unt_rdefnet152':
                    new_box = model(box)
                else:
                    # new_box = model(box, False)
                    b, c = model(box, True)
                    randoms = torch.rand(b.shape[-2], b.shape[-1], device=b.device).unsqueeze(0)
                    randoms = (randoms > 0.15).int()
                    broken = (b[:, 13] == randoms).int()
                    broken_in = (broken == 0).int()
                    b[:, 13] = b[:, 13] * broken_in
                    b[:, 0] = broken + b[:, 0]
                    # broken = broken == b[:, 12]
                    # broken_in = (broken == 0).int()
                    # b[:, 12] = b[:, 12] * broken_in

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
                    cc = [c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11, c_12, c_13, c_14, c_15, c_16]
                    i_num = 0
                    for ccc in cc:
                        i_num = i_num+1
                        print(i_num)
                        ccc_np = ccc[0].permute((1, 2, 0))
                        ccc_np = ccc_np.data.cpu().numpy()
                        ccc_np[:, :, ::1] = ccc_np[:, :, ::-1]
                        cv2.imshow('cc', ccc_np)
                        cv2.waitKey(0)
                    new_box = c_1 + c_2 + c_3 + c_4 + c_5 + c_6 + c_7 + c_8 + c_9 + c_10 + c_11 + c_12 + c_13 + c_14 + c_15 + c_16

            else:
                randoms = torch.rand(box.shape[-2], box.shape[-1], device=box.device).unsqueeze(0)
                for b in range(box.shape[0] - 1):
                    random = torch.rand(box.shape[-2], box.shape[-1], device=box.device).unsqueeze(0)
                    randoms = torch.cat((randoms, random), 0)
                randoms = (randoms > 0.15).int().unsqueeze(1)
                new_box = box * randoms
            
            # img必須在for外面讀取 因為這是原圖
            img = cv2.imread(data_path + image_id + '.jpg')
            new_image = img
            for i in range(len(names)):
                new_box_np = new_box[i].permute((1, 2, 0))
                new_box_np = new_box_np.data.cpu().numpy()
                new_box_np[:, :, ::1] = new_box_np[:, :, ::-1]
                new_image = cut(reverse = True, img_id = image_id, bbx = new_box_np, bbox = box_annotation[i], img=new_image)

                if show:
                    cv2.imshow('new_box', new_box_np)
                    cv2.waitKey(0)

            if write:
                '''
                要經過read_image是因為pad_resize需要tensor, 跟用cv2.imread出來的nparray不同
                '''
                cv2.imwrite(new_data_path+train_val+f'_new_{new_folder}/'+f'{image_id}_new.jpg', new_image)

                img = read_image(data_path + image_id + '.jpg')
                new_image = read_image(new_data_path+train_val+f'_new_{new_folder}/'+f'{image_id}_new.jpg')
                
                img = pad_resize(img)
                new_image = pad_resize(new_image)
                
                cv2.imwrite(new_data_path+train_val+'/'+f'{image_id}.jpg', img)
                cv2.imwrite(new_data_path+train_val+f'_new_{new_folder}/'+f'{image_id}_new.jpg', new_image)
            

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # 切換model時需要到gan.py換註解
    model_name = None
    new_folder = 'D3'
    composite(model_name=model_name, write=True, show=False, new_folder=new_folder, train_val='train')