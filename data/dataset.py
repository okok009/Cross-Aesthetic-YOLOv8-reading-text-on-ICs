import os
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import time

class GenDataset(Dataset):
    def __init__(self, img_dir, transform=None, device = 'cpu'):
        self.img_path = img_dir
        self.img_ids = os.listdir(img_dir)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image = read_image(self.img_path + "/" + self.img_ids[idx])
        image = image / 255
        if self.transform:
            image = self.transform(image)

        return image

class ClsDataset(Dataset):
    def __init__(self, cls_dir, transform=None, device = 'cpu'):
        self.img_path = cls_dir
        class_names = os.listdir(cls_dir)
        self.num_classes = len(class_names)
        self.img_ids = []
        self.cls_ids = []
        for i in range(self.num_classes):
            img_ids = os.listdir(cls_dir + class_names[i])
            for j in range(len(img_ids)):
                self.img_ids.append(class_names[i] + "/" + img_ids[j])
                self.cls_ids.append(i)
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        cls_onehot = torch.zeros([self.num_classes])
        cls_onehot[self.cls_ids[idx]] = 1.
        image = read_image(self.img_path + "/" + self.img_ids[idx]) 
        image = image / 255
        
        if self.transform:
            image = self.transform(image)

        return image, cls_onehot

class SegDataset(Dataset):
    def __init__(self, img_dir, label_dir, target_info, mode, transform=None, target_transform=None, device = 'cpu'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.segimg = os.listdir(label_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.device = device
        self.target_info = target_info
        self.mode = mode

    def __len__(self):
        return len(self.segimg)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.segimg[idx][:-4]+'.jpg')
        label_path = os.path.join(self.label_dir, self.segimg[idx])
        image = read_image(img_path)
        label = read_image(label_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label_t = torch.zeros(self.target_info)

        if self.mode == 'train':
            for i in range(self.target_info[0]):
                class_filter = torch.ones(1, self.target_info[1], self.target_info[2])*i
                class_filter = class_filter == label
                label_t[i] = class_filter

            return image, label_t

        return image, label

def seg_dataloader(image_path: str, label_path: str, target_info: tuple, batch_size: int, mode: str, transform: v2.Transform = None, target_transform: v2.Transform = None,  shuffle: bool = None, num_workers: int = 1, device='cpu'):
    dataset = SegDataset(image_path, label_path, target_info, mode, transform = transform, target_transform=target_transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def cls_dataloader(cls_dir, transform, batch_size=1, shuffle=False, num_workers=2, device='cpu'):
    dataset = ClsDataset(cls_dir, transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def gen_dataloader(img_dir, transform, batch_size=1, shuffle=False, num_workers=2, device='cpu'):
    dataset = GenDataset(img_dir, transform=transform, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

def collate_fn(batch) -> tuple:
    '''
    因為每張圖片的bounding box數量不同,所以自行定義了collate_fn來完成batch。
    將bounding box直接用list的方式輸出,並在各bounding box的字典中加入'img'來代表是來自batch中第幾張照片的框。
    '''
    img, target = zip(*batch)
            
    return img, target

if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    target_transform = v2.Compose([
        v2.Resize((200, 200), antialias=True)
    ])

    # img_txt = 'E:/ray_workspace/CrossAestheticYOLOv8/data/broken_clean_img.txt'
    # img_path = 'E:/Datasets/ICText/train2021/'
    # json_path = 'E:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'
    cls_dir = 'D:/Datasets/ICText_cls/train/'


    dataset = ClsDataset(cls_dir, transform=transform, device = 'cpu')
    train_dataloader = DataLoader(dataset, batch_size=5, shuffle=False, num_workers=3)
    i = 5
    start = time.time()
    for  img, cls in train_dataloader:
        print(img.shape)
    