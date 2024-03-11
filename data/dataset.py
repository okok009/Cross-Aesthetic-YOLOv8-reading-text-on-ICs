import os
import torch
import torchvision.transforms as transforms
from cut_bbx import cut_bbx
from torchvision.io import read_image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import v2

class GenDataset(Dataset):
    def __init__(self, img_txt, img_dir, json_path, transform=None, device = 'cpu'):
        with open(img_txt) as f:
            self.img_ids = f.readlines()
        self.img_path = img_dir
        self.json_path = json_path
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        cut = cut_bbx(self.img_path, self.json_path)
        img, bbx, bbox = cut(img_id = self.img_ids[idx][:-1])
        bbx = torch.tensor(bbx).permute((2, 0, 1))
        return img, bbx, bbox

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

def dataloader(image_path: str, label_path: str, target_info: tuple, batch_size: int, mode: str, transform: v2.Transform = None, target_transform: v2.Transform = None,  shuffle: bool = None):
    dataset = GenDataset(image_path, label_path, target_info, mode, transform = transform, target_transform=target_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return dataloader

if __name__ == "__main__":
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        v2.Resize((200, 200), antialias=True)]
    )
    target_transform = v2.Compose([
        v2.Resize((200, 200), antialias=True)
    ])

    img_txt = 'E:/ray_workspace/CrossAestheticYOLOv8/data/broken_img.txt'
    img_path = 'E:/Datasets/ICText/train2021/'
    json_path = 'E:/Datasets/ICText/annotation/GOLD_REF_TRAIN_FINAL.json'


    dataset = GenDataset(img_txt, img_path, json_path)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for  img, bbx, bbox in train_dataloader:
        print(bbx.shape)
        break
    