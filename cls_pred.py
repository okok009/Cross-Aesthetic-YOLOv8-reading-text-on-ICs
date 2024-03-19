import torch
from nets.classify import vgg
from torchvision.io import read_image
from data.dataset import cls_dataloader
from torchvision.transforms import v2

if __name__ == "__main__":
    
    cls_dir = 'D:/Datasets/ICText_cls/test/'
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    val_data_loader = cls_dataloader(cls_dir, transform, batch_size=1, shuffle=False, num_workers=2)

    model = vgg()
    weight = torch.load('checkpoints/train/ep4-val_loss0.47233937759756506.pth')
    model.load_state_dict(weight)
    model.eval()
    for img, cls_onehot in val_data_loader:
        output = model(img)
        print(output)
