import torch
from nets.classify import vgg
from torchvision.io import read_image
from data.dataset import cls_dataloader
from torchvision.transforms import v2

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    cls_dir = 'D:/Datasets/ICText_cls/test/'
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    val_data_loader = cls_dataloader(cls_dir, transform, batch_size=1, shuffle=False, num_workers=2)

    dis_model_name = 'vgg16'
    dis_weight = f'checkpoints/{dis_model_name}/best.pth'
    dis_weight = torch.load(dis_weight)
    dis_model = vgg(dis_model_name, 2)
    dis_model.load_state_dict(dis_weight)
    dis_model = dis_model.to(device=device)
    dis_model.eval()
    for img, cls_onehot in val_data_loader:
        output = dis_model(img)
        print(output)
