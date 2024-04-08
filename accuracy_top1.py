import torch
import tqdm
from nets.classify import vgg
from data.dataset import cls_dataloader
from torchvision.transforms import v2

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    n_classes = 2
    model_name = 'vgg16'
    model = vgg(model_name, n_classes)
    weight = torch.load('checkpoints/train/ep4-val_loss0.47233937759756506.pth')
    model.load_state_dict(weight)
    model = model.to(device)

    cls_dir = 'D:/Datasets/ICText_cls/test/'
    transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]
    )
    val_data_loader = cls_dataloader(cls_dir, transform, batch_size=1, shuffle=False, num_workers=2)

    top1_acc = 0

    model.eval()
    with torch.no_grad():
        for img, cls_onehot in val_data_loader:
            cls_onehot = cls_onehot.to(device)
            img = img.to(device)
            output = model(img)
            if (output.round() == cls_onehot).sum() == len(cls_onehot[0]):
                top1_acc += 1
    top1_acc /= len(val_data_loader.dataset)
    print(top1_acc)