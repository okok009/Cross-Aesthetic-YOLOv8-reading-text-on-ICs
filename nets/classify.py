import torch
import torchvision
import torch.nn as nn
from torchvision.models.vgg import VGG16_Weights, VGG19_Weights

def vgg(model_name = 'vgg16', n_classes=2, freezing=False):
    
    if model_name == 'vgg16':
        vgg = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    elif model_name == 'vgg19':
        vgg = torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT)

    if freezing:
        for param in vgg.parameters():
            param.requires_grad = False

    n_inputs = vgg.classifier[6].in_features

    vgg.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), 
        nn.ReLU(), 
        nn.Dropout(0.4),
        nn.Linear(256, n_classes), 
        nn.Softmax(dim=1))

    return vgg

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    model_name = 'vgg16'
    model = vgg(model_name, n_classes=2)
    model = model.to(device=device)