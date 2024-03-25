import torch
import torch.nn as nn

from nets.defnet import sdefnet18, srdefnet18, rdefnet18
from nets.defnet import sdefnet34, srdefnet34, rdefnet34
from nets.defnet import sdefnet50, srdefnet50, rdefnet50
from nets.defnet import sdefnet101, srdefnet101, rdefnet101
from nets.defnet import sdefnet152, srdefnet152, rdefnet152

'''
This net finished with "YuTin", my friend in same Lab (7017_toolbox) .
I only use unt_rdefnet18, that why the def has some differences.
'''

class Unet_model(nn.Module):
    def __init__(self, backbone, in_channels_list, num_classes=2, out_channels_list=[64, 128, 256, 512], input_size=400):
        super().__init__()
        self.final = nn.Conv2d(64, num_classes, 1)
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.up_concat1 = Unet_Up(in_channels_list[0], out_channels_list[0], round(input_size / 2 ** 1))
        self.up_concat2 = Unet_Up(in_channels_list[1], out_channels_list[1], round(input_size / 2 ** 2))
        self.up_concat3 = Unet_Up(in_channels_list[2], out_channels_list[2], round(input_size / 2 ** 3))
        self.up_concat4 = Unet_Up(in_channels_list[3], out_channels_list[3], round(input_size / 2 ** 4))
        self.backbone = backbone

    def forward(self, inputs):
        [feat1, feat2, feat3, feat4, feat5] = self.backbone(inputs)
        outputs = self.up_concat4(feat4, feat5)
        outputs = self.up_concat3(feat3, outputs)
        outputs = self.up_concat2(feat2, outputs)
        outputs = self.up_concat1(feat1, outputs)
        
        outputs = self.up_conv(outputs)
        outputs = self.final(outputs)
        return outputs
    
    def load_state_dict(self, path):
        w = torch.load(path)
        super().load_state_dict(w)

class Unet_Up(nn.Module):
    def __init__(self, in_channels, out_channels, inputs1_size):
        super(Unet_Up, self).__init__()
        self.up     = nn.UpsamplingBilinear2d(size=inputs1_size)
        self.conv1  = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2  = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) 
        self.relu   = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

def unt_sdefnet18(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(sdefnet18(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(sdefnet18(), in_channels_list, num_classes=num_classes)
    return model

def unt_srdefnet18(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(srdefnet18(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet18(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_rdefnet18(num_classes=2, pretrained_own=None, input_size=400, freezing=False):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(rdefnet18(), in_channels_list, num_classes=num_classes, input_size=input_size)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(rdefnet18(), in_channels_list, num_classes=num_classes, input_size=input_size)
    
    if freezing:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model

def unt_sdefnet34(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(sdefnet34(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(sdefnet34(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_srdefnet34(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(srdefnet34(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet34(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_rdefnet34(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 320, 640, 768]
    
    if pretrained_own:
        model = Unet_model(rdefnet34(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(rdefnet34(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_sdefnet50(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(sdefnet50(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(sdefnet50(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_srdefnet50(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(srdefnet50(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet50(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_rdefnet50(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(rdefnet50(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(rdefnet50(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_sdefnet101(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(sdefnet101(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(sdefnet101(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_srdefnet101(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(srdefnet101(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet101(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_rdefnet101(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(rdefnet101(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(rdefnet101(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_sdefnet152(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(sdefnet152(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(sdefnet152(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_srdefnet152(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(srdefnet152(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(srdefnet152(), in_channels_list, num_classes=num_classes)
    
    return model

def unt_rdefnet152(num_classes=2, pretrained_own=None):
    '''
    num_classes : colorlabels的種類個數

    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    in_channels_list = [192, 512, 1024, 3072]
    
    if pretrained_own:
        model = Unet_model(rdefnet152(), in_channels_list, num_classes=num_classes)
        model.load_state_dict(pretrained_own)
    else:
        model = Unet_model(rdefnet152(), in_channels_list, num_classes=num_classes)
    
    return model

if __name__ == '__main__':
    input_test = torch.randn(1,3,400,400)
    model = unt_sdefnet18(num_classes=2, pretrained_own=False)
    print(model(input_test).shape)