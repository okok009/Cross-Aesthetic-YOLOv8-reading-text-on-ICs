import torch
import torch.nn as nn
import torchvision

import math

'''
This net finished with "YuTin", my friend in same Lab (7017_toolbox) .
'''

class DefNet(nn.Module):
    '''
    block : BasicBlock(resnet18和resnet34) or Bottleneck(resnet50,resnet101和resnet152)

    layers : 此layer要重複幾次block
    '''
    def __init__(self, block, layers) -> None:
        super().__init__()

        self.basicplanes = 64
        self.block = block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 1, layers[0])
        self.layer2 = self._make_layer(block, 2, layers[1])
        self.layer3 = self._make_layer(block, 3, layers[2])
        self.layer4 = self._make_layer(block, 4, layers[3])

        #設定conv和batchnorm的初始權重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, layer_th, blocks):
        if layer_th - 1 == 0:
            stride = 1
            down_rate = 1
        else:
            stride = 2
            down_rate = block.expansion / 2
        channels = self.basicplanes * 2 ** (layer_th - 1)
        downsample = nn.Sequential(
            nn.Conv2d(int(channels * down_rate), channels * block.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(channels * block.expansion),
        )

        layers = []
        layers.append(block(self.basicplanes, channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(self.basicplanes, channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x       = self.conv1(x)
        x       = self.bn1(x)
        feat1   = self.relu(x)

        x       = self.maxpool(feat1)
        feat2 = self.layer1(x)
        feat3 = self.layer2(feat2)
        feat4 = self.layer3(feat3)
        feat5 = self.layer4(feat4)

        return [feat1, feat2, feat3, feat4, feat5]

class Def_Block_SD(nn.Module):
    '''sdefnet18和sdefnet34的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 1

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 0.5
        else:
            in_channels_rate = self.expansion

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.offset = nn.Conv2d(channels, 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(channels, channels * self.expansion, 3, 1, 1) 
        self.bn2 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.offset(out)
        out = self.dconv(out, offset)
        out = self.bn2(out)

        out = self.relu(out)

        return out

class Def_Block_RD(nn.Module):
    '''srdefnet18和srdefnet34的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 1

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 0.5
        else:
            in_channels_rate = self.expansion

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.offset = nn.Conv2d(channels, 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(channels, channels * self.expansion, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.offset(out)
        out = self.dconv(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class Def_Block_DR(nn.Module):
    '''rdefnet18和rdefnet34的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 1

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 0.5
        else:
            in_channels_rate = self.expansion

        self.offset = nn.Conv2d(int(channels * in_channels_rate), 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(int(channels * in_channels_rate), int(channels * in_channels_rate), 3, 1, 1)

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * self.expansion, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        offset = self.offset(x)
        d_out = self.dconv(x, offset)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            d_out = self.downsample(d_out)

        out = d_out + out
        out = self.relu(out)

        return out

class Def_Bottle_SD(nn.Module):
    '''sdefnet50,sdefnet101和sdefnet152的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 4

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 2
        else:
            in_channels_rate = self.expansion

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.offset = nn.Conv2d(channels, 2*9, 3, stride, 1)
        self.dconv = torchvision.ops.DeformConv2d(channels, channels, 3, stride, 1) 
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.offset(out)
        out = self.dconv(out, offset)
        out = self.bn2(out)

        out = self.conv2(out)
        out = self.bn3(out)

        out = self.relu(out)

        return out

class Def_Bottle_RD(nn.Module):
    '''srdefnet50,srdefnet101和srdefnet152的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 4

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 2
        else:
            in_channels_rate = self.expansion

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.offset = nn.Conv2d(channels, 2*9, 3, stride, 1)
        self.dconv = torchvision.ops.DeformConv2d(channels, channels, 3, stride, 1) 
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        offset = self.offset(out)
        out = self.dconv(out, offset)
        out = self.bn2(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class Def_Bottle_DR(nn.Module):
    '''rdefnet50,rdefnet101和rdefnet152的基本block

    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

    1.若downsample不是none,表殘差分支需進行下採樣,而非identity
    '''
    expansion = 4

    def __init__(self, basicplanes, channels, stride=1, downsample=None):
        super().__init__()
        if downsample:
            if channels == basicplanes:
                in_channels_rate = 1
            else:
                in_channels_rate = 2
        else:
            in_channels_rate = self.expansion

        self.offset = nn.Conv2d(int(channels * in_channels_rate), 2*9, 3, 1, 1)
        self.dconv = torchvision.ops.DeformConv2d(int(channels * in_channels_rate), int(channels * in_channels_rate), 3, 1, 1)

        self.conv1 = nn.Conv2d(int(channels * in_channels_rate), channels, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        offset = self.offset(x)
        d_out = self.dconv(x, offset)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            d_out = self.downsample(d_out)

        out = d_out + out
        out = self.relu(out)

        return torch.sigmoid(out)

def sdefnet18(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_SD, [2, 2, 2, 2])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def srdefnet18(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_RD, [2, 2, 2, 2])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def rdefnet18(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_DR, [2, 2, 2, 2])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def sdefnet34(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_SD, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def srdefnet34(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_RD, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def rdefnet34(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Block_DR, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def sdefnet50(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_SD, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def srdefnet50(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_RD, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def rdefnet50(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_DR, [3, 4, 6, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def sdefnet101(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_SD, [3, 4, 23, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def srdefnet101(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_RD, [3, 4, 23, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def rdefnet101(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_DR, [3, 4, 23, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def sdefnet152(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_SD, [3, 8, 36, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def srdefnet152(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_RD, [3, 8, 36, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model

def rdefnet152(pretrained_own=False):
    '''
    pretrained_own : 非None表要載入自己的權重(權重檔的相對位置)
    '''
    model = DefNet(Def_Bottle_DR, [3, 8, 36, 3])
    if pretrained_own:
        model.load_state_dict(pretrained_own, strict=False)

    return model
if __name__ == '__main__':
    input_test = torch.randn(1,3,400,400)
    model = rdefnet50(pretrained_own=False)
    ooo = model(input_test)
    print(len(ooo))
    print(ooo[0].shape)
    print(ooo[1].shape)
    print(ooo[2].shape)
    print(ooo[3].shape)
    print(ooo[4].shape)