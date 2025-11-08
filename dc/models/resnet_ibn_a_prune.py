import torch
import torch.nn as nn
import math


__all__ = ['ResNet', 'resnet50_ibn_a', 'resnet101_ibn_a']


model_urls = {
    'ibn_resnet50a': 'E:/Code/densityclustering-master/copy_of_server/CMCRL/examples/pretrained/resnet50_ibn_a.pth',
    'ibn_resnet101a': './examples/pretrained/resnet101_ibn_a.pth.tar',
}


### custom layers ####
class prune_conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(prune_conv2d, self).__init__(*args, **kwargs)
        self.prune_mask = torch.ones(list(self.weight.shape)).to(torch.device('cuda'))
        self.prune_flag = False

    def forward(self, input):
        if not self.prune_flag:
            weight = self.weight
            # return nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
            #                  self.bias)(input)
        else:
            weight = self.weight * self.prune_mask
        return self._conv_forward(input, weight, bias=self.bias)

    def set_prune_flag(self, flag):
        self.prune_flag = flag


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return prune_conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=padding, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return prune_conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prune_flag = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1, self.conv2]:
            module.set_prune_flag(flag)
        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, prune_conv2d):
                    layer.set_prune_flag(flag)


class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prune_flag = False

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1, self.conv2, self.conv3]:
            module.set_prune_flag(flag)
        if self.downsample is not None:
            for layer in self.downsample:
                if isinstance(layer, prune_conv2d):
                    layer.set_prune_flag(flag)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet, self).__init__()
        self.conv1 = prune_conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                                  bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.prune_flag = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion,
                        stride=stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def set_prune_flag(self, flag):
        self.prune_flag = flag
        for module in [self.conv1,]:
            module.set_prune_flag(flag)
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in stage:
                if isinstance(layer, BasicBlock) or isinstance(layer, Bottleneck) or isinstance(layer, prune_conv2d):
                    layer.set_prune_flag(flag)


def resnet50_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['ibn_resnet50a'], map_location=torch.device('cpu'))
        state_dict = remove_module_key(state_dict)
        model.load_state_dict(state_dict)
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = torch.load(model_urls['ibn_resnet101a'], map_location=torch.device('cpu'))['state_dict']
        state_dict = remove_module_key(state_dict)
        model.load_state_dict(state_dict)
    return model


def remove_module_key(state_dict):
    for key in list(state_dict.keys()):
        if 'module' in key:
            state_dict[key.replace('module.','')] = state_dict.pop(key)
    return state_dict
