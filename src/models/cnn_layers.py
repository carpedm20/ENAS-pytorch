from torch import nn
import torch


class IdentityModule(nn.AvgPool2d):
    def __init__(self, stride, stack=1):
        super().__init__(kernel_size=1, stride=stride)
        self.stack = stack
        self.stride = stride

    def forward(self, input):
        if self.stack == 1:
            if self.stride == 1:
                return input
            else:
                return super().forward(input)
        else:
            return torch.cat([super().forward(input)] * self.stack, dim=1)

def conv_layer(conv, num_features):
    layer = nn.Sequential(
        conv,
        nn.BatchNorm2d(num_features=num_features, track_running_stats=False),
    )
    layer.input_relu = True
    return layer

def fix_layers(in_planes, out_planes, stride, _funct):
    layer = nn.Sequential(_funct(in_planes, in_planes, stride), identity(in_planes, out_planes, stride=1))
    if hasattr(layer[0], 'input_relu') and layer[0].input_relu:
        layer.input_relu = layer.input_relu
    return layer

def simple_depthwise_1x1(in_planes, out_planes, stride=1):
    if out_planes % in_planes != 0:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, groups=in_planes, bias=True)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True)


###Layer Generation Functions
def identity(in_planes, out_planes, stride=1):
    if out_planes % in_planes == 0:
        return IdentityModule(stride=stride, stack = out_planes//in_planes)
    else:
        raise Exception("unsupported oporation")

def conv_1x7_7x1(in_planes, out_planes, stride):
    layer = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=(1, 7), stride=(1, stride), padding=(0, 3), bias=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=(7, 1), stride=(stride, 1), padding=(3, 0), bias=False)
    )
    return conv_layer(layer, out_planes)


def conv_1x3_3x1(in_planes, out_planes, stride):
    layer = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=(1, stride), padding=(0, 1), bias=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
    )
    return conv_layer(layer, out_planes)


def dilated_3x3(in_planes, out_planes, stride):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                      padding=2, dilation=2, bias=False)
    return conv_layer(layer, out_planes)

def depthwise_1x1(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        return conv1x1(in_planes, out_planes, stride)
        # return fix_layers(in_planes, out_planes, stride, avg3x3)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                          padding=0, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_3x3(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv3x3(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv3x3)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                          padding=1, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_5x5(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv5x5(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv5x5)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                          padding=2, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def depthwise_7x7(in_planes, out_planes, stride):
    if out_planes % in_planes != 0:
        # return conv5x5(in_planes, out_planes, stride)
        return fix_layers(in_planes, out_planes, stride, conv7x7)
    else:
        layer = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                          padding=3, groups=in_planes, bias=False)
        return conv_layer(layer, out_planes)

def conv1x1(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)
    return conv_layer(layer, out_planes)

def conv3x3(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
    return conv_layer(layer, out_planes)

def conv5x5(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride,
                     padding=2, bias=False)
    return conv_layer(layer, out_planes)

def conv7x7(in_planes, out_planes, stride=1):
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)
    return conv_layer(layer, out_planes)

def avg3x3(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, avg3x3)
    else:
        return nn.AvgPool2d(3, stride=stride, padding=1)

def max3x3(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max3x3)
    else:
        return nn.MaxPool2d(3, stride=stride, padding=1)

def max5x5(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max5x5)
    else:
        return nn.MaxPool2d(5, stride=stride, padding=2)

def max7x7(in_planes, out_planes, stride=1):
    if in_planes != out_planes:
        return fix_layers(in_planes, out_planes, stride, max7x7)
    else:
        return nn.MaxPool2d(5, stride=stride, padding=2)


def initialize_layers_weights(m):
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
        elif isinstance(layer, nn.AvgPool2d) or isinstance(layer, nn.MaxPool2d) or isinstance(layer, IdentityModule) or isinstance(layer, nn.Sequential):
            pass
        else:
            raise Exception("Unsupported Layer Type")

CNN_LAYER_CREATION_FUNCTIONS = [identity, conv_1x3_3x1, conv_1x7_7x1, avg3x3, max3x3, conv1x1, conv3x3, depthwise_3x3, depthwise_5x5, depthwise_7x7]