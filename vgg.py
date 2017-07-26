# adapted from https://github.com/kuangliu/pytorch-cifar/
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from collections import OrderedDict
from models import SequentialNamed
from expanding_modules import Conv1dExtendable, Conv2dExtendable, Flatten2d1d, SELU


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, in_channels=3, cfg=[4, 'M', 4, 'M', 4, 4, 'M', 4, 4, 'M', 4, 4, 'M'], out_channels=10):
        super(VGG, self).__init__()

        module_list = []
        layer = 0
        for x in cfg:
            l = str(layer)
            if x == 'M':
                module_list += [("pool_" + l, nn.MaxPool2d(kernel_size=2, stride=2))]
            else:
                module_list += [("conv_" + l, Conv2dExtendable(in_channels=in_channels,
                                                               out_channels=x,
                                                               kernel_size=[3, 3],
                                                               bias=True,
                                                               padding=1)),
                                #("bn_" + l, nn.BatchNorm2d(x)),
                                ("relu_" + l, nn.SELU(inplace=True))]
                in_channels = x
                layer += 1

        classifier_in = cfg[-2]
        module_list += [("flatten", Flatten2d1d(in_channels=classifier_in,
                                                in_w=1,
                                                in_h=1))]
        module_list += [("fc", Conv1dExtendable(in_channels=classifier_in,
                                                out_channels=out_channels,
                                                kernel_size=1,
                                                bias=True,
                                                fixed_feature_count=True))]

        dict = OrderedDict(module_list)

        for l in range(layer - 1):
            dict["conv_" + str(l)].input_tied_modules = [dict["conv_" + str(l + 1)]]
        dict["conv_" + str(layer - 1)].input_tied_modules = [dict["flatten"]]
        dict["flatten"].input_tied_modules = [dict["fc"]]

        self.layer_count = layer
        self.seq = SequentialNamed(dict)

    def forward(self, x):
        out = self.seq(x)
        out = out.squeeze()
        #out = self.classifier(out)
        return out