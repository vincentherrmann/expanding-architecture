import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from expanding_modules import Conv1dExtendable, Conv2dExtendable, Flatten2d1d, MutatingModule


class FC_Net(nn.Module):
    def __init__(self, layer_sizes=[784, 4, 4, 4, 10]):
        # default channel counts: 1-10-20 320-50-10
        super(FC_Net, self).__init__()
        # self.fc1 = Conv1dExtendable(784, 32, kernel_size=1, bias=False)
        # self.fc2 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc3 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc4 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc5 = Conv1dExtendable(32, 10, kernel_size=1, bias=True, fixed_feature_count=True)

        module_list = []

        for idx in range(len(layer_sizes) - 2):
            module_list.append(("fc"+str(idx),
                                Conv1dExtendable(layer_sizes[idx],
                                                 layer_sizes[idx+1],
                                                 kernel_size=1,
                                                 bias=False)))
            module_list.append(("relu"+str(idx), nn.ReLU(True)))

        module_list.append(("fc"+str(len(layer_sizes)-2),
                            Conv1dExtendable(layer_sizes[-2],
                                             layer_sizes[-1],
                                             kernel_size=1,
                                             bias=True,
                                             fixed_feature_count=True)))

        dict = OrderedDict(module_list)

        # define ncc dependencies
        for idx in range(len(layer_sizes) - 2):
            dict["fc"+str(idx)].input_tied_modules = [dict["fc"+str(idx+1)]]

        self.seq = nn.Sequential(dict)
        self.layer_count = len(layer_sizes) - 1

    def forward(self, x):
        x = x.view(-1, 28*28, 1)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)


class Conv_Net(nn.Module):
    def __init__(self, conv=[1, 4, 4], kernels_size=[5, 5], fc=[4, 10]):
        super(Conv_Net, self).__init__()

        module_list = []
        img_size = 28

        for c_idx in range(len(conv) - 1):
            module_list.append(("conv_" + str(c_idx),
                               Conv2dExtendable(conv[c_idx],
                                                conv[c_idx+1],
                                                kernel_size=[kernels_size[c_idx], kernels_size[c_idx]],
                                                bias=False)))
            img_size += -(kernels_size[c_idx] - 1)
            module_list.append(("pool_" + str(c_idx), nn.MaxPool2d(2)))
            img_size = int(img_size/2)
            module_list.append(("relu_" + str(c_idx), nn.ReLU(True)))

        module_list.append(("conv_drop", nn.Dropout2d()))
        module_list.append(("flatten", Flatten2d1d(in_channels=conv[-1],
                                                   in_h=img_size,
                                                   in_w=img_size)))

        for f_idx in range(len(fc)):
            if f_idx == 0:
                in_channels = img_size*img_size*conv[-1]
            else:
                in_channels = fc[f_idx-1]
            if f_idx < len(fc)-1:
                module_list.append(("fc_" + str(f_idx),
                                    Conv1dExtendable(in_channels=in_channels,
                                                     out_channels=fc[f_idx],
                                                     kernel_size=1,
                                                     bias=False)))
            else:
                module_list.append(("fc_" + str(f_idx),
                                    Conv1dExtendable(in_channels=in_channels,
                                                     out_channels=fc[f_idx],
                                                     kernel_size=1,
                                                     bias=True,
                                                     fixed_feature_count=True)))

            module_list.append(("relu", nn.ReLU(True)))

        module_list.append(("fc_drop", nn.Dropout()))

        # module_list.append(("conv_1", Conv2dExtendable(in_channels=1,
        #                                                out_channels=4,
        #                                                kernel_size=[5, 5],
        #                                                bias=False)))
        # module_list.append(("pool_1", nn.MaxPool2d(2)))
        # module_list.append(("relu_1", nn.ReLU(True)))
        # module_list.append(("conv_2", Conv2dExtendable(in_channels=4,
        #                                                out_channels=4,
        #                                                kernel_size=[5, 5],
        #                                                bias=False)))
        #
        # module_list.append(("conv_drop", nn.Dropout2d()))
        # module_list.append(("pool_2", nn.MaxPool2d(2)))
        # module_list.append(("relu_2", nn.ReLU(True)))
        #
        # module_list.append(("flatten", Flatten2d1d(in_channels=4,
        #                                            in_h=4,
        #                                            in_w=4)))
        # module_list.append(("fc_1", Conv1dExtendable(in_channels=64,
        #                                              out_channels=4,
        #                                              kernel_size=1,
        #                                              bias=False)))
        # module_list.append(("relu_3", nn.ReLU(True)))
        # module_list.append(("fc_drop", nn.Dropout()))
        # module_list.append(("fc_2", Conv1dExtendable(in_channels=4,
        #                                              out_channels=10,
        #                                              kernel_size=1,
        #                                              bias=True,
        #                                              fixed_feature_count=True)))

        dict = OrderedDict(module_list)

        for idx in range(len(conv) - 2):
            dict["conv_"+str(idx)].input_tied_modules = [dict["conv_"+str(idx+1)]]
        dict["conv_"+str(len(conv) - 2)].input_tied_modules = [dict["flatten"]]
        dict["flatten"].input_tied_modules = [dict["fc_0"]]
        for idx in range(len(fc) - 1):
            dict["fc_"+str(idx)].input_tied_modules = [dict["fc_"+str(idx+1)]]

        # dict["conv_1"].input_tied_modules = [dict["conv_2"]]
        # dict["conv_2"].input_tied_modules = [dict["flatten"]]
        # dict["flatten"].input_tied_modules = [dict["fc_1"]]
        # dict["fc_1"].input_tied_modules = [dict["fc_2"]]


        self.layer_count = len(conv) + len(fc) - 1
        self.seq = nn.Sequential(dict)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)


class SequentialNamed(nn.Sequential):
    def __init__(self, *args):
        super(SequentialNamed, self).__init__(*args)

    def forward(self, input):
        for (name, module) in self._modules.items():
            input = module(input)
        return input

