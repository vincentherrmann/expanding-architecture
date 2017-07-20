import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math as math
from expanding_modules import Conv1dExtendable, Conv2dExtendable, Flatten2d1d
from logger import Logger


class FC_Net(nn.Module):
    def __init__(self, layer_sizes=[784, 4, 4, 4, 10], logger=None):
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
        self.logger = logger
        self.layer_count = len(layer_sizes) - 1

    def forward(self, x):
        x = x.view(-1, 28*28, 1)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def log_to_tensor_board(self, batch_idx, loss, correct_results, ncc_avg):
        # TensorBoard logging

        # loss
        self.logger.scalar_summary("loss", loss, batch_idx)
        self.logger.scalar_summary("false results", 100-correct_results, batch_idx)
        self.logger.scalar_summary("split feature ncc average", ncc_avg, batch_idx)

        # validation loss
        # validation_position = self.validation_result_positions[-1]
        # if validation_position > self.last_logged_validation:
        #     self.logger.scalar_summary("validation loss", self.validation_results[-1], validation_position)
        #     self.last_logged_validation = validation_position

        # parameter count
        self.logger.scalar_summary("parameter count", self.parameter_count(), batch_idx)

        # parameter histograms
        for tag, value, in self.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx)

        # normalized cross correlation
        for tag, module in self.named_modules():
            tag = tag.replace('.', '/')
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
                self.logger.histo_summary(tag + '/ncc', ncc.cpu().numpy(), batch_idx)


class Conv_Net(nn.Module):
    def __init__(self, conv=[1, 4, 4], kernels_size=5, fc=[4, 10]):
        super(Conv_Net, self).__init__()

        module_list = []

        module_list.append(("conv_1", Conv2dExtendable(in_channels=1,
                                                       out_channels=4,
                                                       kernel_size=[5, 5],
                                                       bias=False)))
        module_list.append(("pool_1", nn.MaxPool2d(2)))
        module_list.append(("relu_1", nn.ReLU(True)))
        module_list.append(("conv_2", Conv2dExtendable(in_channels=4,
                                                       out_channels=4,
                                                       kernel_size=[5, 5],
                                                       bias=False)))

        module_list.append(("conv_drop", nn.Dropout2d()))
        module_list.append(("pool_2", nn.MaxPool2d(2)))
        module_list.append(("relu_2", nn.ReLU(True)))

        module_list.append(("flatten", Flatten2d1d(in_channels=4,
                                                   in_h=4,
                                                   in_w=4,
                                                   out_channels=64)))
        module_list.append(("fc_1", Conv1dExtendable(in_channels=64,
                                                     out_channels=4,
                                                     kernel_size=1,
                                                     bias=False)))
        module_list.append(("relu_3", nn.ReLU(True)))
        module_list.append(("fc_drop", nn.Dropout()))
        module_list.append(("fc_2", Conv1dExtendable(in_channels=4,
                                                     out_channels=10,
                                                     kernel_size=1,
                                                     bias=True,
                                                     fixed_feature_count=True)))

        dict = OrderedDict(module_list)

        dict["conv_1"].input_tied_modules = [dict["conv_2"]]
        dict["conv_2"].input_tied_modules = [dict["flatten"]]
        dict["flatten"].input_tied_modules = [dict["fc_1"]]
        dict["fc_1"].input_tied_modules = [dict["fc_2"]]

        self.layer_count = 4
        self.seq = nn.Sequential(dict)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def log_to_tensor_board(self, batch_idx, loss, correct_results, ncc_avg):
        # TensorBoard logging

        # loss
        self.logger.scalar_summary("loss", loss, batch_idx)
        self.logger.scalar_summary("false results", 100-correct_results, batch_idx)
        self.logger.scalar_summary("split feature ncc average", ncc_avg, batch_idx)

        # validation loss
        # validation_position = self.validation_result_positions[-1]
        # if validation_position > self.last_logged_validation:
        #     self.logger.scalar_summary("validation loss", self.validation_results[-1], validation_position)
        #     self.last_logged_validation = validation_position

        # parameter count
        self.logger.scalar_summary("parameter count", self.parameter_count(), batch_idx)

        # parameter histograms
        for tag, value, in self.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx)

        # normalized cross correlation
        for tag, module in self.named_modules():
            tag = tag.replace('.', '/')
            if type(module) is Conv1dExtendable or type(module) is Conv2dExtendable:
                ncc = module.normalized_cross_correlation()
                self.logger.histo_summary(tag + '/ncc', ncc.cpu().numpy(), batch_idx)
