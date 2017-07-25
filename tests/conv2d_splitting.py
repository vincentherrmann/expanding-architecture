import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from models import FC_Net, Conv_Net
from training import OptimizerMNIST
from torch.autograd import Variable
from expanding_modules import Conv1dExtendable, Conv2dExtendable, Flatten2d1d
from collections import OrderedDict


class Test_Model(nn.Module):
    def __init__(self):
        super(Test_Model, self).__init__()

        module_list = []

        module_list.append(("conv_1", Conv2dExtendable(in_channels=1,
                                                       out_channels=4,
                                                       kernel_size=[5, 5],
                                                       bias=False)))
        module_list.append(("relu_1", nn.ReLU(True)))
        module_list.append(("conv_2", Conv2dExtendable(in_channels=4,
                                                       out_channels=4,
                                                       kernel_size=[5, 5],
                                                       bias=False)))
        module_list.append(("relu_2", nn.ReLU(True)))
        module_list.append(("flatten", Flatten2d1d(in_channels=4,
                                                   in_h=20,
                                                   in_w=20,
                                                   out_channels=1600)))
        module_list.append(("fc_1", Conv1dExtendable(in_channels=1600,
                                                     out_channels=10,
                                                     kernel_size=1,
                                                     bias=False,
                                                     fixed_feature_count=True)))
        # module_list.append(("conv_3", Conv2dExtendable(in_channels=4,
        #                                                out_channels=1,
        #                                                kernel_size=[10, 10],
        #                                                bias=False,
        #                                                fixed_feature_count=True)))

        dict = OrderedDict(module_list)

        dict["conv_1"].input_tied_modules = [dict["conv_2"]]
        #dict["conv_2"].input_tied_modules = [dict["conv_3"]]
        dict["conv_2"].input_tied_modules = [dict["flatten"]]
        dict["flatten"].input_tied_modules = [dict["fc_1"]]

        self.seq = nn.Sequential(dict)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

model = Test_Model()
trainer = OptimizerMNIST(model, epochs=10)

file = "trained_model_784_4x32_10"

# print("initial test run:")
# orig_test_loss, correct = trainer.test()

# Error if in the layer before flatten a feature other than 0 is splitted

test_img = Variable(torch.rand(28, 28))
print(model(test_img))
print("parameter count: ", model.parameter_count())

for name, module in model.named_modules():
    if name == "seq.conv_2": #or name == "seq.conv_2":
        module.split_feature(feature_number=1)

for name, module in model.named_modules():
    if name == "seq.conv_1": #or name == "seq.conv_2":
        module.split_feature(feature_number=3)

for name, module in model.named_modules():
    if name == "seq.conv_2": #or name == "seq.conv_2":
        module.split_feature(feature_number=3)

print(model(test_img))
print("parameter count: ", model.parameter_count())
