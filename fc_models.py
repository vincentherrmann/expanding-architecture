import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
from expanding_modules import Conv1dExtendable, Conv2dExtendable, Flatten2d1d, MutatingModule
from splitting_layers import Conv1dSplittable


class FC_Splittable(nn.Module):
    def __init__(self, layer_sizes=[784, 4, 4, 4, 10]):
        # default channel counts: 1-10-20 320-50-10
        super(FC_Splittable, self).__init__()

        module_list = []

        for idx in range(len(layer_sizes) - 2):
            module_list.append(("fc"+str(idx),
                                Conv1dSplittable(layer_sizes[idx],
                                                 layer_sizes[idx+1],
                                                 kernel_size=1,
                                                 bias=False)))
            module_list.append(("relu"+str(idx), nn.ReLU(True)))

        module_list.append(("fc"+str(len(layer_sizes)-2),
                            Conv1dSplittable(layer_sizes[-2],
                                             layer_sizes[-1],
                                             kernel_size=1,
                                             bias=True,
                                             fixed_feature_count=True)))

        dict = OrderedDict(module_list)

        # define ncc dependencies
        # for idx in range(len(layer_sizes) - 2):
        #     dict["fc"+str(idx)].input_tied_modules = [dict["fc"+str(idx+1)]]

        self.seq = nn.Sequential(dict)
        self.layer_count = len(layer_sizes) - 1

    def forward(self, x):
        x = x.view(-1, 28*28, 1)
        x = self.seq(x).squeeze()
        return F.log_softmax(x)

    def print(self):
        print()
        for module in self.seq.children():
            if isinstance(module, Conv1dSplittable):
                module.print()
        print("out \n")

    def tie_check(self):
        for module in self.modules():
            if isinstance(module, Conv1dSplittable):
                # check output ties
                for output_tied in module.output_tied_modules:
                    if module not in output_tied.output_tied_modules:
                        print("module", module, "is missing in the output ties of", output_tied)
                # check input ties
                for back_tie in module.back_ties:
                    if module not in back_tie.input_tied_modules:
                        print("module", module, "is missing in the input ties of", back_tie)
                # check back ties
                for input_tied in module.input_tied_modules:
                    if module not in input_tied.back_ties:
                        print("module", module, "is missing in the back ties of", input_tied)
