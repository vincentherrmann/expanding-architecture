import torch
import torch.nn as nn
from torch.nn import Parameter
from expanding_modules import MutatingModule, ExpandableParameter


def pseudo_inverse(a):
    u, s, v = torch.svd(a, some=False)
    s_plus = torch.zeros(v.size(0), u.size(1)).type_as(s)
    s_plus[0:len(s), 0:len(s)] = torch.diag(1/s)
    a_plus = torch.mm(torch.mm(v, s_plus), u.t())
    return a_plus




class Conv1dSplittable(nn.Conv1d, MutatingModule):
    def __init__(self,
                 *args,
                 fixed_feature_count=False,
                 input_tied_modules=[],
                 output_tied_modules=[],
                 **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        MutatingModule.__init__(self)
        self.input_tied_modules = []#input_tied_modules # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = [] #output_tied_modules # modules whose output size has to be compatible this this modules output
        #self.back_ties = [] # the modules that determine the number of input channels of this module
        self.fixed_feature_count=fixed_feature_count
        self.first_split = None
        self.split_nl = None
        self.second_split = None
        self.weight = ExpandableParameter(self.weight.data, expandable_module=self)
        if self.bias is not None:
            self.bias = ExpandableParameter(self.bias.data, expandable_module=self)

    def split(self, feature_count=1):

        # Add intermediate layer, the original layer is now a residual connection.
        # A parametric ReLU with initial a=1 is used to create a completely linear connection at first.
        # The original weights are scaled such, that this module behaves exactly as before the split.

        #   |
        #   o---|
        # r |   | second split
        # e |   |
        # s |   o PReLU
        # i |   |
        # d |   | first split
        #   o---|
        #   |

        use_bias = True
        if self.bias is None:
            use_bias = False

        self.first_split = Conv1dSplittable(in_channels=self.in_channels,
                                            out_channels=feature_count,
                                            kernel_size=self.kernel_size,
                                            fixed_feature_count=False,
                                            bias=True)
        self.first_split.bias.data.zero_()

        self.split_nl = nn.PReLU(num_parameters=1, init=1.)

        self.second_split = Conv1dSplittable(in_channels=feature_count,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             fixed_feature_count=self.fixed_feature_count,
                                             bias=use_bias)
        if use_bias:
            self.second_split.bias.data.zero_()

        # scale weights
        w1 = self.first_split.weight.data.permute(2, 0, 1)
        w2 = self.second_split.weight.data.permute(2, 0, 1)
        repar_positions = torch.matmul(w2, w1) / self.weight.data.permute(2, 0, 1)
        self.weight = ExpandableParameter(self.weight.data * (1 - repar_positions).permute(1, 2, 0),
                                          expandable_module=self)

        # manage input and output ties
        # self.first_split.tie_input_to(self.back_ties)
        # self.second_split.tie_input_to([self.first_split])
        # input_ties = list(self.input_tied_modules)
        # for input_tie in input_ties:
        #     input_tie.tie_input_to([self.second_split])
        # self.tie_outputs(self.second_split)

        #self.second_split.back_ties = [self.first_split]
        # self.first_split.input_tied_modules = [self.second_split]
        # self.second_split.output_tied_modules = list(self.output_tied_modules)
        # self.second_split.output_tied_modules.append(self)
        # self.output_tied_modules.append(self.second_split)

        # update ncc
        self.start_ncc.zero_()
        ncc = self.normalized_cross_correlation()
        self.start_ncc = ncc

    def forward(self, input):
        output = nn.Conv1d.forward(self, input)

        # call splits recursively
        if self.first_split is not None:
            x = self.first_split(input)
            x = self.split_nl(x)
            x = self.second_split(x)
            output = output + x

        return output

    def print(self, indent=""):
        print(indent, "Conv1dSplittable_", self.in_channels, "_", self.out_channels, sep='')
        if self.first_split is not None:
            self.first_split.print(indent=indent+" | ")
            self.second_split.print(indent=indent+" | ")

    def name(self):
        name = "C1D_" + str(self.in_channels) + "_" + str(self.out_channels) + "__" + str(id(self))
        return name
