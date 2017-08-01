import torch
import torch.nn as nn
from expanding_modules import MutatingModule


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
        self.input_tied_modules = input_tied_modules # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = output_tied_modules # modules whose output size has to be compatible this this modules output
        self.fixed_feature_count=fixed_feature_count
        self.first_split = None
        self.second_split = None

    def split(self, feature_count=1):
        self.first_split = Conv1dSplittable(in_channels=self.in_channels,
                                            out_channels=feature_count,
                                            kernels_size=self.kernel_size,
                                            fixed_feature_count=False)

        self.second_split = Conv1dSplittable(in_channels=feature_count,
                                             out_channels=self.out_channels,
                                             kernel_size=self.kernel_size,
                                             fixed_feature_count=self.fixed_feature_count)

        repar_positions = torch.mm(self.second_split.weight, self.first_split.weight) / self.weight
        self.weight *= (1 - repar_positions)

    def forward(self, input):
        output = nn.Conv1d.forward(self, input)

        # call splits recursively
        if self.first_split is not None:
            x = self.first_split(input)
            x = self.second_split(x)
            output = output + x

        return output