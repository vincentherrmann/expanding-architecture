import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function
from scipy.stats import rankdata


class ExpandableParameter(Parameter):
    def __init__(self, *args, expandable_module=None, **kwargs):
        Parameter.__init__(self, *args, **kwargs)
        self.expandable_module = expandable_module

    def __new__(cls, data=None, requires_grad=True, expandable_module=None):
        p =  super(ExpandableParameter, cls).__new__(cls, data, requires_grad=requires_grad)
        p.expandable_module = expandable_module
        return p


class MutatingModule(object):
    def __init__(self, *args, **kwargs):
        self.init_ncc()
        self.current_ncc = None

    def init_ncc(self):
        self.register_buffer("t0_weight", self.weight.data.clone())
        self.register_buffer("start_ncc", torch.zeros(self.out_channels).type_as(self.t0_weight))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w_0 = self.t0_weight.view(self.weight.size(0), -1)  # size: (G, F*J*K)
        mean_0 = torch.mean(w_0, dim=1).unsqueeze(1) #.expand_as(w_0)
        t0_factor = w_0 - mean_0
        t0_norm = torch.norm(w_0, p=2, dim=1)

        w = self.weight.data.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)

        # If there is only one input channel, no sensible ncc can be calculated, return instead the ratio of the norms
        if self.in_channels == 1 and sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(t0_norm, 2)
            ncc = ncc - self.start_ncc
            self.current_ncc = ncc
            return ncc

        mean = torch.mean(w, dim=1).unsqueeze(1) #.expand_as(w)
        t_factor = w - mean
        h_product = t0_factor * t_factor
        covariance = torch.sum(h_product, dim=1) #/ (w.size(1)-1)

        denominator = t0_norm * t_norm #+ 0.05 # add a relatively small constant to avoid uncontrolled expansion for small weights

        ncc = covariance / denominator
        ncc = (ncc - self.start_ncc).squeeze()
        self.current_ncc = ncc

        return ncc

    def split_feature(self, feature_number):
        '''
        Use this method as interface!

        :param feature_number:
        :return:
        '''

        if self.fixed_feature_count:
            return self.current_ncc

        new_ncc = self._split_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            print("input tied: ", dep)
            dep._split_input_channel(channel_number=feature_number)
            if dep.in_channels != self.out_channels:
                print("wrong number of in channels in", dep)
                pass
        for dep in self.output_tied_modules:
            print("output tied: ", dep)
            dep._split_output_channel(channel_number=feature_number)
            if dep.out_channels != self.out_channels:
                print("wrong number of out channels in", dep)
                pass

        return new_ncc

    def prune_feature(self, feature_number):
        if self.fixed_feature_count:
            return self.current_ncc
        new_ncc = self._prune_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            dep._prune_input_channel(channel_number=feature_number)
        for dep in self.output_tied_modules:
            dep._prune_output_channel(channel_number=feature_number)

        return new_ncc

    def _split_output_channel(self, channel_number, offset=1):
        '''
        Split one output channel (a feature) in two, but retain the same summed value

        :param channel_number: The number of the channel that will be split
        '''

        # weight tensor: (out_channels, in_channels, kernel_size_x, kernel_size_y)
        self.out_channels += 1

        original_weight = self.weight.data
        slice = original_weight[channel_number, :]
        new_weight = insert_slice(original_weight, slice, dim=0, at_index=channel_number + offset)

        if self.bias is not None:
            original_bias = self.bias.data
            new_bias = insert_slice(original_bias, original_bias[channel_number:channel_number + 1], dim=0,
                                    at_index=channel_number + offset)
            self.bias = ExpandableParameter(new_bias, expandable_module=self)

        self.weight = ExpandableParameter(new_weight, expandable_module=self)

        # update persistent values
        self.t0_weight[channel_number, :] = self.weight.data[channel_number, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight.data[channel_number + offset, :], dim=0,
                                      at_index=channel_number + offset)
        self.start_ncc[channel_number] = 0
        self.start_ncc = insert_slice(self.start_ncc, torch.zeros(1).type_as(self.start_ncc), dim=0, at_index=channel_number + offset)
        ncc = self.normalized_cross_correlation()
        self.start_ncc[channel_number] = ncc[channel_number]
        self.start_ncc[channel_number + offset] = ncc[channel_number + offset]

        return self.normalized_cross_correlation()

    def _prune_output_channel(self, channel_number):
        self.out_channels -= 1
        new_weight = remove_slice(self.weight.data, dim=0, at_index=channel_number)
        self.weight = ExpandableParameter(new_weight, expandable_module=self)

        if self.bias is not None:
            self.bias = ExpandableParameter(remove_slice(self.bias.data, dim=0, at_index=channel_number),
                                                 expandable_module=self)

        # update persistent values
        self.t0_weight = remove_slice(self.t0_weight, dim=0, at_index=channel_number)
        self.start_ncc = remove_slice(self.start_ncc, dim=0, at_index=channel_number)

        return self.normalized_cross_correlation()

    def _split_input_channel(self, channel_number, offset=1):
        if channel_number > self.in_channels:
            print("cannot split in channel ", channel_number)
            return

        self.in_channels += 1
        original_weight = self.weight.data

        # create split positions that scale down the largest weights and scale up the smallest
        duplicated_slice = original_weight[:, channel_number, :].clone()
        ranks = torch.from_numpy(rankdata(torch.abs(duplicated_slice).cpu().numpy(), 'ordinal')).float() # rank by absolute value
        split_positions = ((len(ranks) - ranks.view_as(duplicated_slice)) / len(ranks)) * 2.5 #2.618
        split_positions += torch.zeros(duplicated_slice.size()).uniform_(-0.1, 0.1) # add noise
        split_positions = split_positions.type_as(duplicated_slice)
        # shuffle positive and negative split positions
        split_tensor = torch.stack([split_positions, (1-split_positions)], dim=0)
        noise_size = [1, self.out_channels]
        noise_size.extend(self.kernel_size)
        noise_idx = torch.zeros(noise_size).byte().bernoulli_(0.5).long()
        if self.weight.data.is_cuda:
            noise_idx = noise_idx.cuda()
        #noise_idx = torch.ByteTensor(size=noise_size).bernoulli_(0.5).long()
        split1 = torch.gather(split_tensor, dim=0, index=noise_idx).squeeze(0)
        split2 = torch.gather(split_tensor, dim=0, index=1-noise_idx).squeeze(0)

        slice1 = duplicated_slice * split1
        slice2 = duplicated_slice * split2
        original_weight[:, channel_number, :] = slice1
        new_weight = insert_slice(original_weight,
                                  slice2,
                                  dim=1,
                                  at_index=channel_number + offset)

        self.weight = ExpandableParameter(new_weight, expandable_module=self)

        # update persistent values
        self.t0_weight[:, channel_number, :] = self.weight.data[:, channel_number, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight.data[:, channel_number + offset, :], dim=1,
                                      at_index=channel_number + offset)

        return self.normalized_cross_correlation()

    def _prune_input_channel(self, channel_number):
        self.in_channels += -1
        self.weight = ExpandableParameter(remove_slice(self.weight.data, dim=1, at_index=channel_number),
                                          expandable_module=self)

        # update persistent values
        self.t0_weight = remove_slice(self.t0_weight, dim=1, at_index=channel_number)

        return self.normalized_cross_correlation()

    def tie_input_to(self, modules):
        self.back_ties.extend(modules)
        back_ties = list(self.back_ties)
        for back_tie in back_ties:
            back_tie.input_tied_modules.append(self)

    def tie_outputs(self, module):
        input_ties = module.input_tied_modules + self.input_tied_modules
        previous_ties = list(self.output_tied_modules)
        new_ties = list(module.output_tied_modules)

        # copy ties from module to all previous ties
        for output_tie in previous_ties:
            output_tie.output_tied_modules.extend(new_ties)
            output_tie.output_tied_modules.append(module)
            output_tie.input_tied_modules = list(input_ties)

        # tie all ties from the new module to all previous ties
        for output_tie in new_ties:
            output_tie.output_tied_modules.extend(previous_ties)
            output_tie.output_tied_modules.append(self)
            output_tie.input_tied_modules = list(input_ties)

        self.output_tied_modules.extend(new_ties)
        module.output_tied_modules.extend(previous_ties)

        self.output_tied_modules.append(module)
        module.output_tied_modules.append(self)

        self.input_tied_modules = list(input_ties)
        module.input_tied_modules = list(input_ties)
        pass


class Conv1dExtendable(nn.Conv1d, MutatingModule):
    def __init__(self,
                 *args,
                 fixed_feature_count=False,
                 input_tied_modules=[],
                 output_tied_modules=[],
                 **kwargs):
        nn.Conv1d.__init__(self, *args, **kwargs)
        MutatingModule.__init__(self)
        #super(Conv1dExtendable, self).__init__(*args, **kwargs)
        self.input_tied_modules = input_tied_modules # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = output_tied_modules # modules whose output size has to be compatible this this modules output
        self.fixed_feature_count=fixed_feature_count

    def forward(self, input):
        x = nn.Conv1d.forward(self, input)
        #print("out: ", x.size())
        return x


class Conv2dExtendable(nn.Conv2d, MutatingModule):
    def __init__(self,
                 *args,
                 fixed_feature_count=False,
                 input_tied_modules=[],
                 output_tied_modules=[],
                 **kwargs):
        nn.Conv2d.__init__(self, *args, **kwargs)
        MutatingModule.__init__(self)
        #super(Conv2dExtendable, self).__init__(*args, **kwargs)
        self.input_tied_modules = input_tied_modules # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = output_tied_modules # modules whose output size has to be compatible this this modules output
        self.fixed_feature_count = fixed_feature_count

    def forward(self, input):
        x = nn.Conv2d.forward(self, input)
        #print("out: ", x.size())
        return x


class Flatten2d1d(nn.Module):
    def __init__(self, *args, in_channels, in_h, in_w, out_dimension=1, input_tied_modules=[], **kwargs):
        super(Flatten2d1d, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.offset = in_h * in_w
        self.out_channels = in_channels * self.offset
        self.out_dimensions = out_dimension
        self.input_tied_modules = input_tied_modules

    def forward(self, input):
        x = input.view(-1, self.out_channels, self.out_dimensions)
        return x

    def _split_input_channel(self, channel_number):
        for m in self.input_tied_modules:
            for i in range(self.offset):
                m._split_input_channel(channel_number*self.offset + i, offset=self.offset)

        self.in_channels += 1
        self.out_channels = self.in_channels * self.offset

    def _prune_input_channel(self, channel_number):
        for m in self.input_tied_modules:
            for i in range(self.offset):
                m.prune_input_channel(channel_number*self.offset)

        self.in_channels += -1
        self.out_channels = self.in_channels * self.offset


def insert_slice(tensor, slice, dim=0, at_index=0):
    '''
    insert a slice at a given position into a tensor

    :param tensor: The tensor in which the slice will be inserted
    :param slice: The slice. Should have the same size as the tensor, except the insertion dimension, which should be 1 or missing
    :param dim: The dimension in which the slice gets inserted
    :param at_index: The index at which the slice gets inserted
    :return: The new tensor with the inserted slice
    '''

    if len(slice.size()) < len(tensor.size()):
        slice = slice.unsqueeze(dim)

    if at_index > 0:
        s1 = tensor.narrow(dim, 0, at_index)
        result = torch.cat((s1, slice), dim)
    else:
        result = slice

    s2_length = tensor.size(dim) - at_index
    if s2_length > 0:
        s2 = tensor.narrow(dim, at_index, s2_length)
        result = torch.cat((result, s2), dim)

    return result.contiguous()


def remove_slice(tensor, dim=0, at_index=0):

    if at_index > 0:
        s1 = tensor.narrow(dim, 0, at_index)
    else:
        return tensor.narrow(dim, 1, tensor.size(dim)-1).contiguous()

    s2_length = tensor.size(dim) - at_index - 1
    if s2_length > 0:
        s2 = tensor.narrow(dim, at_index+1, s2_length)
        return torch.cat((s1, s2), dim).contiguous()
    else:
        return s1.contiguous()


class SELU(nn.Module):
    def __init__(self, inplace=False):
        super(SELU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
        self.inplace = inplace

    def forward(self, input):
        return self.scale * F.elu(input, self.alpha, self.inplace)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'alpha=' + str(self.alpha) + 'scale=' + str(self.scale) \
            + inplace_str + ')'