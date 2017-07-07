import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable, Function

class Conv1dExtendable(nn.Conv1d):
    def __init__(self, *args, fixed_feature_count=False, **kwargs):
        super(Conv1dExtendable, self).__init__(*args, **kwargs)
        self.init_ncc()
        self.input_tied_modules = [] # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = [] # modules whose output size has to be compatible this this modules output
        self.current_ncc = None
        self.fixed_feature_count=fixed_feature_count
        #self.feature_lookup = list(range(self.out_channels))

    def init_ncc(self):
        self.register_buffer("t0_weight", self.weight.data.clone())
        self.register_buffer("start_ncc", torch.zeros(self.out_channels))
        #self.t0_weight = self.weight.clone()

        #self.start_ncc = Variable(torch.zeros(self.out_channels))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w_0 = self.t0_weight.view(self.weight.size(0), -1)  # size: (G, F*J)
        mean_0 = torch.mean(w_0, dim=1).expand_as(w_0)
        t0_factor = w_0 - mean_0
        t0_norm = torch.norm(w_0, p=2, dim=1)

        w = self.weight.data.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)

        # If there is only one input channel, no sensible ncc can be calculated, return instead the ratio of the norms
        if self.in_channels == 1 & sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(t0_norm, 2)
            ncc = (ncc - self.start_ncc).squeeze()
            self.current_ncc = ncc
            return ncc

        mean = torch.mean(w, dim=1).expand_as(w)
        t_factor = w - mean
        h_product = t0_factor * t_factor
        covariance = torch.sum(h_product, dim=1) #/ (w.size(1)-1)

        denominator = t0_norm * t_norm + 0.05 # add a relatively small constant to avoid uncontrolled expansion for small weights

        ncc = covariance / denominator
        ncc = (ncc - self.start_ncc).squeeze()
        self.current_ncc = ncc
        #self.feature_lookup = list(range(self.out_channels))

        return ncc

    def split_feature(self, feature_number):
        '''
        Use this method as interface!

        :param feature_number:
        :return:
        '''
        if self.fixed_feature_count:
            return self.current_ncc

        #in_channel_number = self.feature_lookup.index(feature_number)

        new_ncc = self._split_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            dep._split_input_channel(channel_number=feature_number)
        for dep in self.output_tied_modules:
            dep._split_output_channel(channel_number=feature_number)

        #self.feature_lookup.insert(feature_number+1, -1)
        return new_ncc


    def prune_feature(self, feature_number):
        if self.fixed_feature_count:
            return self.current_ncc

        #in_channel_number = self.feature_lookup.index(feature_number)

        new_ncc = self.prune_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            dep.prune_input_channel(channel_number=feature_number)
        for dep in self.output_tied_modules:
            dep.prune_output_channel(channel_number=feature_number)

        #self.feature_lookup.remove(feature_number)
        return new_ncc


    def split_features(self, threshold):
        ncc = self.normalized_cross_correlation()
        for i, ncc_value in enumerate(ncc):
            if ncc_value < threshold:
                print("ncc value for feature ", i, ": ", ncc_value)
                self.split_feature(i)

    def _split_output_channel(self, channel_number):
        '''
        Split one output channel (a feature) in two, but retain the same summed value

        :param channel_number: The number of the channel that will be split
        '''

        # weight tensor: (out_channels, in_channels, kernel_size)
        #channel_number = self.feature_lookup.index(channel_number)
        self.out_channels += 1

        original_weight = self.weight.data
        #stdv = 1.86603
        stdv = 0.5
        split_positions = torch.zeros(self.in_channels, self.kernel_size[0]).uniform_(-stdv, stdv) + 0.5  # uniform distributin with mean 0.5 and expected absolute value of 1
        #split_positions = 2 * torch.rand(self.in_channels, self.kernel_size[0])
        slice = original_weight[channel_number, :, :]
        original_weight[channel_number, :, :] = slice * split_positions
        slice = slice * (stdv - split_positions)
        new_weight = insert_slice(original_weight, slice, dim=0, at_index=channel_number+1)

        if self.bias is not None:
            original_bias = self.bias.data
            new_bias = insert_slice(original_bias, original_bias[channel_number:channel_number+1], dim=0, at_index=channel_number+1)
            self.bias = Parameter(new_bias)

        self.weight = Parameter(new_weight)

        # update persistent values
        self.t0_weight[channel_number, :, :] = self.weight.data[channel_number, :, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight.data[channel_number + 1, :, :], dim=0, at_index=channel_number + 1)
        self.start_ncc[channel_number] = 0
        self.start_ncc = insert_slice(self.start_ncc, torch.zeros(1), dim=0, at_index=channel_number + 1)
        ncc = self.normalized_cross_correlation()
        self.start_ncc[channel_number:channel_number + 2] = ncc[channel_number:channel_number + 2]

        return self.normalized_cross_correlation()

    def prune_output_channel(self, channel_number):
        #channel_number = self.feature_lookup.index(channel_number)

        self.out_channels -= 1
        new_weight = remove_slice(self.weight.data, dim=0, at_index=channel_number)
        self.weight = Parameter(new_weight)

        if self.bias is not None:
            self.bias = Parameter(remove_slice(self.bias.data, dim=0, at_index=channel_number))

        # update persistent values
        self.t0_weight = remove_slice(self.t0_weight, dim=0, at_index=channel_number)
        self.start_ncc = remove_slice(self.start_ncc, dim=0, at_index=channel_number)

        return self.normalized_cross_correlation()

    def _split_input_channel(self, channel_number):

        if channel_number > self.in_channels:
            print("cannot split in channel ", channel_number)
            return

        self.in_channels += 1
        original_weight = self.weight.data
        duplicated_slice = original_weight[:, channel_number, :]
        original_weight[: ,channel_number, :] = duplicated_slice
        new_weight = insert_slice(original_weight, duplicated_slice, dim=1, at_index=channel_number+1)

        self.weight = Parameter(new_weight)

        # update persistent values
        self.t0_weight[:, channel_number, :] = self.weight.data[:, channel_number, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight.data[:, channel_number + 1, :], dim=1,
                                      at_index=channel_number + 1)

        return self.normalized_cross_correlation()

    def prune_input_channel(self, channel_number):
        self.in_channels += -1
        self.weight = Parameter(remove_slice(self.weight.data, dim=1, at_index=channel_number))

        # update persistent values
        self.t0_weight = remove_slice(self.t0_weight, dim=1, at_index=channel_number)

        return self.normalized_cross_correlation()

    def forward(self, input):
        return nn.Conv1d.forward(self, input)

class Conv2dExtendable(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2dExtendable, self).__init__(*args, **kwargs)
        self.init_ncc()
        self.input_tied_modules = [] # modules whose input is sensitive to the output size of this module
        self.output_tied_modules = [] # modules whose output size has to be compatible this this modules output
        self.current_ncc = None

    def init_ncc(self):
        self.t0_weight = self.weight.clone()

        self.start_ncc = Variable(torch.zeros(self.out_channels))
        self.start_ncc = self.normalized_cross_correlation()

    def normalized_cross_correlation(self):
        w_0 = self.t0_weight.view(self.weight.size(0), -1)  # size: (G, F*J*K)
        mean_0 = torch.mean(w_0, dim=1).expand_as(w_0)
        t0_factor = w_0 - mean_0
        t0_norm = torch.norm(w_0, p=2, dim=1)

        w = self.weight.view(self.weight.size(0), -1)
        t_norm = torch.norm(w, p=2, dim=1)

        # If there is only one input channel, no sensible ncc can be calculated, return instead the ratio of the norms
        if self.in_channels == 1 & sum(self.kernel_size) == 1:
            ncc = w.squeeze() / torch.norm(t0_norm, 2)
            ncc = ncc - self.start_ncc
            self.current_ncc = ncc
            return ncc

        mean = torch.mean(w, dim=1).expand_as(w)
        t_factor = w - mean
        h_product = t0_factor * t_factor
        covariance = torch.sum(h_product, dim=1) #/ (w.size(1)-1)

        denominator = t0_norm * t_norm + 0.05 # add a relatively small constant to avoid uncontrolled expansion for small weights

        ncc = covariance / denominator
        ncc = ncc - self.start_ncc
        self.current_ncc = ncc

        return ncc

    def split_feature(self, feature_number):
        '''
        Use this method as interface!

        :param feature_number:
        :return:
        '''
        self._split_output_channel(channel_number=feature_number)
        for dep in self.input_tied_modules:
            dep._split_input_channel(channel_number=feature_number)
        for dep in self.output_tied_modules:
            dep._split_output_channel(channel_number=feature_number)

    def split_features(self, threshold):
        ncc = self.normalized_cross_correlation()
        for i, ncc_value in enumerate(ncc):
            if ncc_value < threshold:
                print("ncc value for feature ", i, ": ", ncc_value)
                self.split_feature(i)

    def _split_output_channel(self, channel_number):
        '''
        Split one output channel (a feature) in two, but retain the same summed value

        :param channel_number: The number of the channel that will be split
        '''

        # weight tensor: (out_channels, in_channels, kernel_size)
        self.out_channels += 1

        original_weight = self.weight.data
        stdv = 1.86603
        split_positions = torch.zeros(self.in_channels, self.kernel_size[0]).uniform_(-stdv, stdv) + 0.5 # uniform distributin iwht mean 0.5 and expected absolute value of 1
        #split_positions = 2 * torch.rand(self.in_channels, self.kernel_size[0])
        slice = original_weight[channel_number, :, :, :]
        original_weight[channel_number, :, :] = slice * split_positions
        slice = slice * (0.5 - split_positions) + 0.5
        new_weight = insert_slice(original_weight, slice, dim=0, at_index=channel_number+1)

        if self.bias is not None:
            original_bias = self.bias.data
            new_bias = insert_slice(original_bias, original_bias[channel_number:channel_number+1], dim=0, at_index=channel_number+1)
            self.bias = Parameter(new_bias)

        self.weight = Parameter(new_weight)

        # update persistent values
        self.t0_weight[channel_number, :, :] = self.weight[channel_number, :, :, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight[channel_number + 1, :, :, :], dim=0, at_index=channel_number + 1)
        self.start_ncc[channel_number] = torch.zeros(1)
        self.start_ncc = insert_slice(self.start_ncc, torch.zeros(1), dim=0, at_index=channel_number + 1)
        ncc = self.normalized_cross_correlation()
        self.start_ncc[channel_number:channel_number + 2] = ncc[channel_number:channel_number + 2]

    def _split_input_channel(self, channel_number):

        if channel_number > self.in_channels:
            print("cannot split in channel ", channel_number)
            return

        self.in_channels += 1
        original_weight = self.weight.data
        duplicated_slice = original_weight[:, channel_number, :, :]
        original_weight[: ,channel_number, :, :] = duplicated_slice
        new_weight = insert_slice(original_weight, duplicated_slice, dim=1, at_index=channel_number+1)

        self.weight = Parameter(new_weight)

        # update persistent values
        self.t0_weight[:, channel_number, :, :] = self.weight[:, channel_number, :, :]
        self.t0_weight = insert_slice(self.t0_weight, self.weight[:, channel_number + 1, :, :], dim=1,
                                      at_index=channel_number + 1)


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