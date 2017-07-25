from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math as math
from torchvision import datasets, transforms
from torch.autograd import Variable
from expanding_modules import Conv1dExtendable, Conv2dExtendable, MutatingModule
from logger import Logger


class Expander:
    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger
        self.splitted_ncc_avg = 0

    def prune_n_features(self, n):
        if n == 0:
            return

        for name, module in self.model.named_modules():
            if type(module) is Conv1dExtendable or type(module) is Conv2dExtendable:
                ncc = module.normalized_cross_correlation()

        for _ in range(n):
            min_module = None
            min_name = ""
            min_val = 1000
            min_idx = 0
            ncc_avg = 0
            for name, module in self.model.named_modules():
                if not isinstance(module, MutatingModule):
                    continue

                if module.fixed_feature_count:
                    continue

                this_val, this_idx = torch.min(torch.abs(module.current_ncc), 0)  # prune least important features
                if this_val[0] == 0:
                    continue
                if this_val[0] < min_val:
                    min_val = this_val[0]
                    min_idx = this_idx[0]
                    min_module = module
                    min_name = name
            if min_module is not None:
                min_module.prune_feature(feature_number=min_idx)
                print("in ", min_name, ", prune feature number ", min_idx, " with ncc ", min_val)
                ncc_avg += min_val

        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.splitted_ncc_avg = ncc_avg / n

    def expand_n_features(self, n):
        if n == 0:
            return

        for name, module in self.model.named_modules():
            if isinstance(module, MutatingModule):
                ncc = module.normalized_cross_correlation()

        for _ in range(n):
            max_module = None
            max_name = ""
            max_val = 0
            #max_val = 100
            max_idx = 0
            ncc_avg = 0
            for name, module in self.model.named_modules():
                if not isinstance(module, MutatingModule):
                    continue

                if module.fixed_feature_count:
                    continue

                this_val, this_idx = torch.max(torch.abs(module.current_ncc), 0) # extend most important features
                #this_val, this_idx = torch.min(torch.abs(module.current_ncc), 0)  # extend most important features
                #this_val, this_idx = torch.max(torch.rand(module.out_channels), 0) # extend random feature
                if this_val[0] > max_val:
                #if this_val[0] < max_val:
                    max_val = this_val[0]
                    max_idx = this_idx[0]
                    max_module = module
                    max_name = name
            if max_module is not None:
                max_module.split_feature(feature_number=max_idx)
                print("in ", max_name, ", split feature number ", max_idx, " with ncc ", max_val)
                ncc_avg += max_val

        print("new parameter count: ", self.parameter_count())
        #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.splitted_ncc_avg = ncc_avg / n

    def expand_and_prune_with_thresholds(self, expand_thr, prune_thr):

        for name, module in self.model.named_modules():
            if isinstance(module, MutatingModule):
                ncc = module.normalized_cross_correlation()

        splitted = False

        for name, module in self.model.named_modules():
            if not isinstance(module, MutatingModule):
                continue

            if module.fixed_feature_count:
                continue

            # print large weights:
            (max_val, max_idx) = torch.max(torch.abs(module.weight.view(-1)), dim=0)
            if torch.max(max_val.data) > 2:
                idx = np.unravel_index(max_idx.data[0], module.weight.size())
                print("maximum weight at index ", idx, " with value ", max_val.data[0])

            if len(module.output_tied_modules) > 0:
                all_nccs = [module.current_ncc] + [m.current_ncc for m in module.output_tied_modules]
                ncc_tensor = torch.abs(torch.stack(all_nccs))
                ncc = torch.mean(ncc_tensor, dim=0)
            else:
                ncc = module.current_ncc

            offset = 0

            feature_number = 0
            while feature_number < ncc.size(0):
                # weighted_value = ncc[feature_number] / (math.sqrt(module.in_channels))
                # weighted_value = ncc[feature_number] / (math.log(module.in_channels)+1)
                weighted_value = ncc[feature_number]
                if abs(weighted_value) > expand_thr:
                    print("in ", name, ", split feature number ", feature_number + offset, ", ncc: ", weighted_value)
                    ncc = module.split_feature(feature_number=feature_number + offset)
                    all_modules = [module] + module.output_tied_modules
                    [m.normalized_cross_correlation() for m in all_modules]
                    splitted = True
                    self.allow_pruning = True
                    feature_number = 0
                    continue
                if abs(weighted_value) < prune_thr and weighted_value != 0 and self.allow_pruning:
                    if feature_number >= module.out_channels:
                        continue
                    print("in ", name, ", prune feature number ", feature_number)
                    ncc = module.prune_feature(feature_number=feature_number + offset)
                    all_modules = [module] + module.output_tied_modules
                    [m.normalized_cross_correlation() for m in all_modules]
                    splitted = True
                    feature_number = 0
                    continue
                feature_number += 1

        if splitted:
            # self.optimizer = optim.SGD(self.model.parameters(),
            #                            lr=self.lr,
            #                            momentum=self.momentum,
            #                            weight_decay=self.weight_decay)
            #self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            print("new parameter count: ", self.parameter_count())
            # print("test result after expanding: ")
            # self.test(epoch)
        else:
            print("No feature changed enough to split")

        return splitted

    def log_to_tensor_board(self, batch_idx, loss, correct_results, ncc_avg):
        if self.logger is None:
            return
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
        for tag, value, in self.model.named_parameters():
            tag = tag.replace('.', '/')
            self.logger.histo_summary(tag, value.data.cpu().numpy(), batch_idx)
            if value.grad is not None:
                self.logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), batch_idx)

        # normalized cross correlation
        for tag, module in self.model.named_modules():
            tag = tag.replace('.', '/')
            if type(module) is Conv1dExtendable or type(module) is Conv2dExtendable:
                ncc = module.normalized_cross_correlation()
                self.logger.histo_summary(tag + '/ncc', ncc.cpu().numpy(), batch_idx)

        # model size histo
        channels = [1]
        for tag, module in self.model.named_modules():
            if isinstance(module, MutatingModule):
                channels.append(module.out_channels)
        self.logger.list_summary("model_shape", channels, batch_idx)

    def parameter_count(self):
        par = list(self.model.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

class OptimizerMNIST(Expander):
    def __init__(self, model,
                 epochs=10,
                 report_interval=10,
                 batch_size=64,
                 cuda=False,
                 expand_interval=0,
                 log_interval=0,
                 lr=0.01,
                 momentum=0.5,
                 weight_decay=0,
                 expand_threshold=0.01,
                 prune_threshold=0.0001,
                 expand_rate=0,
                 prune_rate=0):
        super(OptimizerMNIST, self).__init__(model=model)
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cuda = cuda
        self.report_interval = report_interval
        self.log_interval = log_interval
        self.expand_interval = expand_interval
        self.expand_threshold = expand_threshold
        self.prune_threshold = prune_threshold
        self.expand_rate = expand_rate
        self.prune_rate = prune_rate
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=batch_size, shuffle=True)
        self.allow_pruning = False

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            self.test()

    def _train_one_epoch(self, epoch):
        self.model.train()
        offset = int((epoch-1) * len(self.train_loader.dataset) / self.train_loader.batch_size)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            idx = batch_idx + offset
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            if self.log_interval != 0 and idx % self.log_interval == 0:
                test_loss, test_correct = self.test()
                self.log_to_tensor_board(idx, test_loss, test_correct, self.splitted_ncc_avg)
                self.model.train()

            if self.expand_interval!= 0 and idx % self.expand_interval == 0:
                # splitted = self.expand_and_prune_with_thresholds(expand_thr=self.expand_threshold,
                #                                                  prune_thr=self.prune_threshold)
                self.prune_n_features(self.prune_rate)
                self.expand_n_features(self.expand_rate)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            self.optimizer.step()

            if idx % self.report_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data[0]))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(self.test_loader)  # loss function already averages over batch size
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return test_loss, (100 * correct / len(self.test_loader.dataset))


class OptimizerCIFAR10(Expander):
    def __init__(self, model,
                 epochs=10,
                 report_interval=10,
                 batch_size=64,
                 cuda=False,
                 expand_interval=0,
                 log_interval=0,
                 lr=0.01,
                 momentum=0.5,
                 weight_decay=0,
                 expand_threshold=0.01,
                 prune_threshold=0.0001,
                 expand_rate=0,
                 prune_rate=0):
        super(OptimizerCIFAR10, self).__init__(model=model)
        self.epochs = epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        #self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.cuda = cuda
        self.report_interval = report_interval
        self.log_interval = log_interval
        self.expand_interval = expand_interval
        self.expand_threshold = expand_threshold
        self.prune_threshold = prune_threshold
        self.expand_rate = expand_rate
        self.prune_rate = prune_rate

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data',
                                                                         train=True,
                                                                         download=True,
                                                                         transform=transform_train),
                                                        batch_size=batch_size,
                                                        shuffle=True)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('../data',
                                                                        train=False,
                                                                        download=True,
                                                                        transform=transform_test),
                                                       batch_size=batch_size,
                                                       shuffle=True)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.allow_pruning = False
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            self.test()

    def _train_one_epoch(self, epoch):
        self.model.train()
        offset = int((epoch-1) * len(self.train_loader.dataset) / self.train_loader.batch_size)
        for batch_idx, (data, target) in enumerate(self.train_loader):
            idx = batch_idx + offset
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()

            if self.log_interval != 0 and idx % self.log_interval == 0:
                test_loss, test_correct = self.test()
                self.log_to_tensor_board(idx, test_loss, test_correct, self.splitted_ncc_avg)
                self.model.train()

            if self.expand_interval!= 0 and idx % self.expand_interval == 0:
                # splitted = self.expand_and_prune_with_thresholds(expand_thr=self.expand_threshold,
                #                                                  prune_thr=self.prune_threshold)
                self.prune_n_features(self.prune_rate)
                self.expand_n_features(self.expand_rate)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            self.optimizer.step()

            if idx % self.report_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data[0]))

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = self.model(data)
            test_loss += self.criterion(output, target).data[0]
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()

        test_loss = test_loss
        test_loss /= len(self.test_loader)  # loss function already averages over batch size
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))

        return test_loss, (100 * correct / len(self.test_loader.dataset))


