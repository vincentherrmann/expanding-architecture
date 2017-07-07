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
from expanding_modules import Conv1dExtendable, Conv2dExtendable
from logger import Logger

class MNIST_Optimizer:
    def __init__(self, model,
                 epochs=10,
                 report_interval=10,
                 batch_size=64,
                 cuda=False,
                 extend_interval=0,
                 log_interval=10,
                 lr=0.01,
                 momentum=0.5,
                 extend_threshold=0.01,
                 prune_threshold=0.0001):
        self.model = model
        self.epochs = epochs
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.lr = lr
        self.momentum = momentum
        self.cuda = cuda
        self.report_interval = report_interval
        self.log_interval = log_interval
        self.extend_interval = extend_interval
        self.extend_threshold = extend_threshold
        self.prune_threshold = prune_threshold
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

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self._train_one_epoch(epoch)
            self.test(epoch)

    def _train_one_epoch(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            if (self.extend_interval != 0) and (batch_idx % self.extend_interval == 0):
                self.extend_and_prune(epoch)

            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data[0]))

    def extend_and_prune(self, epoch):
        #print("test result before expanding: ")
        #self.test(epoch)

        # model.log_to_tensor_board(batch_idx, loss.data[0])

        for name, module in self.model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()

        splitted = False

        for name, module in self.model.named_modules():
            if type(module) is Conv1dExtendable:
                if len(module.output_tied_modules) > 0:
                    all_nccs = [module.current_ncc] + [m.current_ncc for m in module.output_tied_modules]
                    ncc_tensor = torch.abs(torch.stack(all_nccs))
                    ncc = torch.mean(ncc_tensor, dim=0)
                else:
                    ncc = module.current_ncc

                offset = 0

                if module.fixed_feature_count:
                    continue

                for feature_number, value in enumerate(ncc):
                    weighted_value = value.data[0] / math.log(module.in_channels)
                    if (abs(weighted_value) > self.extend_threshold) & (self.model.parameter_count() < 25000):
                        print("in ", name, ", split feature number ", feature_number + offset)
                        module.split_feature(feature_number=feature_number + offset)
                        all_modules = [module] + module.output_tied_modules
                        [m.normalized_cross_correlation() for m in all_modules]
                        splitted = True
                        offset += 1
                    if (abs(weighted_value) < self.prune_threshold) & (weighted_value != 0):
                        if feature_number >= module.out_channels:
                            continue
                        print("in ", name, ", prune feature number ", feature_number)
                        module.prune_feature(feature_number=feature_number + offset)
                        all_modules = [module] + module.output_tied_modules
                        [m.normalized_cross_correlation() for m in all_modules]
                        splitted = True
                        offset += -1

        if splitted:
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            print("new parameter count: ", self.model.parameter_count())
            print("test result after expanding: ")
            self.test(epoch)
        else:
            print("No feature changed enough to split")

    def test(self, epoch=0):
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

        return test_loss
