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

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--extend-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before extending the architecture')
parser.add_argument('--extend-threshold', type=float, default=0.03, metavar='T',
                    help='threshold for architecture expansion')
parser.add_argument('--prune-threshold', type=float, default=0.0001, metavar='T',
                    help='threshold for architecture expansion')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        # default channel counts: 1-10-20 320-50-10
        super(Net, self).__init__()
        # self.fc1 = Conv1dExtendable(784, 32, kernel_size=1, bias=False)
        # self.fc2 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc3 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc4 = Conv1dExtendable(32, 32, kernel_size=1, bias=False)
        # self.fc5 = Conv1dExtendable(32, 10, kernel_size=1, bias=True, fixed_feature_count=True)

        self.fc1 = Conv1dExtendable(784, 4, kernel_size=1, bias=False)
        self.fc2 = Conv1dExtendable(4, 4, kernel_size=1, bias=False)
        self.fc3 = Conv1dExtendable(4, 4, kernel_size=1, bias=False)
        self.fc4 = Conv1dExtendable(4, 4, kernel_size=1, bias=False)
        self.fc5 = Conv1dExtendable(4, 4, kernel_size=1, bias=False)
        self.fc6 = Conv1dExtendable(4, 4, kernel_size=1, bias=False)
        self.fc7 = Conv1dExtendable(4, 10, kernel_size=1, bias=True, fixed_feature_count=True)

        self.fc1.input_tied_modules = [self.fc2]
        self.fc2.input_tied_modules = [self.fc3]
        self.fc3.input_tied_modules = [self.fc4]
        self.fc4.input_tied_modules = [self.fc5]
        self.fc5.input_tied_modules = [self.fc6]
        self.fc6.input_tied_modules = [self.fc7]

        self.logger = Logger('./logs')


    def forward(self, x):
        x = x.view(-1, 28*28, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x).squeeze()
        return F.log_softmax(x)

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def log_to_tensor_board(self, batch_idx, loss):
        # TensorBoard logging

        # loss
        self.logger.scalar_summary("loss", loss, batch_idx)

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
                self.logger.histo_summary(tag + '/ncc', ncc.data.cpu().numpy(), batch_idx)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        if batch_idx % args.extend_interval == 0:

            print("test result before expanding: ")
            test(epoch)

            #model.log_to_tensor_board(batch_idx, loss.data[0])

            for name, module in model.named_modules():
                if type(module) is Conv1dExtendable:
                    ncc = module.normalized_cross_correlation()

            splitted = False

            for name, module in model.named_modules():
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
                        if (abs(weighted_value) > args.extend_threshold) & (model.parameter_count() < 25000):
                            print("in ", name, ", split feature number ", feature_number + offset)
                            module.split_feature(feature_number=feature_number + offset)
                            all_modules = [module] + module.output_tied_modules
                            [m.normalized_cross_correlation() for m in all_modules]
                            splitted = True
                            offset += 1
                        if (abs(weighted_value) < args.prune_threshold) & (weighted_value != 0):
                            if feature_number >= module.out_channels:
                                continue
                            print("in ", name, ", prune feature number ", feature_number + offset)
                            module.prune_feature(feature_number=feature_number + offset)
                            all_modules = [module] + module.output_tied_modules
                            [m.normalized_cross_correlation() for m in all_modules]
                            splitted = True
                            offset += -1

            if splitted:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
                print("new parameter count: ", model.parameter_count())
                print("test result after expanding: ")
                test(epoch)
            else:
                print("No feature changed enough to split")

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

    return optimizer


def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    optimizer = train(epoch, optimizer)
    test(epoch)
