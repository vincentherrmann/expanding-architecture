import torch
import argparse
from vgg import VGG
from training import OptimizerMNIST, OptimizerCIFAR10
from logger import Logger

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--epochs', default=20, type=int, help='training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--exp_itv', default=200, type=int, help='expand interval')
parser.add_argument('--exp_rate', default=4, type=int, help='expand rate')
parser.add_argument('--pr_rate', default=0, type=int, help='prune rate')

args = parser.parse_args()

print(args)

cuda = torch.cuda.is_available()

model = VGG()
if cuda:
    model.cuda()

trainer = OptimizerCIFAR10(model,
                           epochs=args.epochs,
                           expand_interval=args.exp_itv,
                           log_interval=400,
                           expand_rate=args.exp_rate,
                           prune_rate=args.pr_rate,
                           lr=args.lr,
                           weight_decay=0,
                           cuda=cuda)

#print("exp_rate: ", trainer.expand_rate)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_expand_" + str(trainer.expand_rate) + "_prune_" + str(trainer.prune_rate) + "_interval_" + str(trainer.expand_interval) + "_lr_" + str(trainer.lr) + "test"
folder = "./experiments/CIFAR10/" + name

logger=Logger(folder)
trainer.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")