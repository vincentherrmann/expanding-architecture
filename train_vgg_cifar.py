import torch
import os
from vgg import VGG
from training import OptimizerMNIST, OptimizerCIFAR10
from expanding_modules import Conv1dExtendable
from logger import Logger

model = VGG()
trainer = OptimizerCIFAR10(model,
                           epochs=20,
                           expand_interval=200,
                           log_interval=400,
                           expand_rate=15,
                           prune_rate=0,
                           lr=0.001,
                           weight_decay=0)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_expand_" + str(trainer.expand_rate) + "_prune_" + str(trainer.prune_rate) + "_interval_" + str(trainer.expand_interval) + "_lr_" + str(trainer.lr) + "test"
folder = "./experiments/CIFAR10/" + name

logger=Logger(folder)
trainer.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")