import torch
import os
from models import FC_Net, Conv_Net
from training import MNIST_Optimizer
from expanding_modules import Conv1dExtendable
from logger import Logger

#model = FC_Net(layer_sizes=[784, 4, 4, 10])
model = Conv_Net()
trainer = MNIST_Optimizer(model,
                          epochs=20,
                          extend_interval=300,
                          log_interval=300,
                          extend_threshold=0.05,
                          prune_threshold=0.0,
                          extension_rate=6,
                          pruning_rate=2,
                          lr=0.0003,
                          weight_decay=0)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extension_rate) + "_prune_" + str(trainer.pruning_rate) + "_interval_" + str(trainer.extend_interval) + "_lr_" + str(trainer.lr) + "_fixed"
folder = "./experiments/ConvNetExpansion/" + name

logger=Logger(folder)
model.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")