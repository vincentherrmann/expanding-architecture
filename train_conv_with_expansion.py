import torch
import os
from models import FC_Net, Conv_Net
from training import OptimizerMNIST
from expanding_modules import Conv1dExtendable
from logger import Logger

#model = FC_Net(layer_sizes=[784, 4, 4, 10])
model = Conv_Net(conv=[1, 4, 4], fc=[4, 10])
trainer = OptimizerMNIST(model,
                         epochs=20,
                         expand_interval=100,
                         log_interval=400,
                         expand_rate=1,
                         prune_rate=0,
                         lr=0.0003,
                         weight_decay=0)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_expand_" + str(trainer.expand_rate) + "_prune_" + str(trainer.prune_rate) + "_interval_" + str(trainer.expand_interval) + "_lr_" + str(trainer.lr) + "_selu"
folder = "./experiments/ConvNetExpansion/" + name

logger=Logger(folder)
trainer.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")