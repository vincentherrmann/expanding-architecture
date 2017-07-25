import torch
import os
from models import FC_Net
from training import OptimizerMNIST
from expanding_modules import Conv1dExtendable
from logger import Logger

model = FC_Net(layer_sizes=[784, 4, 4, 10])
trainer = OptimizerMNIST(model,
                         epochs=20,
                         extend_interval=400,
                         log_interval=400,
                         extend_threshold=0.05,
                         prune_threshold=0.0,
                         extension_rate=8,
                         lr=0.0001,
                         weight_decay=0)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extension_rate) + "_interval_" + str(trainer.extend_interval) + "_lr_" + str(trainer.lr) + "_min"
folder = "./experiments/FixedExpansionRate/" + name

logger=Logger(folder)
trainer.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")