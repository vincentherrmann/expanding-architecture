import torch
import os
from fc_models import FC_Splittable
from training import OptimizerMNIST
from expanding_modules import Conv1dExtendable
from logger import Logger
from graphs import find_dependencies

dummy_input = torch.autograd.Variable(torch.rand([28, 28]))
model = FC_Splittable(layer_sizes=[784, 4, 10])
output = model(dummy_input)
find_dependencies(model, output)

trainer = OptimizerMNIST(model,
                         epochs=20,
                         expand_interval=400,
                         log_interval=400,
                         expand_threshold=0.0,
                         prune_threshold=0.0,
                         split_threshold=0.03,
                         expand_rate=4,
                         lr=0.001,
                         weight_decay=0)

#name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_Adam"
name = str(model.layer_count) + "_layers_" + "_expand_" + str(trainer.expand_rate) + "_interval_" + str(trainer.expand_interval) + "_lr_" + str(trainer.lr) + "_min"
folder = "./experiments/FixedExpansionRate/" + name

#logger=Logger(folder)
#trainer.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")