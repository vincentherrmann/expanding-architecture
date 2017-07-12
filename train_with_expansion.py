import torch
import os
from models import FC_Net
from training import MNIST_Optimizer
from expanding_modules import Conv1dExtendable
from logger import Logger

model = FC_Net(layer_sizes=[784, 4, 4, 4, 4, 10])
trainer = MNIST_Optimizer(model,
                          epochs=20,
                          extend_interval=200,
                          log_interval=200,
                          extend_threshold=0.002,
                          prune_threshold=0.0,
                          weight_decay=0)

name = str(model.layer_count) + "_layers_" + "_extend_" + str(trainer.extend_threshold) + "_prune_" + str(trainer.prune_threshold) + "_stdv_1.5"
folder = "./experiments/TensorBoardLogs/" + name

logger=Logger(folder)
model.logger = logger

trainer.train()
torch.save(model.state_dict(), folder + "/model_trained_for_" + str(trainer.epochs) + "_epochs")