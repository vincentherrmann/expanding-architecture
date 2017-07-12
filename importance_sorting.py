import torch
import os
from models import FC_Net
from training import MNIST_Optimizer
from expanding_modules import Conv1dExtendable

model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
trainer = MNIST_Optimizer(model, epochs=10)

file = "trained_model_784_4x32_10"

if os.path.exists(file):
    print("load model")
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
else:
    print("save model")
    trainer.train()
    torch.save(model.state_dict(), file)

print("initial test run:")
orig_test_loss, correct = trainer.test()

def sort_importance_by_ncc(module_name):
    module = model.seq.__getattr__(module_name)
    ncc = module.normalized_cross_correlation()
    (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
    print("indices of the features sorted by importance:")
    print(sorted_idx)
    return sorted_idx

def sort_importance_by_testing(module_name):
    module = trainer.model.seq.__getattr__(module_name)
    test_results = []
    for feature_number in range(module.out_channels):
        module = trainer.model.seq.__getattr__(module_name)
        module.prune_feature(feature_number)
        res = trainer.test()
        test_results.append(res)

        model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
        state_dict = torch.load(file)
        model.load_state_dict(state_dict)
        trainer.model = model

    tr = torch.FloatTensor(test_results)
    (sorted_val, sorted_idx) = torch.sort(torch.abs(tr), dim=0, descending=False)
    print("indices of the features sorted by importance:")
    print(sorted_idx)
    return sorted_idx

layer = "fc2"
print("layer ", layer, " sorted by ncc:")
ncc_idx = sort_importance_by_ncc(layer).tolist()
print("layer ", layer, " sorted by testing:")
testing_idx = sort_importance_by_testing(layer).tolist()

index_difference = []
for (i, position) in enumerate(ncc_idx):
    index_difference.append(i-testing_idx.index(position))
print("index difference between ncc and testing in increasing importance: ")
print(index_difference)