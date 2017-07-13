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

def expanding_sweep_most_important(expand_count):
    model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trainer.model = model

    print("Expand the most important feature of each layer a given number of times without reloading the model")
    for i in range(expand_count):
        print("Split most important feature (", i, ")")
        for name, module in model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
            else:
                continue

            (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=True)
            feature_numbers = sorted_idx[0:1]
            for i, feature_number in enumerate(feature_numbers):
                print("in ", name, ", split feature number ", feature_number, " with ncc ", sorted_val[i])
                module.split_feature(feature_number=feature_number)

        test_loss, correct = trainer.test()
        print("change of test loss: ", (test_loss/orig_test_loss))
        print("")


def expanding_sweep_least_important(expand_count):
    model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trainer.model = model

    print("Expand the least important feature of each layer a given number of times without reloading the model")
    for i in range(expand_count):
        print("Split least important feature (", i, ")")
        for name, module in model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
            else:
                continue

            (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
            feature_numbers = sorted_idx[0:1]
            for i, feature_number in enumerate(feature_numbers):
                print("in ", name, ", split feature number ", feature_number, " with ncc ", sorted_val[i])
                module.split_feature(feature_number=feature_number)

        (test_loss, correct) = trainer.test()
        print("change of test loss: ", (test_loss/orig_test_loss))
        print("")


#expanding_sweep_least_important(10)
expanding_sweep_most_important(10)