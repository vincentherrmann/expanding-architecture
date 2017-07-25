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

# trainer.extend_threshold = 10
# trainer.prune_threshold = 0.12
# trainer.extend_and_prune(0)

# prune_count = 20
#
# print("\n prune the ", prune_count, " most important features seperately\n ")
# for name, module in model.named_modules():
#     if type(module) is Conv1dExtendable:
#         ncc = module.normalized_cross_correlation().squeeze()
#     else:
#         continue
#
#     (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
#     (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=True)
#     feature_numbers = sorted_idx[0:prune_count]
#     for i, feature_number in enumerate(feature_numbers):
#         #print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
#         module.prune_feature(feature_number=feature_number)
#     trainer.test(0)
#
#     model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
#     state_dict = torch.load(file)
#     model.load_state_dict(state_dict)
#     trainer.model = model
#
# print("\noriginal test run")
# trainer.test(0)
#
# print("\n prune the ", prune_count, " least important features seperately\n")
# for name, module in model.named_modules():
#     if type(module) is Conv1dExtendable:
#         ncc = module.normalized_cross_correlation().squeeze()
#     else:
#         continue
#
#     (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
#     (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
#     feature_numbers = sorted_idx[0:prune_count]
#     for i, feature_number in enumerate(feature_numbers):
#         #print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
#         module.prune_feature(feature_number=feature_number)
#     trainer.test(0)
#
#     model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
#     state_dict = torch.load(file)
#     model.load_state_dict(state_dict)
#     trainer.model = model

def prune_least_important_features(prune_count):
    model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trainer.model = model

    print("\n prune the ", prune_count, " least important features for all layers at once\n")
    for name, module in model.named_modules():
        if type(module) is Conv1dExtendable:
            ncc = module.normalized_cross_correlation()
        else:
            continue

        for _ in range(prune_count):
            (min_val, min_idx) = torch.min(torch.abs(ncc), 0)
            print("in ", name, ", prune feature number ", min_idx[0], " with ncc ", min_val[0])
            ncc = module.prune_feature(feature_number=min_idx[0])

        # (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
        # feature_numbers = sorted_idx[0:prune_count]
        # for i, feature_number in enumerate(feature_numbers):
        #     print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
        #     module.prune_feature(feature_number=feature_number)

    test_loss, correct = trainer.test()
    print("change of test loss: ", (test_loss / orig_test_loss))
    print("")


# for prune_count in range(1, 11):
#     print("\n prune the ", prune_count, " most important features for all layers at once\n")
#     for name, module in model.named_modules():
#         if type(module) is Conv1dExtendable:
#             ncc = module.normalized_cross_correlation().squeeze()
#         else:
#             continue
#
#         (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
#         (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=True)
#         feature_numbers = sorted_idx[0:prune_count]
#         for i, feature_number in enumerate(feature_numbers):
#             # print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
#             module.prune_feature(feature_number=feature_number)
#
#     trainer.test(0)
#
#     model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
#     state_dict = torch.load(file)
#     model.load_state_dict(state_dict)
#     trainer.model = model
#
#     print("\n prune the ", prune_count, " least important features for all layers at once\n")
#     for name, module in model.named_modules():
#         if type(module) is Conv1dExtendable:
#             ncc = module.normalized_cross_correlation().squeeze()
#         else:
#             continue
#
#         (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
#         (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
#         feature_numbers = sorted_idx[0:prune_count]
#         for i, feature_number in enumerate(feature_numbers):
#             # print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
#             module.prune_feature(feature_number=feature_number)
#
#     trainer.test(0)
#
#     model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
#     state_dict = torch.load(file)
#     model.load_state_dict(state_dict)
#     trainer.model = model


def pruning_sweep_least_important(prune_count):
    model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trainer.model = model

    print("Prune the least important feature of each layer a given number of times without reloading the model")
    for i in range(prune_count):
        print("reduce least important feature (", i, ")")
        for name, module in model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
            else:
                continue

            (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
            (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=False)
            feature_numbers = sorted_idx[0:1]
            for i, feature_number in enumerate(feature_numbers):
                print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
                module.prune_feature(feature_number=feature_number)

        test_loss, correct = trainer.test()
        print("change of test loss: ", (test_loss/orig_test_loss))
        print("")

def pruning_sweep_most_important(prune_count):
    model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
    state_dict = torch.load(file)
    model.load_state_dict(state_dict)
    trainer.model = model

    print("Prune the most important feature of each layer a given number of times without reloading the model")
    for i in range(prune_count):
        print("reduce most important feature (", i, ")")
        for name, module in model.named_modules():
            if type(module) is Conv1dExtendable:
                ncc = module.normalized_cross_correlation()
            else:
                continue

            (max_val, max_idx) = torch.max(torch.abs(ncc), 0)
            (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=True)
            feature_numbers = sorted_idx[0:1]
            for i, feature_number in enumerate(feature_numbers):
                print("in ", name, ", prune feature number ", feature_number, " with ncc ", sorted_val[i])
                module.prune_feature(feature_number=feature_number)

        test_loss = trainer.test()
        print("change of test loss: ", (test_loss/orig_test_loss))
        print("")

for pc in range(1, 11):
    prune_least_important_features(prune_count=pc)




