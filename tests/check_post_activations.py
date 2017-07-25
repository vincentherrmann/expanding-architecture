import torch
import os
from models import FC_Net, Conv_Net
from training import MNIST_Optimizer
from torch.autograd import Variable
from expanding_modules import Conv1dExtendable, Conv2dExtendable

model = FC_Net(layer_sizes=[784, 32, 32, 32, 32, 10])
#model = Conv_Net()
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

test_img = Variable(torch.rand(28, 28))
print(model(test_img))

for name, module in model.named_modules():
    if type(module) is Conv1dExtendable or type(module) is Conv2dExtendable:
        ncc = module.normalized_cross_correlation()
    else:
        continue

    (sorted_val, sorted_idx) = torch.sort(torch.abs(ncc), dim=0, descending=True)
    feature_numbers = sorted_idx[0:1]
    for i, feature_number in enumerate(feature_numbers):
        print("in ", name, ", split feature number ", feature_number, " with ncc ", sorted_val[i])
        module.split_feature(feature_number=feature_number)

print(model(test_img))

