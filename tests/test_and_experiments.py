import torch
import torch.nn as nn
from scipy.stats import rankdata

# idx = torch.ByteTensor(8, 1).bernoulli_(0.5).long()
# rand1 = torch.FloatTensor(8).uniform_(0, 1)
# rand2 = torch.FloatTensor(8).uniform_(0, 1)
# rand = torch.stack([rand1, rand2], 1)
# print(idx, rand)
# g = torch.gather(rand, 1, idx)
# print(g)

# r = torch.rand(6)
# print(r)
# ranks = rankdata(r.numpy(), 'ordinal')
# print(ranks)

# state_dict = torch.load("./experiments/FixedExpansionRate/3_layers__extend_8_interval_400_lr_0.0001/model_trained_for_20_epochs")
# print(state_dict)

ls = torch.linspace(0, 23, 24)
ns = ls.view(2, 3, 4)
print(ns)
ls = ns.view(-1)
print(ls)

r = torch.rand(3, 4, 5)
print(r)
s = r[1,:]
print(s)

print(s.size())

seq = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )
pass