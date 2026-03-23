import torch
import torch.nn as nn
w=nn.Softmax(dim=1)
a=torch.Tensor([[0.5]])
b=w(a)
print(b)