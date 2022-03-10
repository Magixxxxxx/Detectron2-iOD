
import torch

a = torch.tensor([[1.,2.] ,[1.,3.]], requires_grad = True)
b = torch.tensor([[2.]], requires_grad = True)

c = torch.mul(a, b)
print(c)