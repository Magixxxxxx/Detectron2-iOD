
import torch

a = torch.tensor([[1.,2.]], requires_grad = True)
b = torch.tensor([[2.,3.]], requires_grad = True)

c = sum(sum(a * b)) - 10.
c.backward()
print(a)
print(b)
print(c)


optimizer = torch.optim.SGD([a,b], lr=0.1)
optimizer.step()


print(a)
print(b)
print(c)

optimizer.step()


print(a)
print(b)
print(c)