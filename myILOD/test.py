import detectron2
import os

import torch

a = torch.tensor([[1,2]])
b = torch.tensor([[2,3]])


print(torch.cat((a,b), 0))