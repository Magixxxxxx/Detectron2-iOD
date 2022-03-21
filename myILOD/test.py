import numpy as np
import torch
torch.load()
def meanstr(a):
    a = np.array([float(i) for i in a.split(',')]) 
    return np.mean(a)

b = '76.3322,75.5895,70.3832,50.6659,40.4493,64.4941,76.7894,73.4659,14.6185,66.6397,35.7321,72.9624,78.6670,75.6609,61.2704'

print(meanstr(b))

