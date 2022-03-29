from numpy import mean
import numpy
import torch

# print('ok')
# a = '76.8957,72.8880,64.3620,57.2994,58.2126,78.4685,75.5601,78.3353,29.5870,65.0724,40.7605,63.8765,67.1991,64.5589,72.2478,31.5087,60.3115,50.5364,68.6831,57.7613'
# a = [float(i) for i in a.split(',')]
# print(mean(a[:10]))
# print(mean(a[10:]))

a = [numpy.random.beta(8,8) for _ in range(100)]
print(a)