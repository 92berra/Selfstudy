import torch
import numpy as np

# 산술연산
# 두 텐서 간의 행렬 곱(matrix multiplication)
# tensor.T 는 텐서의 전치(transpose)를 반환
tensor = torch.ones(4, 4)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)