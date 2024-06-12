import torch
import numpy as np

# 3 다른 텐서로부터 생성
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

