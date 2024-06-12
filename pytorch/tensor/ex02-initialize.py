import torch
import numpy as np

# 2 NumPy 배열로부터 생성
data = [[1, 2],[3, 4]]
np_array = np.array(data)
x_np = torch.from_numpy(np_array)