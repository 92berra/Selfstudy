import torch 
import numpy as np

tensor = torch.rand(3,4)

if torch.cuda.is_available():
    tensor = tensor.to("cuda")
    print("Cuda is available")
else:
    print("Only CPU")


# NumPy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)