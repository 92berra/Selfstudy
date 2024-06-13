import numpy as np
import torch

# 바꿔치기 연산
tensor = torch.ones(4, 4)

print(f"{tensor}\n")
tensor.add_(5)
print(tensor)