import torch
import numpy as np

# 텐서를 NumPy 배열로 변환하기
t = torch.ones(5)
print(f"t: {t}")

n = t.numpy()
print(f"n: {n}")


# 텐서의 변경사항이 NumPy 배열에 반영된다.
t.add_(1)

print(f"t: {t}")
print(f"n: {n}")