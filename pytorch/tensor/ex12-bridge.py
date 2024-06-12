import torch
import numpy as np

# NumPy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

# NumPy 배열의 변경 사항이 텐서에 반영된다.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

