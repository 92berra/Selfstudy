import torch
import numpy as np

# single-element 텐서
tensor = torch.ones(4, 4)

agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))