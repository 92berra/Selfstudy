import sys
import torch
import pandas as pd
import sklearn as sk

print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")

mps_device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"mps is", "available" if torch.backends.mps.is_available() else "not available")