#%% import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
#%%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)