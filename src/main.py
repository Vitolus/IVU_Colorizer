#%% import libraries
import os
import re
import cv2
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchinfo import summary
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import matplotlib.pyplot as plt
#%%
path = '../data'
SIZE = 160
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)
#%%
def sort_files(folder):
    convert_func = lambda x: int(x) if x.isdigit() else x.lower()
    key_func = lambda x: [convert_func(c) for c in re.split('([0-9]+)', x)]
    return sorted(folder, key=key_func)

data_color = []
path = '../data/color'
folder = os.listdir(path)
folder = sort_files(folder)
for file in tqdm(folder):
    img = cv2.imread(path + '/' + file, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (SIZE, SIZE))
    data_color.append(img)
data_color = np.array(data_color)

data_gray = []
path = '../data/gray'
folder = os.listdir(path)
folder = sort_files(folder)
for file in tqdm(folder):
    img = cv2.imread(path + '/' + file, 1)
    img = cv2.resize(img, (SIZE, SIZE))
    data_gray.append(img)
data_gray = np.array(data_gray)
#%%
for _ in range(5):
    idx = np.random.randint(0, len(data_color) - 1)
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 3, 1)
    plt.title('Color Image', fontsize=20)
    plt.imshow(data_color[idx])
    plt.subplot(1, 3, 2)
    plt.title('Grayscale Image ', fontsize=20)
    plt.imshow(data_gray[idx])
    plt.show()
#%%
color_min = np.min(data_color, axis=(0, 1, 2), keepdims=True)
color_max = np.max(data_color, axis=(0, 1, 2), keepdims=True)
gray_min = np.min(data_gray, axis=(0, 1, 2), keepdims=True)
gray_max = np.max(data_gray, axis=(0, 1, 2), keepdims=True)
data_color_scaled = (data_color - color_min) / (color_max - color_min)
data_gray_scaled = (data_gray - gray_min) / (gray_max - gray_min)
color_mean = np.mean(data_color_scaled, axis=(0, 1, 2))
color_std = np.std(data_color_scaled, axis=(0, 1, 2))
gray_mean = np.mean(data_gray_scaled, axis=(0, 1, 2))
gray_std = np.std(data_gray_scaled, axis=(0, 1, 2))
print(color_mean, color_std)
print(gray_mean, gray_std)
#%%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_gray, data_color, gray_transform=None, color_transform=None):
        self.data_color = data_color
        self.data_gray = data_gray
        self.gray_transform = gray_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.data_gray)

    def __getitem__(self, idx):
        gray_img = self.data_gray[idx]
        color_img = self.data_color[idx]
        if self.gray_transform:
            gray_img = self.gray_transform(gray_img)
        if self.color_transform:
            color_img = self.color_transform(color_img)
        return gray_img, color_img
#%%
#TODO: need to adjust stratify
gray_train, gray_test, color_train, color_test = (
    train_test_split(data_gray, data_color, test_size=0.2, random_state=42, stratify=data_color))
print(len(gray_train), len(gray_test))
#%%
gray_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=gray_mean, std=gray_std)
])
color_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=color_mean, std=color_std)
])
trainset = MyDataset(gray_train, color_train, gray_transform=gray_transform, color_transform=color_transform)
testset = MyDataset(gray_test, color_test, gray_transform=gray_transform, color_transform=color_transform)
#%%
class EarlyStopping:
    def __init__(self, patience=15, delta=0.5, window_size=10):
        self.patience = patience
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.delta = delta
        self.window_size = window_size
        self.val_window = []

    def __call__(self, val_loss, net):
        self.val_window.append(val_loss)
        if len(self.val_window) > self.window_size:
            self.val_window.pop(0)
        avg_val = np.mean(self.val_window)

        if avg_val == self.best_score or avg_val > self.best_score + self.delta:
            self.counter += 1
        elif avg_val < self.best_score:
            self.best_score = avg_val
            self.save_checkpoint(net)
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), '../models/checkpoint.pth')
#%% Loss functions
class ImageGradientLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ImageGradientLoss, self).__init__()

    def forward(self, inputs, targets):
        dh_comp = inputs.diff(dim=3) - targets.diff(dim=3) # (B, C, H, W-1)
        dv_comp = inputs.diff(dim=2) - targets.diff(dim=2) # (B, C, H-1, W)
        dh_comp = dh_comp.pow(2).sum(dim=1)
        dv_comp = dv_comp.pow(2).sum(dim=1)
        return dh_comp.sum() + dv_comp.sum()

def fit(net, trainloader, optimizer, loss_fn1=nn.MSELoss(reduction='sum'), loss_fn2=ImageGradientLoss(), lamda1=0.5, lamda2=0.5):
    net.train()
    total_loss, acc, count = 0, 0, 0
    for grays, colors in tqdm(trainloader):
        grays, colors = grays.to(device), colors.to(device)
        optimizer.zero_grad()
        out = net(grays)
        loss1, loss2 = loss_fn1(out, colors), loss_fn2(out, colors)
        loss = (lamda1 * loss1 + lamda2 * loss2) / out.numel()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        predicted = torch.max(out, 1)[1]
        acc += (predicted == colors).sum().item()
        count += len(colors)
    return total_loss / count, acc / count

def predict(net, testloader, loss_fn1=nn.MSELoss(reduction='sum'), loss_fn2=ImageGradientLoss(), lamda1=0.5, lamda2=0.5):
    net.eval()
    total_loss, acc, count = 0, 0, 0
    with torch.no_grad():
        for grays, colors in tqdm(testloader):
            grays, colors = grays.to(device), colors.to(device)
            out = net(grays)
            loss1, loss2 = loss_fn1(out, colors), loss_fn2(out, colors)
            loss = (lamda1 * loss1 + lamda2 * loss2) / out.numel()
            total_loss += loss.item()
            predicted = torch.max(out, 1)[1]
            acc += (predicted == colors).sum().item()
            count += len(colors)
    return total_loss / count, acc / count

