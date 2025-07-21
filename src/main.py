#%% import libraries
import os
import re
import cv2
from tqdm import tqdm
import optuna
from optuna.trial import TrialState
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchinfo import summary
from sklearn.model_selection import KFold, train_test_split
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
gray_train, gray_test, color_train, color_test = (
    train_test_split(data_gray, data_color, test_size=0.2, random_state=42))
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

def objective(trial, trainset, X, y):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64,256, 512, 1024])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    val_accs = []
    train_acc, train_loss, val_acc, val_loss, mean_acc = 0, 0, 0, 0, 0
    split_n = 0
    prog_bar = tqdm(kf.split(X, y), desc="Splits")
    for train_idx, val_idx in prog_bar:
        split_n += 1
        trainloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(train_idx))
        valloader = DataLoader(trainset, batch_size=batch_size, sampler=SubsetRandomSampler(val_idx))
        net = Net().to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        early_stopping = EarlyStopping()
        for epoch in range(100):
            train_loss, train_acc = fit(net, trainloader, optimizer)
            val_loss, val_acc = predict(net, valloader)
            scheduler.step(val_acc)
            early_stopping(val_loss, net)
            prog_bar.set_description(
                f"Split {split_n} - Epoch {epoch + 1}, Train acc={train_acc:.3f}, Train loss={train_loss:.3f}, "
                f"Validation acc={val_acc:.3f}, Validation loss={val_loss:.3f}")
            if early_stopping.early_stop:
                break
        val_accs.append(val_acc)
        meanacc = np.mean(val_accs)
        trial.report(meanacc, split_n)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return mean_acc
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = lambda in_c, out_c, kernel, stride: nn.Conv2d(in_c, out_c, kernel, stride)
        self.convT = lambda in_c, out_c, kernel, stride: nn.ConvTranspose2d(in_c, out_c, kernel, stride)
        self.bnorm = lambda c: nn.BatchNorm2d(c)
        self.dropout = nn.Dropout(0.2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        d1 = self.lrelu(self.conv(3, 128, 3, 2)(x)) # (B, 128, H-2, W-2) downsample 1
        d2 = self.lrelu(self.conv(128, 128, 3, 2)(d1)) # (B, 128, H-4, W-4) downsample 2
        d3 = self.lrelu(self.bnorm(256)(self.conv(128, 256, 3, 2)(d2))) # (B, 256, H-6, W-6) downsample 3 with batch norm
        d4 = self.lrelu(self.bnorm(512)(self.conv(256, 512, 3, 2)(d3))) # (B, 512, H-8, W-8) downsample 4 with batch norm
        d5 = self.lrelu(self.bnorm(512)(self.conv(512, 512, 3, 2)(d4))) # (B, 512, H-10, W-10) downsample 5 with batch norm
        u1 = self.lrelu(self.convT(512, 512, 3, 2)(d5)) # (B, 512, H-12, W-12) upsample 1
        u1 = torch.cat([u1, d4], dim=1)
        u2 = self.lrelu(self.convT(512, 256, 3, 2)(u1)) # (B, 256, H-14, W-14) upsample 2
        u2 = torch.cat([u2, d3], dim=1)
        u3 = self.lrelu(self.convT(256, 128, 3, 2)(u2)) # (B, 128, H-16, W-16) upsample 3
        u3 = torch.cat([u3, d2], dim=1)
        u4 = self.lrelu(self.convT(128, 128, 3, 2)(u3)) # (B, 128, H-18, W-18) upsample 4
        u4 = torch.cat([u4, d1], dim=1)
        u5 = self.lrelu(self.convT(128, 3, 3, 2)(u4)) # (B, 3, H-2, W-2) upsample 5
        u5 = torch.cat([u5, x], dim=1)
        x = self.conv(3, 3, 2, 1)(u5)
        return x
