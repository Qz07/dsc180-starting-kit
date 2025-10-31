import os
import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.models import resnet18

def load_data():
    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    
    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )
    # test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    
    n = len(held_out)            # 10000 images in CIFAR10 test set
    n_test = n // 2              # 5000
    n_val = n - n_test           # 5000 (use remainder to avoid off-by-one)
    
    test_set, val_set = torch.utils.data.random_split(
        held_out, [n_test, n_val], generator=RNG
    )
    
    
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)
    
    # download the forget and retain index split
    local_path = "forget_idx.npy"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    forget_idx = np.load(local_path)
    
    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]
    
    # split train set into a forget and a retain set
    forget_set = torch.utils.data.Subset(train_set, forget_idx)
    retain_set = torch.utils.data.Subset(train_set, retain_idx)
    
    forget_loader = torch.utils.data.DataLoader(
        forget_set, batch_size=128, shuffle=True, num_workers=2
    )
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )
    return forget_loader, retain_loader