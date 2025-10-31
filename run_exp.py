import argparse
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

from truth_ratio import truth_ratio
from rmu import train_rmu
from orginal_metrics import accuracy, run_simple_mia
from load_data import load_data
from orginal_method import unlearning




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True, help='choose method')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--k', type=float, default=0.25)

    args = parser.parse_args()
    # print("Arguments:", args)

    # assert args.method in ['rmu', 'orginal', 'retrain'], 
    forget_loader, retain_loader, val_loader, test_loader = load_data()
    if args.method == 'rmu':
        print("Running RMU method")
        local_path = "weights_resnet18_cifar10.pth"
        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        ft_model = resnet18(weights=None, num_classes=10)
        ft_model.load_state_dict(weights_pretrained)
        ft_model.to(DEVICE)

        unlearned_mdl = train_rmu(ft_model, forget_loader, retain_loader, args.k, epochs=args.num_epochs, device=DEVICE)

    elif args.method == 'original':
        print("Running Original method")
        local_path = "weights_resnet18_cifar10.pth"
        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        ft_model = resnet18(weights=None, num_classes=10)
        ft_model.load_state_dict(weights_pretrained)
        ft_model.to(DEVICE)

        unlearned_mdl = unlearning(ft_model, retain_loader, forget_loader, val_loader)

    elif args.method == 'retrain':
        print("Running eval on Retrain method")

        local_path = "retrain_weights_resnet18_cifar10.pth"

        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        # load model with pre-trained weights
        unlearned_mdl = resnet18(weights=None, num_classes=10)
        unlearned_mdl.load_state_dict(weights_pretrained)
        unlearned_mdl.to(DEVICE)

    else:
        raise ValueError("Invalid method selected.")
    
    unlearned_mdl.eval()
    acc = accuracy(unlearned_mdl, test_loader)
    tr = truth_ratio(unlearned_mdl, forget_loader)
    mia = run_simple_mia(unlearned_mdl, forget_loader, test_loader)
    print(f"Results after unlearning: through the method {args.method}")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Truth Ratio: {tr:.4f}")
    print(f"Membership Inference Attack Accuracy: {mia.mean():.4f}")