#!/usr/bin/env python3
import argparse
import itertools
import json
import os
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import torch
from torchvision.models import resnet18

from truth_ratio import truth_ratio
from rmu import train_rmu
from orginal_metrics import accuracy, run_simple_mia
from load_data import load_data
from orginal_method import unlearning

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())


# --------------------------
# Small helpers
# --------------------------
def parse_float_list(s: str) -> List[float]:
    if s is None or s.strip() == "":
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def parse_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in {"1", "true", "yes", "y"}:
        return True
    if s in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean from: {s}")

def ensure_dir(p: str):
    if p and p.strip():
        os.makedirs(os.path.dirname(p), exist_ok=True)


# --------------------------
# Single run wrapper
# --------------------------
def run_once(args, forget_loader, retain_loader, val_loader, test_loader, rmu_params: Dict[str, Any]) -> Dict[str, Any]:
    method = args.method.lower()

    if method == "rmu":
        print("Running RMU method")
        local_path = args.init_weights
        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        ft_model = resnet18(weights=None, num_classes=10)
        ft_model.load_state_dict(weights_pretrained)
        ft_model.to(DEVICE)

        # Build k schedule
        if args.k_schedule:
            k_schedule = parse_float_list(args.k_schedule)
        else:
            # if user provided --k, use that for all epochs; else use default [0.75, 1.0, ...]
            if args.k is not None:
                k_schedule = [float(args.k)] * args.num_epochs
            else:
                k_schedule = None  # let train_rmu apply its own default

        unlearned_mdl = train_rmu(
            model=ft_model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            epochs=args.num_epochs,
            k_schedule=k_schedule,
            lr=rmu_params["lr"],
            weight_decay=rmu_params["weight_decay"],
            alpha=rmu_params["alpha"],
            alpha_start=rmu_params["alpha_start"],
            alpha_end=rmu_params["alpha_end"],
            c=rmu_params["c"],
            auto_scale_c=rmu_params["auto_scale_c"],
            c_scale=rmu_params["c_scale"],
            grad_clip=rmu_params["grad_clip"],
            device=DEVICE,
            seed=rmu_params["seed"],
            verbose=not args.quiet,
        )

    elif method == "original":
        print("Running Original method")
        local_path = args.init_weights
        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        ft_model = resnet18(weights=None, num_classes=10)
        ft_model.load_state_dict(weights_pretrained)
        ft_model.to(DEVICE)

        unlearned_mdl = unlearning(ft_model, retain_loader, forget_loader, val_loader)

    elif method == "retrain":
        print("Running eval on Retrain method")
        local_path = args.retrain_weights
        weights_pretrained = torch.load(local_path, map_location=DEVICE)

        unlearned_mdl = resnet18(weights=None, num_classes=10)
        unlearned_mdl.load_state_dict(weights_pretrained)
        unlearned_mdl.to(DEVICE)

    else:
        raise ValueError("Invalid method selected.")

    # ---- Evaluate ----
    unlearned_mdl.eval()
    acc = accuracy(unlearned_mdl, test_loader)
    tr = truth_ratio(unlearned_mdl, forget_loader)
    mia = run_simple_mia(unlearned_mdl, forget_loader, test_loader)
    mia_mean = float(np.mean(mia))

    result = {
        "method": method,
        "epochs": args.num_epochs,
        "acc": float(acc),
        "truth_ratio": float(tr),
        "mia_acc": mia_mean,
    }

    return result


# --------------------------
# Argument parser
# --------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--method', type=str, required=True, choices=['rmu', 'original', 'retrain'],
                   help='Unlearning method to run')
    p.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    # Initialization weights
    p.add_argument('--init_weights', type=str, default='weights_resnet18_cifar10.pth',
                   help='Path to initial pretrained weights for RMU/original')
    p.add_argument('--retrain_weights', type=str, default='retrain_weights_resnet18_cifar10.pth',
                   help='Path to retrained model weights (for method=retrain)')

    # K / schedule controls
    p.add_argument('--k', type=float, default=None, help='If set, use this k for all epochs (overridden by k_schedule)')
    p.add_argument('--k_schedule', type=str, default='',
                   help='Comma-separated ks per epoch (e.g. "0.75,1.0,1.0"). Overrides --k if provided.')

    # RMU hyperparameters (single or sweep lists)
    p.add_argument('--lr', type=str, default='3e-4', help='Float or comma-list for sweep (e.g. "3e-4,1e-4")')
    p.add_argument('--weight_decay', type=str, default='1e-4', help='Float or comma-list')
    p.add_argument('--alpha', type=str, default='1000.0', help='Float or comma-list')
    p.add_argument('--alpha_start', type=str, default='-1.0', help='Float or comma-list (-1 disables ramp)')
    p.add_argument('--alpha_end', type=str, default='-1.0', help='Float or comma-list (-1 disables ramp)')
    p.add_argument('--c', type=str, default='4.0', help='Float or comma-list')
    p.add_argument('--auto_scale_c', type=str, default='false', help='Bool or comma-list (true/false)')
    p.add_argument('--c_scale', type=str, default='3.0', help='Float or comma-list (used if auto_scale_c)')
    p.add_argument('--grad_clip', type=str, default='1.0', help='Float or comma-list (0 to disable)')
    p.add_argument('--seed', type=str, default='42', help='Int or comma-list')

    # Sweeps / logging
    p.add_argument('--sweep', action='store_true', help='If set, run Cartesian product over any comma-lists above')
    p.add_argument('--save_csv', type=str, default='',
                   help='If set, append results to this CSV file')
    p.add_argument('--save_jsonl', type=str, default='',
                   help='If set, append results as JSONL to this file')
    p.add_argument('--quiet', action='store_true', help='Reduce per-epoch logs from RMU')

    return p


def coerce_list_or_single(s: str, is_bool: bool = False, is_int: bool = False) -> List[Any]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if not parts:
        return []
    if is_bool:
        return [parse_bool(x) for x in parts]
    if is_int:
        return [int(float(x)) for x in parts]  # tolerate "42.0"
    # float
    return [float(x) for x in parts]


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load data once
    forget_loader, retain_loader, val_loader, test_loader = load_data()

    # Prepare value lists (for sweep or single)
    grid = {
        "lr": coerce_list_or_single(args.lr),
        "weight_decay": coerce_list_or_single(args.weight_decay),
        "alpha": coerce_list_or_single(args.alpha),
        "alpha_start": coerce_list_or_single(args.alpha_start),
        "alpha_end": coerce_list_or_single(args.alpha_end),
        "c": coerce_list_or_single(args.c),
        "auto_scale_c": coerce_list_or_single(args.auto_scale_c, is_bool=True),
        "c_scale": coerce_list_or_single(args.c_scale),
        "grad_clip": coerce_list_or_single(args.grad_clip),
        "seed": coerce_list_or_single(args.seed, is_int=True),
    }

    # If not sweeping, collapse to single values (first of list or parsed scalar)
    def default_or_first(name: str, default):
        vals = grid[name]
        if len(vals) == 0:
            # parse the single (non-list) input again
            if name in {"auto_scale_c"}:
                return parse_bool(getattr(args, name))
            if name in {"seed"}:
                return int(float(getattr(args, name)))
            return float(getattr(args, name))
        return vals[0]

    if not args.sweep:
        rmu_params = {
            "lr": default_or_first("lr", 3e-4),
            "weight_decay": default_or_first("weight_decay", 1e-4),
            "alpha": default_or_first("alpha", 1000.0),
            "alpha_start": default_or_first("alpha_start", -1.0),
            "alpha_end": default_or_first("alpha_end", -1.0),
            "c": default_or_first("c", 4.0),
            "auto_scale_c": default_or_first("auto_scale_c", False),
            "c_scale": default_or_first("c_scale", 3.0),
            "grad_clip": default_or_first("grad_clip", 1.0),
            "seed": default_or_first("seed", 42),
        }

        result = run_once(args, forget_loader, retain_loader, val_loader, test_loader, rmu_params)
        print(f"Results after unlearning: method={args.method}")
        print(f"Test Accuracy: {result['acc']:.4f}")
        print(f"Truth Ratio: {result['truth_ratio']:.4f}")
        print(f"Membership Inference Attack Accuracy: {result['mia_acc']:.4f}")

        # Persist results
        if args.save_csv:
            ensure_dir(args.save_csv)
            header_needed = not os.path.exists(args.save_csv)
            with open(args.save_csv, "a") as f:
                if header_needed:
                    f.write("timestamp,method,epochs,lr,weight_decay,alpha,alpha_start,alpha_end,c,auto_scale_c,c_scale,grad_clip,seed,acc,truth_ratio,mia_acc\n")
                f.write("{ts},{method},{epochs},{lr},{wd},{alpha},{as_},{ae_},{c},{asc},{cs},{gc},{seed},{acc},{tr},{mia}\n".format(
                    ts=datetime.now().isoformat(),
                    method=args.method,
                    epochs=args.num_epochs,
                    lr=rmu_params["lr"],
                    wd=rmu_params["weight_decay"],
                    alpha=rmu_params["alpha"],
                    as_=rmu_params["alpha_start"],
                    ae_=rmu_params["alpha_end"],
                    c=rmu_params["c"],
                    asc=int(bool(rmu_params["auto_scale_c"])),
                    cs=rmu_params["c_scale"],
                    gc=rmu_params["grad_clip"],
                    seed=rmu_params["seed"],
                    acc=result["acc"],
                    tr=result["truth_ratio"],
                    mia=result["mia_acc"],
                ))

        if args.save_jsonl:
            ensure_dir(args.save_jsonl)
            payload = {
                "timestamp": datetime.now().isoformat(),
                "method": args.method,
                "epochs": args.num_epochs,
                "k_schedule": parse_float_list(args.k_schedule) if args.k_schedule else ([args.k] * args.num_epochs if args.k is not None else None),
                **rmu_params,
                **result,
            }
            with open(args.save_jsonl, "a") as f:
                f.write(json.dumps(payload) + "\n")

        return

    # ---- Sweep mode ----
    # Build Cartesian product of lists (only params with length>0 are swept)
    sweep_keys = [k for k, v in grid.items() if len(v) > 0]
    if not sweep_keys:
        raise ValueError("You passed --sweep but no comma-lists to sweep over. Provide lists like --lr 3e-4,1e-4")

    combos = list(itertools.product(*[grid[k] for k in sweep_keys]))
    print(f"Sweep over {len(combos)} configurations across: {sweep_keys}")

    for i, combo in enumerate(combos, 1):
        rmu_params = {
            "lr": default_or_first("lr", 3e-4),
            "weight_decay": default_or_first("weight_decay", 1e-4),
            "alpha": default_or_first("alpha", 1000.0),
            "alpha_start": default_or_first("alpha_start", -1.0),
            "alpha_end": default_or_first("alpha_end", -1.0),
            "c": default_or_first("c", 4.0),
            "auto_scale_c": default_or_first("auto_scale_c", False),
            "c_scale": default_or_first("c_scale", 3.0),
            "grad_clip": default_or_first("grad_clip", 1.0),
            "seed": default_or_first("seed", 42),
        }
        for k, v in zip(sweep_keys, combo):
            rmu_params[k] = v

        # Stamp per run (kept for log uniqueness if desired)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{i:03d}"

        print(f"\n=== Sweep {i}/{len(combos)} ===")
        print(json.dumps(rmu_params, indent=2))

        result = run_once(args, forget_loader, retain_loader, val_loader, test_loader, rmu_params)
        print(f"[{i}/{len(combos)}] acc={result['acc']:.4f}  tr={result['truth_ratio']:.4f}  mia={result['mia_acc']:.4f}")

        if args.save_csv:
            ensure_dir(args.save_csv)
            header_needed = not os.path.exists(args.save_csv)
            with open(args.save_csv, "a") as f:
                if header_needed:
                    f.write("timestamp,method,epochs,lr,weight_decay,alpha,alpha_start,alpha_end,c,auto_scale_c,c_scale,grad_clip,seed,acc,truth_ratio,mia_acc\n")
                f.write("{ts},{method},{epochs},{lr},{wd},{alpha},{as_},{ae_},{c},{asc},{cs},{gc},{seed},{acc},{tr},{mia}\n".format(
                    ts=datetime.now().isoformat(),
                    method=args.method,
                    epochs=args.num_epochs,
                    lr=rmu_params["lr"],
                    wd=rmu_params["weight_decay"],
                    alpha=rmu_params["alpha"],
                    as_=rmu_params["alpha_start"],
                    ae_=rmu_params["alpha_end"],
                    c=rmu_params["c"],
                    asc=int(bool(rmu_params["auto_scale_c"])),
                    cs=rmu_params["c_scale"],
                    gc=rmu_params["grad_clip"],
                    seed=rmu_params["seed"],
                    acc=result["acc"],
                    tr=result["truth_ratio"],
                    mia=result["mia_acc"],
                ))

        if args.save_jsonl:
            ensure_dir(args.save_jsonl)
            payload = {
                "timestamp": datetime.now().isoformat(),
                "sweep_index": i,
                "method": args.method,
                "epochs": args.num_epochs,
                "k_schedule": parse_float_list(args.k_schedule) if args.k_schedule else ([args.k] * args.num_epochs if args.k is not None else None),
                **rmu_params,
                **result,
            }
            with open(args.save_jsonl, "a") as f:
                f.write(json.dumps(payload) + "\n")


if __name__ == "__main__":
    main()
