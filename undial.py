import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, optim
import random
import numpy as np


# --------------------------
# Utils
# --------------------------
def set_all_seeds(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


# --------------------------
# UNDIAL training
# --------------------------
def train_undial(
    model: nn.Module,
    forget_loader,
    retain_loader,
    *,
    epochs: int = 12,
    alpha: float = 12.0,          # logit depression on forget
    temp: float = 4.0,            # KD temperature
    lambda_retain: float = 1.0,   # retain loss weight
    gamma: float = 0.8,           # entropy-to-uniform on forget (warmup)
    warmup_epochs: int = 2,       # epochs with entropy warmup
    forget_ratio: int = 2,        # extra forget steps during warmup
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    clip_norm: Optional[float] = 1.0,
    use_amp: Optional[bool] = None,
    device: str = "cuda",
    seed: int = 42,
    verbose: bool = True,
) -> nn.Module:
    """
    Run UNDIAL on a pretrained model.

    Args:
        model:        Pretrained full-data model (will NOT be modified).
        forget_loader: DataLoader over the FORGET set.
        retain_loader: DataLoader over the RETAIN set.
        epochs:      Number of UNDIAL epochs.
        alpha:       Logit depression applied to teacher logits on forget samples.
        temp:        Distillation temperature.
        lambda_retain: Weight on retain KD loss.
        gamma:       Weight on entropy-to-uniform term during warmup epochs.
        warmup_epochs: Number of initial epochs with extra entropy regularization.
        forget_ratio: Extra forget-only steps per retain step during warmup.
        lr, weight_decay, momentum: SGD hyperparameters.
        clip_norm:   Max gradient norm (None/<=0 disables clipping).
        use_amp:     If None, enabled when CUDA is available.
        device:      "cuda" or "cpu".
        seed:        Random seed for reproducibility.
        verbose:     Print per-epoch stats.

    Returns:
        undial_student: the unlearned model after UNDIAL training.
    """
    set_all_seeds(seed)

    # Resolve device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    use_amp = torch.cuda.is_available() if use_amp is None else use_amp

    # Frozen teacher = copy of original model
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # UNDIAL student = another copy we will train
    undial_student = copy.deepcopy(model).to(device)
    undial_student.train()

    # Optimizer & scheduler
    opt = optim.SGD(
        undial_student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ------------ inner step function ------------
    def one_step(x_r, y_r, x_f, y_f, epoch_idx: int):
        undial_student.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Teacher logits on forget + logit depression
            with torch.no_grad():
                z_t_f = teacher(x_f)  # (B, C)
                z_adj = z_t_f.clone()
                idx = torch.arange(y_f.size(0), device=y_f.device)
                z_adj[idx, y_f] = z_adj[idx, y_f] - alpha
                p_adj = F.softmax(z_adj / temp, dim=-1)

            # Student forward
            z_s_f = undial_student(x_f)
            z_s_r = undial_student(x_r)

            # Forget loss: KL(student || adjusted teacher)
            loss_forget = F.kl_div(
                F.log_softmax(z_s_f / temp, dim=-1),
                p_adj,
                reduction="batchmean",
            ) * (temp**2)

            # Entropy-to-uniform warmup on forget
            if epoch_idx < warmup_epochs and gamma > 0:
                loss_entropy = gamma * F.kl_div(
                    F.log_softmax(z_s_f, dim=-1),
                    torch.full_like(z_s_f, 1.0 / z_s_f.size(1)),
                    reduction="batchmean",
                )
                loss_forget = loss_forget + loss_entropy

            # Retain loss: KD to teacher on retain set
            with torch.no_grad():
                z_t_r = teacher(x_r)
            loss_retain = lambda_retain * F.kl_div(
                F.log_softmax(z_s_r / temp, dim=-1),
                F.softmax(z_t_r / temp, dim=-1),
                reduction="batchmean",
            ) * (temp**2)

            loss = loss_forget + loss_retain

        opt.zero_grad()
        scaler.scale(loss).backward()
        if clip_norm is not None and clip_norm > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(undial_student.parameters(), clip_norm)
        scaler.step(opt)
        scaler.update()

        return loss_forget.detach().item(), loss_retain.detach().item()

    # ------------ training loop ------------
    for ep in range(epochs):
        lf_sum = lr_sum = 0.0

        # Paired iteration over retain/forget
        steps = min(len(retain_loader), len(forget_loader))
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)

        for _ in range(steps):
            # Get retain batch (cycle if needed)
            try:
                x_r, y_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                x_r, y_r = next(retain_iter)

            # Get forget batch (cycle if needed)
            try:
                x_f, y_f = next(forget_iter)
            except StopIteration:
                forget_iter = iter(forget_loader)
                x_f, y_f = next(forget_iter)

            x_r, y_r = x_r.to(device), y_r.to(device).long()
            x_f, y_f = x_f.to(device), y_f.to(device).long()

            # Main balanced step
            lf, lr_ = one_step(x_r, y_r, x_f, y_f, ep)
            lf_sum += lf
            lr_sum += lr_

            # Extra forget-only steps during warmup (approx FORGET_RATIO:1)
            if ep < warmup_epochs:
                for _ in range(max(0, forget_ratio - 1)):
                    try:
                        x_f2, y_f2 = next(forget_iter)
                    except StopIteration:
                        forget_iter = iter(forget_loader)
                        x_f2, y_f2 = next(forget_iter)

                    x_f2, y_f2 = x_f2.to(device), y_f2.to(device).long()
                    lf2, lr2 = one_step(x_r, y_r, x_f2, y_f2, ep)
                    lf_sum += lf2
                    lr_sum += lr2

        sched.step()

        if verbose:
            acc_f = _accuracy(undial_student, forget_loader, device)
            acc_r = _accuracy(undial_student, retain_loader, device)
            print(
                f"[UNDIAL] ep {ep+1}/{epochs} "
                f"loss_f={lf_sum/steps:.3f} loss_r={lr_sum/steps:.3f} "
                f"acc_f={acc_f:.3f} acc_r={acc_r:.3f}"
            )

    undial_student.eval()
    if verbose:
        print("UNDIAL training complete — returning unlearned student model.")

    return undial_student