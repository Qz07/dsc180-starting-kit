#!/usr/bin/env python3
# kl_student_ref.py
#
# Utility to compute KL(student || reference) over a data loader.

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def kl_student_vs_ref(
    student: torch.nn.Module,
    ref: torch.nn.Module,
    loader,
    device: str = "cuda",
    T: float = 4.0,
) -> float:
    """
    Compute KL(student || ref) over a loader using temperature-scaled logits.

    KL is computed as:
        KL( p_ref || p_student ) or KL( p_student || p_ref )?
    Here we follow your original code:
        pa_log = log_softmax(student_logits / T)
        pb     = softmax(ref_logits / T)
        kl = KL(pa || pb) = E_pa[ log(pa) - log(pb) ]
      implemented as F.kl_div(pa_log, pb, reduction='batchmean') * T^2

    Args:
      student:  The (unlearned) student model.
      ref:      The reference model (e.g., retain-only retrain or teacher).
      loader:   DataLoader yielding (x, y) or (x, _) batches.
      device:   "cuda" or "cpu".
      T:        Temperature used in softmax/log-softmax.

    Returns:
      Scalar float: mean KL over all batches.
    """
    device = device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"

    student = student.to(device).eval()
    ref = ref.to(device).eval()

    kls = []

    for x, _ in loader:
        x = x.to(device)
        za = student(x)  # student logits
        zb = ref(x)      # ref logits

        pa_log = F.log_softmax(za / T, dim=-1)
        pb = F.softmax(zb / T, dim=-1)

        kl = F.kl_div(pa_log, pb, reduction="batchmean") * (T ** 2)
        kls.append(kl.item())

    if len(kls) == 0:
        return float("nan")
    return float(np.mean(kls))