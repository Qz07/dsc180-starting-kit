import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import cycle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_simnpo(
    net: nn.Module,
    retain: torch.utils.data.DataLoader,
    forget: torch.utils.data.DataLoader,
    validation: torch.utils.data.DataLoader,
    epochs: int = 5,
    alpha: float = 0.8,   # weight for retain CE
    beta: float = 5.0,   # sharpness in SimNPO
    delta: float = 0.3,   # margin / tolerance
):
    """
    Unlearning by SimNPO-style finetuning.

    Args:
      net: nn.Module, pre-trained model to unlearn
      retain: DataLoader over retain set
      forget: DataLoader over forget set
      validation: DataLoader over validation set (for reporting only)
      epochs: number of epochs
      alpha: weight for retain CE term
      beta: SimNPO sharpness (usually 5–10)
      delta: SimNPO margin (0.1–0.3 is typical)
      save_path: if not None, save model weights here at the end

    Returns:
      net: updated (unlearned) model
    """
    net = net.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # we'll loop forever over the forget loader so every retain batch gets a forget batch
    forget_iter = cycle(forget)

    net.train()
    for ep in range(epochs):
        total_batches = len(retain)
        for i, (x_ret, y_ret) in enumerate(retain, start=1):
            x_for, y_for = next(forget_iter)

            x_ret, y_ret = x_ret.to(DEVICE), y_ret.to(DEVICE)
            x_for, y_for = x_for.to(DEVICE), y_for.to(DEVICE)

            optimizer.zero_grad()

            # 1) retain term (normal CE) --------------------------------------
            out_ret = net(x_ret)
            loss_retain = criterion(out_ret, y_ret)

            # 2) forget term (SimNPO) -----------------------------------------
            out_for = net(x_for)  # logits on forget
            # log p(y|x)
            logp_for = F.log_softmax(out_for, dim=1).gather(
                1, y_for.view(-1, 1)
            ).squeeze(1)  # [B]
            # SimNPO loss: -(2/β) * log σ( -β*logp - δ )
            s = -beta * logp_for - delta
            loss_forget = -(2.0 / beta) * torch.log(torch.sigmoid(s) + 1e-12).mean()

            # 3) total ---------------------------------------------------------
            loss = loss_forget + alpha * loss_retain
            loss.backward()
            optimizer.step()

        scheduler.step()

    net.eval()

    return net