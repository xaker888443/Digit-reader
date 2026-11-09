"""Обучение CNN-классификатора цифр."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data import generate_dataset
from model import DigitCNN


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    samples_per_class: int = 300
    val_split: float = 0.2
    patience: int = 5
    checkpoint: Path = Path("digit_net.pt")


def _split(X: np.ndarray, y: np.ndarray, val_frac: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    cut = int((1.0 - val_frac) * len(X))
    tr, va = order[:cut], order[cut:]
    return (
        TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
        TensorDataset(torch.from_numpy(X[va]), torch.from_numpy(y[va])),
    )


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = seen = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training:
                optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += xb.size(0)
    return total_loss / seen, correct / seen


def train(cfg: TrainConfig | None = None) -> float:
    cfg = cfg or TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    X, y = generate_dataset(cfg.samples_per_class)
    print(f"[data]   X={X.shape} y={y.shape}")

    train_ds, val_ds = _split(X, y, cfg.val_split)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = DigitCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=8, gamma=0.5)

    best_acc, bad_epochs = 0.0, 0
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_dl, loss_fn, device, opt)
        sched.step()
        _, va_acc = _run_epoch(model, val_dl, loss_fn, device)

        print(
            f"epoch {epoch:02d}/{cfg.epochs}  "
            f"loss={tr_loss:.4f}  train={tr_acc * 100:.2f}%  val={va_acc * 100:.2f}%"
        )

        if va_acc > best_acc:
            best_acc, bad_epochs = va_acc, 0
            torch.save(model.state_dict(), cfg.checkpoint)
            print(f"  * new best -> {cfg.checkpoint}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stop (patience={cfg.patience})")
                break

    print(f"\nbest val acc: {best_acc * 100:.2f}%")
    return best_acc


if __name__ == "__main__":
    train()