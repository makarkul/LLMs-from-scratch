"""Step-4a training loop.

On-the-fly synthetic data: every step samples a fresh batch of subframes
with random SNR in [SNR_MIN_DB, SNR_MAX_DB], fresh EPA channel, fresh AWGN,
fresh random data bits. No stored dataset.

Usage (local smoke):
    python train.py --steps 500 --batch 8

Usage (GPU run):
    python train.py --steps 30000 --batch 32 --out checkpoints/
"""

import argparse
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import generate_batch, SNR_MIN_DB, SNR_MAX_DB
from model import LearnedReceiver


def cells_to_bit_targets(tx_bits, data_mask, grid_shape):
    """Scatter flat tx_bits (B, n_data, 2) into full (B, T, S, 2) target grid,
    using data_mask (T, S) in row-major order. Cells outside data_mask are 0
    and will be ignored via the loss mask."""
    B, T, S = grid_shape
    target = torch.zeros(B, T, S, 2, dtype=torch.float32, device=tx_bits.device)
    mask_flat = data_mask.view(-1)
    idx = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)  # (n_data,)
    # target[B, :, :, :][:, idx, :] <- tx_bits
    tgt_flat = target.view(B, T * S, 2)
    tgt_flat[:, idx, :] = tx_bits.float()
    return target


def compute_loss_and_errs(logits, target, data_mask):
    """BCE-with-logits loss on data cells only. Returns (loss, n_errors, n_bits)."""
    # logits, target: (B, T, S, 2); data_mask: (T, S)
    mask = data_mask.view(1, *data_mask.shape, 1).expand_as(logits).float()
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    loss = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    with torch.no_grad():
        pred = (logits > 0).to(target.dtype)
        errs = ((pred != target).float() * mask).sum().item()
        nbits = mask.sum().item()
    return loss, errs, nbits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=8)
    p.add_argument("--val-batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    val_rng = np.random.default_rng(args.seed + 777)

    device = torch.device(args.device)
    model = LearnedReceiver(
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ff=args.d_ff,
    ).to(device)
    print(f"model params: {model.count_params():,}")
    print(f"device: {device}")

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(args.steps - args.warmup, 1))

    def warmup_lr(step):
        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / args.warmup

    best_val_ber = 1.0
    history = []
    t0 = time.time()

    for step in range(args.steps):
        warmup_lr(step)
        batch = generate_batch(args.batch, rng)
        rx = batch["rx_grid"].to(device)
        tx = batch["tx_bits"].to(device)
        data_mask = batch["data_mask"].to(device)

        B, T, S, _ = rx.shape
        target = cells_to_bit_targets(tx, data_mask, (B, T, S))
        logits = model(rx)
        loss, errs, nbits = compute_loss_and_errs(logits, target, data_mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step >= args.warmup:
            sched.step()

        if step % args.log_every == 0:
            ber = errs / max(nbits, 1)
            lr = opt.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"step {step:6d}  loss {loss.item():.4f}  "
                  f"batch_BER {ber:.3e}  lr {lr:.2e}  "
                  f"elapsed {elapsed:6.1f}s")

        if step and step % args.val_every == 0:
            model.eval()
            v_errs = v_bits = 0
            v_loss = 0.0
            with torch.no_grad():
                for _ in range(args.val_batches):
                    vb = generate_batch(args.val_batch_size, val_rng)
                    vrx = vb["rx_grid"].to(device)
                    vtx = vb["tx_bits"].to(device)
                    vmask = vb["data_mask"].to(device)
                    vB, vT, vS, _ = vrx.shape
                    vtarget = cells_to_bit_targets(vtx, vmask, (vB, vT, vS))
                    vlogits = model(vrx)
                    vl, ve, vn = compute_loss_and_errs(vlogits, vtarget, vmask)
                    v_loss += vl.item()
                    v_errs += ve
                    v_bits += vn
            model.train()
            val_ber = v_errs / max(v_bits, 1)
            val_loss = v_loss / args.val_batches
            print(f"=== step {step}  VAL  loss {val_loss:.4f}  BER {val_ber:.3e} ===")
            history.append({"step": step, "val_loss": val_loss, "val_ber": val_ber})
            if val_ber < best_val_ber:
                best_val_ber = val_ber
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "step": step,
                    "val_ber": val_ber,
                }, os.path.join(args.out, "best.pt"))
                print(f"    -> saved best.pt  (BER {val_ber:.3e})")

    # Final save
    torch.save({
        "model": model.state_dict(),
        "args": vars(args),
        "step": args.steps,
        "val_ber": best_val_ber,
        "history": history,
    }, os.path.join(args.out, "final.pt"))
    print(f"Done. best val BER {best_val_ber:.3e}. Saved to {args.out}/")


if __name__ == "__main__":
    main()
