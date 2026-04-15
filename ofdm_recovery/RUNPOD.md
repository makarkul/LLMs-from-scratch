# RunPod runbook — step 4a

This is the exact sequence to train the learned OFDM receiver on a RunPod
GPU and push the results back to GitHub.

## 1. Spin up the pod

On runpod.io:

- **GPU**: RTX 4090 (Community Cloud) — ~$0.35–0.50 / hr.
  Alternatives that also work fine: RTX 3090, A4000.
- **Template**: any recent PyTorch template, e.g.
  `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`.
- **Container disk**: 20 GB (default is usually fine).
- **Volume**: not needed for a single run.
- **Expose**: SSH on 22 is enough (no Jupyter required). TCP is fine.

Once the pod is Running, connect via the web-terminal or SSH.

## 2. Clone, install, train, eval, push

```bash
# --- one-time setup ---
apt-get update && apt-get install -y git
git config --global user.email "you@example.com"
git config --global user.name  "your name"

git clone https://github.com/<YOUR_USER>/LLMs-from-scratch.git
cd LLMs-from-scratch
git checkout ofdm-recovery
cd .claude/worktrees/nice-banach/ofdm_recovery

pip install -r requirements.txt

# --- quick GPU sanity check ---
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# --- train (~10–20 min on a 4090) ---
python train.py --steps 30000 --batch 32 --out checkpoints/

# --- BER sweep ---
python evaluate.py --ckpt checkpoints/best.pt --out figures/

# --- push results ---
cd ../../../..                                # back to repo root
git add .claude/worktrees/nice-banach/ofdm_recovery/checkpoints/best.pt \
        .claude/worktrees/nice-banach/ofdm_recovery/figures/*.pdf
git commit -m "step 4a: trained learned receiver + BER sweep"

# If this is the first push from the pod you'll need a Personal Access Token.
# On github.com -> Settings -> Developer settings -> Tokens (classic) ->
#   scope repo, expiration short. Then:
git push https://<YOUR_USER>:<TOKEN>@github.com/<YOUR_USER>/LLMs-from-scratch.git ofdm-recovery
```

## 3. Shut down the pod

Stop or terminate the pod on the RunPod dashboard so you stop billing.
Checkpoint and figures are now on GitHub.

## Expected results

A successful run should print (roughly):

```
step     0  loss 0.73   batch_BER 5.1e-01
step  5000  loss 0.30   batch_BER 2.0e-01
step 20000  loss 0.20   batch_BER 1.1e-01
step 30000  loss 0.18   batch_BER 9e-02
=== VAL  BER  around 1e-01 at end (averaged over full SNR range 0–25 dB) ===
```

The evaluate step should show learned-receiver BER sitting **between the
EPA LS+interp curve and the EPA perfect-CSI curve** at high SNR. That's
the step-4a bar: "beat LS+interp."

## If something goes wrong

- **CUDA OOM**: reduce `--batch 16`.
- **Slow batch gen**: the data generator is numpy, single-threaded. If
  data loading is >30% of step time on GPU, we can vectorize later.
- **Checkpoint too big for git**: best.pt should be ~3–4 MB for the default
  arch, no LFS needed. If you changed d_model and it balloons, either use
  git-lfs or `scp` the file off the pod.

## Local iteration while GPU run is cooking

No need to wait — back on your laptop you can:
- Review `evaluate.py` logic
- Design step 4b (spectrogram-tile / standard-agnostic version)
- Tune the BER sweep parameters
