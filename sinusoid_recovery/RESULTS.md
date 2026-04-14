# Low-SNR Experiments and Parameter-Tuning Observations

This document summarizes empirical observations on how the tiny transformer
in this folder behaves as the input SNR drops, and what architectural levers
matter most when you need to recover usable signal from heavy noise.

All experiments use the next-sample-prediction setup from `train.py`, with
randomized sinusoids (freq 1–20 Hz, amplitude 0.5–2.0, phase 0–2π) at
`fs = 100 Hz` and a context length of 128 (unless scaled).

Scripts:
- [`snr_sweep.py`](snr_sweep.py) — evaluate a trained model across noise levels
- [`retrain_lowsnr.py`](retrain_lowsnr.py) — retrain at hard noise, base vs scaled

---

## 1. How the original model degrades as noise increases

Model: `TS_TRANSFORMER_CONFIG` (context 128, emb 32, 2 heads, 2 layers, 29 K params),
trained at `amp_std=0.2, phase_std=0.1` for 30 epochs.

| `amp_std` | `phase_std` | Input SNR (dB) | Output SNR (dB) | Gain (dB) | Val MSE |
|---|---|---|---|---|---|
| 0.10 | 0.05 | 3.05 | 17.31 | **+14.26** | 0.017 |
| 0.20 | 0.10 | 2.85 | 14.65 | **+11.80** | 0.031 |
| 0.40 | 0.20 | 2.14 | 9.61 | **+7.47** | 0.098 |
| 0.60 | 0.30 | 1.17 | 5.90 | **+4.72** | 0.231 |
| 0.80 | 0.40 | 0.12 | 3.26 | **+3.14** | 0.424 |
| 1.00 | 0.50 | -0.93 | 1.37 | **+2.30** | 0.654 |

**Observations**

- Gain collapses from +11.8 dB at the training noise to +2.3 dB at 5× noise.
- Two effects are entangled here: (i) the harder physical problem, and
  (ii) the train/test distribution mismatch.

---

## 2. Retraining at hard noise — same arch vs scaled arch

Hard setting: `amp_std=0.8, phase_std=0.4` (input SNR ≈ 0 dB). Trained for
10 epochs with 4000 sequences/epoch, AdamW, lr 1e-3.

| Scenario | Arch (ctx / emb / heads / layers) | Params | Train time | Out SNR | Gain | Val MSE |
|---|---|---|---|---|---|---|
| Original (no retrain) | 128 / 32 / 2 / 2 | 29 K | — | 3.26 dB | **+3.14** | 0.424 |
| (A) Base, retrained | 128 / 32 / 2 / 2 | 29 K | 144 s | 3.81 dB | **+3.69** | 0.368 |
| (B) Scaled, retrained | 256 / 72 / 4 / 3 | 208 K | 1213 s | 5.85 dB | **+6.00** | 0.220 |

Training trajectory (val SNR improvement per epoch):

| Epoch | (A) Base | (B) Scaled |
|---|---|---|
| 1 | +2.0 | +2.1 |
| 3 | +2.2 | +2.6 |
| 5 | +2.9 | +3.2 |
| 7 | +3.0 | +4.4 |
| 9 | +3.5 | +5.5 |
| 10 | +3.7 | +5.9 |

**Observations**

- **Retraining alone barely helps** (+3.14 → +3.69 dB). The base architecture
  is saturated; a pure AR(2) predictor has little room to average noise.
- **Capacity buys real gain**: 7× parameters and 2× context → +2.3 dB extra.
  The longer window is the dominant lever.
- (B) was still climbing at epoch 10 — with more training it likely reaches
  +7 to +8 dB.
- Neither config recovers the +11.8 dB gain seen at easy noise. Heavy phase
  noise makes the instantaneous frequency wobble, which is a genuinely
  harder signal-recovery problem.

---

## 3. Parameter-tweaking recommendations (DSP-grounded)

At lower SNR, **averaging more samples** is the dominant denoising mechanism.
The three DSP-anchored hyperparameters shift as follows:

| Parameter | Easy-noise rule | Low-SNR rule | Why |
|---|---|---|---|
| **`context_length`** | `≈ fs / f_min` | `≈ (fs / f_min) · k`, k ≈ 2–4 | Extra √k noise averaging |
| **`n_heads`** | = effective AR order | AR order + smoothing/MA heads | FIR-style low-pass averaging beyond AR |
| **`head_dim`** | `≈ 2 · log₂(ctx)` | same rule, scales with `ctx` | Keeps per-position discrimination under noise |
| **`n_layers`** | 2 | 3–4 | Iterative denoising refinement |
| **`emb_dim`** | falls out | falls out | `= n_heads · head_dim` |

Non-architectural levers (all three matter):

1. **Match training noise to deployment noise.** Out-of-distribution noise
   is the single largest source of degradation.
2. **Curriculum or mixed-σ training.** Sample noise from a range so the
   model is robust across a span of SNRs rather than specialized to one point.
3. **Train longer.** On CPU, (B) needed ~20 minutes for 10 epochs; a full
   30–40 epochs is feasible and typically adds 1–2 dB.

---

## 4. FFT-embedding front-end — a much stronger inductive bias

Replacing the scalar `Linear(1, emb_dim)` input projection with a
**causal sliding-window FFT** changes the picture dramatically. At every
position `t`, the token now carries the FFT of the last `W=32` samples
(real + imag bins), then projected to `emb_dim`. Everything downstream
(positional embedding, attention, FFN, head) is unchanged.

Implementation: [`ts_transformer_fft.py`](ts_transformer_fft.py).
Comparison driver: [`compare_fft.py`](compare_fft.py).

### 4a. Easy noise (`amp_std=0.2, phase_std=0.1`), 8 epochs, ~30 K params

| Variant | Params | Final MSE | Out SNR | Gain | Train time |
|---|---|---|---|---|---|
| Scalar `Linear(1, 32)` | 29,473 | 0.091 | 9.92 dB | **+7.31 dB** | 120 s |
| **FFT sliding window** | 30,529 | 0.021 | 16.39 dB | **+13.78 dB** | 113 s |

Convergence comparison (val SNR gain by training step):

| Step | Scalar | FFT |
|---|---|---|
| 0 | -4.6 | -3.6 |
| 100 | +0.8 | **+7.8** |
| 300 | +2.2 | +11.3 |
| 500 | +4.0 | +12.1 |
| 700 | +5.5 | +13.0 |
| 900 | +6.7 | +13.5 |

FFT reaches the scalar model's final result after just 100 steps — roughly
8× faster wall-clock to a given SNR target.

### 4b. Hard noise (`amp_std=0.8, phase_std=0.4`), 10 epochs, ~30 K params

| Variant | Params | Final MSE | Out SNR | Gain |
|---|---|---|---|---|
| Scalar `Linear(1, 32)` | 29,473 | 0.368 | 3.81 dB | **+3.69 dB** |
| **FFT sliding window** | 30,529 | 0.108 | 9.12 dB | **+9.00 dB** |

Cross-reference: the **scaled scalar model** from §2 — 7× larger (208 K params),
2× context, 4 heads, 3 layers — only reached **+6.0 dB** at hard noise.
The FFT model at the original size **beats it by +3 dB**.

### 4c. Why the FFT front-end helps so much

A pure sinusoid is a delta in the frequency domain. The FFT embedding
hands the transformer a **near-diagonal representation** of the task:

- Each token already encodes the spectrum of the last 32 samples
- Noisy input becomes a peak-plus-noise-floor in each token
- The transformer's job collapses to peak-tracking and bin-interpolation
- It no longer has to rediscover periodicity from raw time-domain samples

Equivalently, the FFT acts as **parallel matched filters** (one per bin)
before attention even runs — built-in processing gain.

### 4d. Trade-offs

| Aspect | Observation |
|---|---|
| Inference overhead | ~160 MACs/sample for the sliding FFT; negligible vs ~50 K MACs inside the blocks |
| Inductive bias | Strong — ideal for stationary/periodic signals; less safe for chirps or impulsive signals |
| Window size `W` | Classic STFT trade-off: large `W` = better frequency resolution, small `W` = better time resolution |
| Convergence | Reaches good performance in ~1 epoch; early-stopping friendly |
| Interpretability | Tokens are physically meaningful (FFT bins) — attention maps are easier to read |

### 4e. When to choose which embedding

- **FFT embedding** — band-limited / periodic / stationary signals. Huge sample-efficiency and SNR-gain win.
- **Scalar embedding** — general-purpose default when signal structure is unknown.
- **Learned `Conv1d(1, emb_dim, kernel_size=W)`** — middle ground: same compute as FFT, no frequency prior, model learns the filterbank.

---

## 5. Perspective vs. classical DSP

At input SNR ≈ 0 dB, a Kalman filter on the known sinusoid model typically
delivers **+8 to +12 dB** gain at ~30 MACs per output sample. The scaled
transformer in (B) reached **+6 dB** at millions of MACs per sample.

The transformer's value here is not efficiency — it is architectural
flexibility. The efficiency gap only closes when the signal model becomes
rich or uncertain enough that a compact classical estimator cannot be
written down (multi-tone, non-Gaussian noise, regime switches, chirps, etc.).

---

## 6. How to reproduce

```bash
# Evaluate the released model across a noise sweep
python snr_sweep.py --model-path ts_transformer.pth

# Retrain base and scaled configs at hard noise
python retrain_lowsnr.py --amp-std 0.8 --phase-std 0.4 --epochs 10

# Compare scalar-embedding vs FFT-embedding at matched param budget
python compare_fft.py --amp-std 0.2 --phase-std 0.1 --epochs 8
python compare_fft.py --amp-std 0.8 --phase-std 0.4 --epochs 10
```

Numbers will vary slightly run-to-run because the validation set is generated
on the fly.
