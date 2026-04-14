# Sinusoid Recovery with a Tiny Transformer

Recover a clean sinusoid from noisy observations using a small transformer (~29K parameters) trained as an autoregressive next-sample predictor. This is a bridge between classical DSP (AR/MA models, PACF) and modern sequence modeling.

## Motivation

In classical time-series analysis, we fit an **AR (AutoRegressive)** or **MA (Moving Average)** model by finding filter taps such that the residual is white noise. The number of taps is guided by the **PACF** (Partial AutoCorrelation Function).

**Key DSP fact:** A pure sinusoid at angular frequency ω₀ satisfies the second-order linear recurrence

```
x[n] = 2·cos(ω₀)·x[n-1] - x[n-2]
```

i.e., it is an **AR(2)** process. So in principle, only 2 past samples are needed to predict the next one.

**Thesis:** A transformer doing next-sample prediction on a sequence of noisy samples should *implicitly* learn this AR structure. Self-attention acts as a data-dependent filter, and the learned attention weights over past positions should correspond to AR coefficients.

## Architecture overview

```
   noisy sample x[t] (scalar)
            |
      Linear(1, 32)          <- input projection (replaces token embedding)
            |
     + PosEmbedding[t]
            |
   ┌────────▼────────┐
   │ TransformerBlock│       <- 2 layers, causal self-attention
   │ TransformerBlock│          (reused from ch04 GPT code)
   └────────┬────────┘
            |
       LayerNorm
            |
      Linear(32, 1)          <- output head (scalar, not vocab)
            |
     predicted x̂[t+1]
```

**Configuration** (in `ts_transformer.py`):

| Field | Value | Why |
|---|---|---|
| `context_length` | 128 | Covers ≥1 period of the lowest frequency (1 Hz at fs=100 → 100 samples) |
| `emb_dim` | 32 | Each token is a scalar; small representation suffices |
| `n_heads` | 2 | One head can latch onto lag-1, another on lag-2 |
| `n_layers` | 2 | Minimal depth; the problem is near-linear |
| `drop_rate` | 0.05 | Light regularisation |

Total trainable parameters: **~29K** — ~4 orders of magnitude smaller than a typical LLM.

## Pipeline walkthrough

### 1. Data generation (`signal_dataset.py`)

Each training example is generated **on the fly**. For every `__getitem__` call:

1. Sample **freq** ∼ U(1, 20) Hz, **amplitude** ∼ U(0.5, 2.0), **phase** ∼ U(0, 2π)
2. Build time vector `t = [0, 1, ..., N] / fs` with `fs = 100` Hz (well above Nyquist)
3. Compute the **clean** signal:
   ```
   clean[n] = A · sin(2π·f·t[n] + φ)
   ```
4. Inject **amplitude noise** (AWGN on A) and **phase noise** (AWGN on φ):
   ```
   noisy[n] = (A + ε_A[n]) · sin(2π·f·t[n] + φ + ε_φ[n])
   ```
   where `ε_A ∼ N(0, 0.2)` and `ε_φ ∼ N(0, 0.1)` by default.
5. Return:
   - `input_seq  = noisy[0 .. N-1]`   (context)
   - `target_seq = clean[1 .. N]`     (next clean sample at every position)

This shift-by-one, parallel target-at-every-position setup mirrors standard GPT training: at position `t` the model must predict the clean signal value at time `t+1` given noisy history `[0..t]`.

Randomising frequency every batch forces the model to learn a **generic AR predictor**, not memorise one specific sinusoid.

### 2. Model forward pass (`ts_transformer.py`)

Given `x` of shape `(batch, 128)`:

1. **Unsqueeze** to `(batch, 128, 1)` so each scalar can be linearly projected.
2. **Input projection** `Linear(1 → 32)` embeds each sample.
3. **Positional embedding** `Embedding(128 → 32)` adds learned per-position vectors — this is critical: the model needs to know *when* each sample was observed to exploit periodicity.
4. Pass through two **TransformerBlock**s (imported from `transformer_blocks.py`, a verbatim extract from the repo's ch04 GPT). Each block is:
   - LayerNorm → causal MultiHeadAttention → residual
   - LayerNorm → FFN (GELU) → residual
5. Final **LayerNorm** + **output head** `Linear(32 → 1)`, then squeeze to `(batch, 128)`.

Because the attention mask is strictly causal (upper-triangular `-inf`), the prediction at position `t` only sees positions `0..t`.

### 3. Training (`train.py`)

- **Loss:** MSE between predicted scalar and clean target at every position (regression, not classification — so no cross-entropy, no tokenisation, no vocab).
- **Optimizer:** AdamW, lr = 1e-3, weight_decay = 0.01.
- **Batches:** 32 sequences per batch; 10,000 synthetic sequences per epoch (fresh each time).
- **Metric:** SNR improvement (dB) compared to a **naive persistence predictor** that uses `noisy[t]` as the prediction for `clean[t+1]`. This is a fair baseline because both the model and the naive predictor see the same information up to time `t`.

Run:
```bash
python train.py --epochs 30
```

Output files:
- `ts_transformer.pth` — trained weights
- `training_results.pdf` — loss curves (log-scale MSE) + SNR improvement over steps

Observed convergence: MSE drops from ~1.4 → ~0.03 over 30 epochs on CPU; final output SNR ≈ naive-baseline SNR, meaning the model successfully predicts the next *clean* sample from noisy history (which is strictly harder than just matching the current noisy value).

### 4. Analysis (`analyze.py`)

Four diagnostics connect the learned model back to DSP concepts:

#### a. Attention heatmaps (`attention_analysis.pdf`)
For each test frequency, feed a clean sinusoid into the trained model and plot the attention matrix `(query × key)` for each layer. We expect a strong diagonal band just below the main diagonal — indicating queries attend to a few recent keys.

#### b. Attention vs lag profile (`attention_lag_profile.pdf`)
Collapse the 2-D attention matrix into a 1-D profile: *average attention weight as a function of lag `q - k`*. For an AR(2) process, this should peak at **lag 1 and lag 2**. The theoretical AR(1) coefficient `2·cos(ω₀)` is printed for comparison — it changes with frequency, which is why the model must be *frequency-aware* (hence randomising training frequencies).

#### c. Denoising comparison (`denoising_XHz.pdf`)
Time-domain overlay: noisy input vs model prediction vs clean target. Bottom panel is the FFT magnitude — the noisy spectrum has a broadband floor while the model's output concentrates energy at the fundamental frequency, similar to a narrowband filter.

#### d. Autoregressive generation (`autoreg_XHz.pdf`)
Feed the model a short noisy seed, then let it extend the signal by *consuming its own predictions* (sliding window of the last 128 samples). Because the model has learned the AR(2) structure for that frequency, the generated samples lock onto a clean sinusoid — visible as the output converging to the clean reference past the seed/generation boundary.

Run:
```bash
python analyze.py --test-freqs 2.0 5.0 10.0 18.0
```

## File structure

```
sinusoid_recovery/
├── README.md                   (this file)
├── transformer_blocks.py       Transformer building blocks (MultiHeadAttention,
│                               TransformerBlock, LayerNorm, GELU, FeedForward).
│                               Copied verbatim from ch04/01_main-chapter-code/gpt.py
│                               to avoid a tiktoken import.
├── signal_dataset.py           Noisy sinusoid generator + PyTorch Dataset/DataLoader
├── ts_transformer.py           TimeSeriesTransformer (scalar-in, scalar-out GPT variant)
├── train.py                    MSE training loop, SNR metric, loss plotting
├── analyze.py                  Attention analysis, denoising plots, autoregressive gen
├── ts_transformer.pth          (generated) trained model weights
├── training_results.pdf        (generated) loss curves
├── attention_analysis.pdf      (generated) attention heatmaps
├── attention_lag_profile.pdf   (generated) attention-vs-lag bar plots
├── denoising_*Hz.pdf           (generated) denoising comparison at test freqs
└── autoreg_*Hz.pdf             (generated) autoregressive generation plots
```

## How to reproduce end-to-end

```bash
cd sinusoid_recovery

# 1. Sanity-check the model definition
python ts_transformer.py

# 2. Train (30 epochs on CPU takes a few minutes)
python train.py --epochs 30 --train-size 10000

# 3. Generate all analysis plots
python analyze.py --test-freqs 2.0 5.0 10.0 18.0
```

## Connection to AR/MA theory — what to look for

| DSP concept | Transformer analogue |
|---|---|
| AR order (how many lags matter) | Effective receptive field of attention — read off from the attention-vs-lag profile |
| AR coefficients | Encoded jointly in the attention weights, value projections, and FFN |
| PACF cutoff | Lag beyond which attention weight falls to ~0 |
| Fitting AR by least squares (Yule-Walker) | Gradient descent on MSE loss |
| Wiener filter / Kalman filter | The trained model acts as a learned non-linear filter; for a pure sinusoid + AWGN both should converge to essentially the same optimum |

## Caveats and honest limitations

- **This problem is massively over-engineered for an LLM.** A 2-tap linear AR(2) estimator (Yule-Walker on a buffer of a few hundred samples) will match or beat this model at a tiny fraction of the compute. The point of this exercise is *pedagogical*: showing that a transformer naturally absorbs AR structure.
- **Generalisation is bounded by the training distribution.** Frequencies outside [1, 20] Hz won't be handled well; neither will non-stationary signals or multi-tone mixtures.
- **FFT as embedding** (from the original proposal) is not implemented here — raw scalar + positional embedding was sufficient. A short-time FFT front-end would be the natural next experiment for more complex signals.
- **d_model ↔ PACF mapping is loose.** PACF governs *how far back* to look (→ `context_length`), not the per-position representation width (`emb_dim`). For a pure sinusoid, `context_length` could in principle be as small as ~3.
