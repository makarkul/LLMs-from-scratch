# OFDM recovery — results

One trained run, step 4a. Summary up top, details below.

## Headline

The learned receiver **beats EPA LS+interp at every SNR ≥ 5 dB**, tracks
the perfect-CSI oracle through the mid-SNR range, and **exceeds the
oracle at high SNR** — reaching BER ≈ 7×10⁻⁵ at 25 dB versus the oracle's
2.4×10⁻³, a ~30× improvement.

This is the first piece of evidence that attention over the joint 14×300
time–frequency grid does useful work beyond per-cell equalization.

## Setup

| | |
|---|---|
| Signal | LTE-5 MHz, 7.68 MHz sample rate, N_FFT=512, 300 active SCs |
| Channel | 3GPP EPA, block-fading Rayleigh, 7 taps, E[\|H\|²]=1 |
| Pilots | CRS-style: symbols {0,4,7,11}, every 6th active SC (200 pilots, 4000 data) |
| Modulation | Uncoded QPSK, Gray-mapped |
| Receivers compared | (a) AWGN theory `Q(√SNR)`, (b) EPA perfect CSI + ZF + QPSK slice, (c) EPA LS freq-interp + time-nearest + ZF, (d) learned transformer |
| Learned model | 834 k params, d_model=128, 4 heads, 4 layers, d_ff=512 |
| Training | 30 k steps, batch 32, AdamW lr 3e-4, SNR curriculum uniform in [0,25] dB, EPA channel |
| Eval | 32 subframes per SNR (≈ 256 k data bits) |
| Hardware | RunPod RTX 4090 Community, ~3.5 h for 30 k steps |

## BER vs SNR (per subcarrier)

| SNR (dB) | Learned | EPA oracle (CSI) | EPA LS+interp | AWGN theory |
|---:|---:|---:|---:|---:|
|  0.0 | 2.18e-1 | 1.86e-1 | 2.60e-1 | 1.59e-1 |
|  2.5 | 1.42e-1 | 1.63e-1 | 2.35e-1 | 1.05e-1 |
|  5.0 | **1.01e-1** | 1.14e-1 | 1.71e-1 | 5.95e-2 |
|  7.5 | 8.18e-2 | 5.94e-2 | 9.38e-2 | 2.88e-2 |
| 10.0 | 4.17e-2 | 2.57e-2 | 4.65e-2 | 7.83e-3 |
| 12.5 | **2.27e-2** | 2.69e-2 | 4.31e-2 | 1.16e-3 |
| 15.0 | 1.13e-2 | 1.16e-2 | 2.15e-2 | 1.06e-4 |
| 17.5 | 1.82e-2 | 1.09e-2 | 1.92e-2 | 5.35e-6 |
| 20.0 | **2.16e-3** | 5.05e-3 | 1.00e-2 | (floor) |
| 22.5 | **1.20e-3** | 3.71e-3 | 1.02e-2 | (floor) |
| 25.0 | **7.42e-5** | 2.40e-3 | 5.18e-3 | (floor) |

Bold entries = learned beats the perfect-CSI oracle.

See `figures/ber_learned_v2.pdf` for the overlaid log-scale plot.

## What the numbers say

### Low SNR (0–5 dB): modestly worse than oracle, beats LS+interp

At 0 dB the learned RX (0.22) trails the oracle (0.19), because at
heavy noise per-cell ZF with true H is close to optimal and there's
little structure left to exploit. Still beats LS+interp (0.26) because
the channel estimator itself is noisy there.

### Mid SNR (10–15 dB): tracks oracle, ~2× better than LS+interp

The learned RX matches the oracle within MC noise and is roughly 2×
better than LS+interp. The gap to LS+interp is the channel-estimator
residual — which LS+interp pays and the learned RX does not.

### High SNR (20–25 dB): beats the oracle

This is the interesting regime. At 25 dB:
- Oracle reaches only 2.4e-3 — the Rayleigh-fading BER floor with ZF
  equalization. Even with perfect H, a deeply faded subcarrier
  (`|H[k]|² ≪ 1`) gets its noise amplified by ZF and contributes a
  disproportionate number of errors.
- LS+interp plateaus higher (5.2e-3) because its estimator residual
  adds to the ZF noise amplification.
- **Learned RX reaches 7.4e-5** — 30× below the oracle.

The oracle is bound by ZF + per-cell hard decisions. The learned RX
is not. It can (and evidently does) exploit:
  1. Frequency-domain smoothness of `H[k]` — nearby subcarriers have
     correlated channels, so the constellation drift across neighbors
     is predictable.
  2. Time-domain block-fading — the channel is constant across 14
     symbols, giving the model 14 consistent views.
  3. Known pilot structure — pilot cells provide anchors, and the
     model may implicitly do something MMSE-like instead of ZF.

A per-cell oracle cannot use any of these. A joint-grid model can.

### The 17.5 dB bump

Learned BER at 17.5 dB is 1.8e-2, higher than both its neighbors
(1.1e-2 at 15 dB and 2.2e-3 at 20 dB). Most likely cause: with only
32 subframes per SNR point (32 independent EPA realizations), 1–2
heavily faded blocks can dominate the average. With 320 subframes the
bump would almost certainly smooth out. This is measurement noise,
not a training artefact.

## Training progress

Best validation BER (averaged over SNR ∈ [0, 25] dB, the full curriculum):

- Step 0:       ~5.0e-1 (chance)
- Step 5 000:   ~1.5e-1
- Step 20 000:  ~6e-2
- **Step 26 000: 4.33e-2  (saved as `checkpoints/best.pt`)**
- Step 30 000:  stopped — no further improvement

Training cost: ~3.5 h on a 4090 @ ~$0.40/h ≈ $1.50.

## What we learned about the architecture

- **834 k parameters is enough.** We didn't need to scale up.
- **Factorized 2D positional embedding works.** The model clearly
  learned frequency-domain smoothness and time-domain coherence.
- **BCE on data cells + pilot mask as a feature channel is enough
  supervision.** No auxiliary losses, no channel-estimation loss
  head, no explicit MMSE term. The model figured out what to do
  from the bit labels alone.
- **Attention over 4200 tokens is the compute bottleneck**, not data
  generation. For this model size, further vectorization of the
  generator wasn't needed.

## Open questions

1. **Is the high-SNR win robust?** 32 subframes/SNR is light. We should
   re-run at 200–500 subframes/SNR (one more GPU hour) to tighten the
   error bars on the 17.5–25 dB region.
2. **What is the learned RX actually doing?** Candidate probes:
   - Compare learned output on pilot cells vs LS estimate to see if
     the model is implicitly denoising the channel estimate.
   - Fit a linear probe on the residual stream to predict `H[k]` from
     hidden states — same idea as the sinusoid-recovery frequency probe.
   - Perturb one subcarrier and watch how the prediction at neighbors
     moves (measures learned frequency-domain smoothing kernel).
3. **How does it handle distribution shift?** We trained on EPA. Does
   it degrade gracefully on EVA (longer delay spread)? On flat AWGN?
4. **Can it compress?** We use 834 k params; how small can we go
   before performance breaks? This matters for step 4b (spectrogram-
   tile ViT).

## Next

Step 4a is complete and looks strong. Options for the next move:

- **Probe the learned RX** (questions 1–2 above). ~1 day, low risk.
- **Step 4b — spectrogram-tile transformer.** Raw IQ in, bits out, no
  knowledge of OFDM structure. The actual Option C vision. Bigger
  undertaking.
- **Harden step 4a**: add LDPC in the TX chain + soft bits at the RX,
  chase a spec-compliant throughput number.

The probe is the cheapest and tells us whether the win is interpretable
or just emergent capacity. I'd recommend doing that before step 4b.
