# ofdm_recovery

Step 1 of the Option C arc: a clean, inspectable **LTE-5 MHz OFDM subframe
generator** with AWGN, plus three visual diagnostics. No transformer, no
demodulator yet — just a correct signal we can trust.

## Why this folder exists

The long-term goal is a *standard-agnostic* RF sniffer: a transformer that
ingests arbitrary baseband IQ, finds where a signal sits in the band, and
demodulates it without being told the standard or given pilots. Before any
modeling, we need a realistic, spec-accurate signal to feed it. This folder
is that foundation.

## LTE-5 MHz FDD parameters (normal CP)

| Parameter | Value |
|---|---|
| Sample rate `Fs` | 7.68 MHz |
| FFT size `N_FFT` | 512 |
| Subcarrier spacing | 15 kHz |
| Active subcarriers | 300 (25 RBs × 12) |
| Subcarrier layout | indices `1..150` and `362..511`; DC + guards zeroed |
| CP (1st symbol of each slot) | 40 samples |
| CP (other symbols) | 36 samples |
| Symbols per subframe | 14 (2 slots × 7) |
| Samples per subframe | 7680 |
| Modulation | QPSK, Gray-coded, `{±1 ± 1j}/√2` |

## Files

- `lte_params.py` — single source of truth for the constants above
- `ofdm_signal.py` — `generate_subframe(snr_db, rng)`, `strip_cp_and_fft(...)`
- `visualize.py` — `plot_time_domain`, `plot_spectrogram`, `plot_constellation`
- `demo.py` — end-to-end waveform script
- `baseline_receiver.py` — classical CP-strip → FFT → QPSK-slice receiver,
  plus pilot-aware channel-estimating receiver (LS + freq/time interp)
- `ber_sweep.py` — BER vs SNR sweep across AWGN and EPA fading
- `channel.py` — 3GPP EPA Rayleigh block-fading channel
- `pilots.py` — CRS-style pilot pattern (symbols {0,4,7,11}, every 6th SC)

## Running the demo

```bash
python demo.py --snr 20 --out figures
```

Outputs three PDFs and prints a sanity summary:

```
sample count          : 7680  (expected 7680)
tx guard max |X|      : 0.000e+00  (expect 0)
round-trip max err    : 8.7e-16   (expect ~1e-12)
measured per-SC SNR   : +20.03 dB (target +20.0 dB)
Parseval (symbol 0)   : LHS 300.000  RHS 300.000
total tx freq energy  : 4200.0    (expected 14 × 300)
```

## Expected figures

- **time_domain.pdf** — first 512 samples of I and Q, clean vs noisy overlaid.
- **spectrogram.pdf** — STFT magnitude in dB. Energy occupies roughly
  ±2.25 MHz around DC; noise floor outside.
- **constellation.pdf** — post-FFT scatter of all 14 × 300 = 4200 active-
  subcarrier samples. At `--snr 40`, four tight clusters at `(±1 ± 1j)/√2`.
  At `--snr 5`, four blurry but still separable clusters.

## BER baseline

```bash
python ber_sweep.py --snr-min -2 --snr-max 12 --snr-step 1 --n-subframes 50
```

Produces `figures/ber_vs_snr.pdf`. Measured BER matches the theoretical
uncoded Gray-coded QPSK-in-AWGN curve `BER = Q(√SNR)` across four orders of
magnitude (−2 dB → 12 dB). Representative points:

| SNR (dB) | Measured BER | Theory BER |
|---:|---:|---:|
| 0  | 1.59e-01 | 1.59e-01 |
| 5  | 3.77e-02 | 3.77e-02 |
| 10 | 8.29e-04 | 7.83e-04 |
| 12 | 3.33e-05 | 3.43e-05 |

This is the reference any learned receiver must match on this task.

## Fading baseline — EPA block-fading + pilot channel estimation

```bash
python ber_sweep.py --snr-min 0 --snr-max 30 --snr-step 2 --n-subframes 80
```

Three curves on `figures/ber_vs_snr_all.pdf`:

| Curve | What it means |
|---|---|
| AWGN theory `Q(√SNR)` | upper bound — no fading |
| EPA + perfect CSI | irreducible Rayleigh-fading floor (~1/SNR slope) |
| EPA + LS + linear-freq + nearest-time interp | realistic baseline |

Representative BER (SNR 20 dB):

| Setting | BER |
|---|---:|
| AWGN | 0 (below floor) |
| EPA perfect CSI | 6.3e-03 |
| EPA LS + interp | 1.2e-02 |

The ~3 dB gap between perfect CSI and LS+interp, plus the BER plateau
around 3e-3 at high SNR, is the room any learned receiver can exploit.

## SNR convention

`snr_db` is the **post-FFT per-active-subcarrier SNR**. For numpy's
`ifft`/`fft` convention, that corresponds to time-domain complex noise
variance `σ² = 1 / (N_FFT · 10^(snr_db/10))`.

## Scope (explicitly)

In: AWGN channel, QPSK data on all 300 active subcarriers, one subframe.
Out (later steps): demodulator + BER, multi-tap channels, pilots,
spectrogram-tile transformer, CFO/phase noise, higher-order modulations.
