import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def generate_noisy_sinusoid(n_samples, freq, amplitude, phase, fs,
                            amp_noise_std=0.2, phase_noise_std=0.1):
    """Generate a noisy sinusoid and its clean version.

    Args:
        n_samples: Number of samples to generate
        freq: Sinusoid frequency in Hz
        amplitude: Sinusoid amplitude
        phase: Sinusoid phase in radians
        fs: Sampling frequency in Hz
        amp_noise_std: Std dev of additive amplitude noise
        phase_noise_std: Std dev of phase noise (radians)

    Returns:
        noisy: numpy array of noisy samples
        clean: numpy array of clean samples
    """
    t = np.arange(n_samples) / fs
    phase_noise = np.random.randn(n_samples) * phase_noise_std
    amp_noise = np.random.randn(n_samples) * amp_noise_std

    clean = amplitude * np.sin(2 * math.pi * freq * t + phase)
    noisy = (amplitude + amp_noise) * np.sin(2 * math.pi * freq * t + phase + phase_noise)

    return noisy.astype(np.float32), clean.astype(np.float32)


class SinusoidDataset(Dataset):
    """On-the-fly generation of noisy sinusoid sequences for next-sample prediction.

    Each __getitem__ generates a fresh random sinusoid with randomized
    frequency, amplitude, and phase. Returns:
        input_seq:  noisy samples [0..N-1]  shape (context_length,)
        target_seq: clean samples [1..N]    shape (context_length,)

    The model learns to predict the next *clean* sample given past noisy samples.
    """

    def __init__(self, context_length, dataset_size, fs=100.0,
                 freq_range=(1.0, 20.0), amp_range=(0.5, 2.0),
                 amp_noise_std=0.2, phase_noise_std=0.1):
        self.context_length = context_length
        self.dataset_size = dataset_size
        self.fs = fs
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.amp_noise_std = amp_noise_std
        self.phase_noise_std = phase_noise_std

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        freq = np.random.uniform(*self.freq_range)
        amplitude = np.random.uniform(*self.amp_range)
        phase = np.random.uniform(0, 2 * math.pi)

        # Generate context_length + 1 samples so we can shift by 1
        n_samples = self.context_length + 1
        noisy, clean = generate_noisy_sinusoid(
            n_samples, freq, amplitude, phase, self.fs,
            self.amp_noise_std, self.phase_noise_std
        )

        input_seq = torch.tensor(noisy[:self.context_length])    # noisy [0..N-1]
        target_seq = torch.tensor(clean[1:self.context_length+1])  # clean [1..N]

        return input_seq, target_seq


def create_sinusoid_dataloaders(context_length, train_size=10000, val_size=1000,
                                batch_size=32, fs=100.0,
                                freq_range=(1.0, 20.0), amp_range=(0.5, 2.0),
                                amp_noise_std=0.2, phase_noise_std=0.1,
                                num_workers=0):
    """Create train and validation dataloaders."""
    train_dataset = SinusoidDataset(
        context_length, train_size, fs, freq_range, amp_range,
        amp_noise_std, phase_noise_std
    )
    val_dataset = SinusoidDataset(
        context_length, val_size, fs, freq_range, amp_range,
        amp_noise_std, phase_noise_std
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=num_workers
    )

    return train_loader, val_loader
