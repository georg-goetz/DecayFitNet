"""
The file contains useful functions such as plotting, padding, and MSE calculations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import List
import warnings


def plot_fft(signal: torch.Tensor, fs: int, title: str = 'Frequency Response'):
    """Plots the frequency response (magnitude) of a signal. """
    assert len(signal.shape) == 2, 'The signal should be [channels, timesteps] for plotting.'
    n = signal.shape[-1]
    T = 1 / fs
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(n / 2))
    fig = plt.figure()
    for channel in range(signal.shape[0]):
        yf = 20 * np.log10(np.abs(scipy.fft.fft(signal[channel, :].numpy())))
        plt.plot(xf, yf.reshape(-1)[:n // 2])
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlim([63, fs / 2])
    plt.ylim([-80, 10])
    plt.xticks([125, 250, 500, 1000, 2000, 4000, 8000], [125, 250, 500, 1000, 2000, 4000, 8000])
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    fig.suptitle(title)
    plt.show()


def plot_waveform(waveform: torch.Tensor, sample_rate: int, title: str = "Waveform",
                  xlim: List = None, ylim: List = None):
    """Plots a waveform in time domain."""
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(dim=-2)  # Add freq_band
    assert len(waveform.shape) == 3, 'The signal should be [channels, freqs, timesteps]'
    waveform = waveform.numpy()
    num_channels, freq_bands, num_frames = waveform.shape[-3::]
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(1, num_channels, sharey=True)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        for freq in range(freq_bands):
            axes[c].plot(time_axis, waveform[c, freq, :], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def calc_mse(ground_truth_edc, estimated_edc):
    # Calculate MSE between ground truth and estimated fit. Returns mse in frequency bands
    loss_fn = torch.nn.MSELoss(reduction='none')
    this_mse = torch.mean(loss_fn(10 * torch.log10(ground_truth_edc), 10 * torch.log10(estimated_edc)), 2)
    print('==== Average MSE between input EDCs and estimated fits: {:.02f} dB ===='.format(float(torch.mean(this_mse))))
    this_mse_bands = this_mse.squeeze().tolist()
    print('MSE between input EDC and estimated fit for different frequency bands: 125 Hz: {:.02f} dB -- '
          '250 Hz: {:.02f} dB -- 500 Hz: {:.02f} dB -- 1 kHz: {:.02f} dB -- 2 kHz: {:.02f} dB -- '
          '4 kHz: {:.02f} dB'.format(*this_mse_bands))
    if torch.mean(this_mse) > 5:
        warnings.warn('High MSE value detected. The obtained fit may be bad.')
        print('!!! WARNING !!!: High MSE value detected. The obtained fit may be bad. You may want to try:')
        print('1) Increase fadeout_length. This decreases the upper limit of integration, thus cutting away more from '
              'the end of the EDC. Especially if your RIR has fadeout windows or very long silence at the end, this can'
              'improve the fit considerably.')
        print('2) Manually cut away direct sound and potentially strong early reflections that would cause the EDC to '
              'drop sharply in the beginning.')

    return this_mse


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(model, filename, device):
    model.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
    print('Model loaded from %s.' % filename)
    model.to(device)
    model.eval()
