'''
The file contains useful functions such as plotting and padding.
'''

import torch
import matplotlib.pyplot as plt
import numpy as np
import scipy
from typing import Iterable, Tuple, TypeVar, Callable, Any, List


def plot_fft(signal: torch.Tensor, fs: int, title: str ='Frequency Response'):
    '''Plots the frequency response (magnitude) of a signal. '''
    assert len(signal.shape) == 2, 'The signal should be [channels, timesteps] for plotting.'
    n = signal.shape[-1]
    T = 1 / fs
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(n / 2))
    fig = plt.figure()
    for channel in range(signal.shape[0]):
        yf = 20 * np.log10(np.abs(scipy.fft.fft(signal[channel,:].numpy())))
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
    '''Plots a waveform in time domain.'''
    if len(waveform.shape) == 2:
        waveform = waveform.unsqueeze(dim=-2)   # Add freq_band
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
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()
