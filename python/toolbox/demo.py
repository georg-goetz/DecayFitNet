## Demo for the DecayNet Toolbox

import torch
import torchaudio
import numpy as np
import os
import matplotlib.pyplot as plt

from decaynet_toolbox import DecaynetToolbox
from utils import plot_waveform, plot_fft

## ===============================================================================
## Parameters
audio_path = '/m/cs/work/falconr1/datasets/MusicSamples'
audio_path = '/Volumes/scratch/work/falconr1/datasets/MusicSamples'
rir_fname = 'Single_503_1_RIR.wav'
audio_path = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/summer830/raw_rirs'
rir_fname = '0825_1_raw_rirs.wav'
rir_fname = '0825_4_raw_rirs.wav'
rir_fname = '0001_4_raw_rirs.wav'
rir_fname = '0001_1_raw_rirs.wav'  # First measurement

## ===============================================================================
# Load some impulse
impulse, fs = torchaudio.load(os.path.join(audio_path, rir_fname))
if len(impulse.shape) > 1:
    impulse = impulse[0:1,:]

# Prepare the model
decaynet = DecaynetToolbox(sample_rate=fs)

# Process
prediction = decaynet.estimate_parameters(impulse)
generated_edc = decaynet.estimate_EDC(prediction[0],
                                      prediction[1],
                                      prediction[2],
                                      prediction[3])

# Plot
plot_waveform(impulse, fs, title='Impulse')
plot_waveform(generated_edc.permute([1,0,2]), fs, title='Generated EDC, linear')
plot_waveform(10 * torch.log10(generated_edc.permute([1,0,2])), sample_rate=2400, title='Generated EDC, dB')

