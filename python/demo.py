## Demo for the DecayFitNet Toolbox

import torch
import torchaudio
import os
from pathlib import Path
import matplotlib.pyplot as plt

from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from toolbox.utils import plot_waveform, calc_mse
from toolbox.core import discard_last5

## ===============================================================================
## Parameters
# audio_path = '/m/cs/work/falconr1/datasets/MusicSamples'
# audio_path = '/Volumes/scratch/work/falconr1/datasets/MusicSamples'
# rir_fname = 'Single_503_1_RIR.wav'
# audio_path = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/summer830/raw_rirs'
# rir_fname = '0825_1_raw_rirs.wav'
# rir_fname = '0825_4_raw_rirs.wav'
# rir_fname = '0001_4_raw_rirs.wav'

audio_path = '/Volumes/ARTSRAM/AkuLab_Datasets/roomtransition/Wav Files/Meeting Room to Hallway/Source in Room/No Line of Sight'
rir_fname = 'RIR_25cm.wav'
fadeout_length = 0.1  # in seconds

# audio_path = '/Volumes/ARTSRAM/AkuLab_Datasets/summer830/sh_rirs'
# rir_fname = '0001_1_sh_rirs.wav'  # First measurement
# fadeout_length = 0  # in seconds

# audio_path = '/Volumes/ARTSRAM/AkuLab_Datasets/arni/sh_rirs'
# rir_fname = '03333_sh_rirs.wav'
# fadeout_length = 1  # in seconds


audio_path = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/summer830/raw_rirs'
#rir_fname = '0825_1_raw_rirs.wav'
#rir_fname = '0825_4_raw_rirs.wav'
#rir_fname = '0001_4_raw_rirs.wav'  # Huge
rir_fname = '0001_1_raw_rirs.wav'  # First measurement
fadeout_length = 1  # in seconds


audio_path = Path.joinpath(Path(__file__).parent.parent, 'model')
#rir_fname = '0825_1_raw_rirs.wav'
#rir_fname = '0825_4_raw_rirs.wav'
#rir_fname = '0001_4_raw_rirs.wav'  # Huge
rir_fname = '0001_1_raw_rirs.wav'  # First measurement
fadeout_length = 1  # in seconds

## ===============================================================================
# Load some impulse
rir, fs = torchaudio.load(os.path.join(audio_path, rir_fname))
if len(rir.shape) > 1:
    rir = rir[0:1, :]

# Delete potential fade-out windows
if fadeout_length > 0:
    rir = rir[:, 0:round(-fadeout_length*fs)]

# Prepare the model
decayfitnet = DecayFitNetToolbox(sample_rate=fs)

# Generate time axis
time_axis = (torch.linspace(0, rir.shape[1] - 1, rir.shape[1]) / fs)

# Process
prediction = decayfitnet.estimate_parameters(rir)
print('==== Estimated T values (in seconds, T=0 indicates an inactive slope): ====\n' + str(prediction[0]))
print('==== Estimated A values (linear scale, A=0 indicates an inactive slope): ====\n' + str(prediction[1]))
print('==== Estimated N values (linear scale): ====\n' + str(prediction[2]))

fitted_edc = decayfitnet.generate_EDCs(prediction[0],
                                       prediction[1],
                                       prediction[2],
                                       time_axis=time_axis)

# Get ground truth EDCs from raw RIRs:
# 1) Schroeder integration
true_edc = decayfitnet._preprocess.schroeder(rir)
# 2) Discard last 5 percent of EDC
true_edc = discard_last5(true_edc)
# 3) Permute into same order as estimated fit
true_edc = true_edc.permute(1, 0, 2)

# Calculate MSE between true EDC and fitted EDC
mse_per_frequencyband = calc_mse(true_edc, fitted_edc)

# Plot
time_axis = time_axis[0:round(0.95*len(time_axis))]  # discard last 5 percent of plot time axis
plot_waveform(rir, fs, title='Impulse')
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for band_idx in range(true_edc.shape[0]):
    plt.plot(time_axis, 10 * torch.log10(true_edc[band_idx, 0, :].squeeze()),
             colors[band_idx], label='Measured EDC, {} Hz'.format(decayfitnet.filter_frequencies[band_idx]))
    plt.plot(time_axis, 10 * torch.log10(fitted_edc[band_idx, 0, :].squeeze()),
             colors[band_idx] + '--', label='DecayFitNet fit, {} Hz'.format(decayfitnet.filter_frequencies[band_idx]))

plt.xlabel('time [s]')
plt.ylabel('energy [dB]')
plt.subplots_adjust(right=0.6)
plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
plt.show()

# plot_waveform(generated_edc.permute([1, 0, 2]), fs, title='Generated EDC, linear')
# plot_waveform(10 * torch.log10(generated_edc.permute([1, 0, 2])), sample_rate=2400, title='Generated EDC, dB')
