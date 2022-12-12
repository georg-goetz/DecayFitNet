# Demo for the DecayFitNet Toolbox

import torch
import numpy as np
import torchaudio
import os
from pathlib import Path
import matplotlib.pyplot as plt

from toolbox.DecayFitNetToolbox import DecayFitNetToolbox
from toolbox.utils import calc_mse
from toolbox.core import discard_last_n_percent, decay_model, PreprocessRIR

from toolbox.BayesianDecayAnalysis import BayesianDecayAnalysis

# ===============================================================================
# Parameters
audio_path = os.path.join(Path(__file__).parent.parent.resolve(), 'data', 'exampleRIRs')
rir_fname = 'doubleslope_0178_3_sh_rirs.wav'  # First measurement
fadeout_length = 0  # in seconds

n_slopes = 0  # 0 = number of active slopes is determined by network or bayesian analysis (between 1 and 3)

filter_frequencies = [125, 250, 500, 1000, 2000, 4000]

# Bayesian paramters: a_range and n_range are both exponents, i.e., actual range = 10^a_range or 10^n_range
parameter_ranges = {'t_range': [0.1, 3.5],
                    'a_range': [-3, 0],
                    'n_range': [-10, -2]}
n_iterations = 100

# ===============================================================================
# Load impulse response
rir, fs = torchaudio.load(os.path.join(audio_path, rir_fname))

# Use only omni channel
if len(rir.shape) > 1:
    rir = rir[0:1, :]

# Delete potential fade-out windows
if fadeout_length > 0:
    rir = rir[:, 0:round(-fadeout_length * fs)]

# ===============================================================================
# Get ground truth EDCs from raw RIRs:

# Init Preprocessing
rir_preprocessing = PreprocessRIR(sample_rate=fs, filter_frequencies=filter_frequencies)
# Schroeder integration
true_edc, __ = rir_preprocessing.schroeder(rir)
time_axis = (torch.linspace(0, true_edc.shape[2] - 1, true_edc.shape[2]) / fs)
# Permute into [n_bands, n_batches, n_samples]
true_edc = true_edc.permute(1, 0, 2)

# ===============================================================================
# Analyze with DecayFitNet

# Prepare the model
decayfitnet = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs, filter_frequencies=filter_frequencies)

# Process
estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rir)
print('==== DecayFitNet: Estimated T values (in seconds, T=0 indicates an inactive slope): ====\n' +
      str(estimated_parameters_decayfitnet[0]))
print('==== DecayFitNet: Estimated A values (linear scale, A=0 indicates an inactive slope): ====\n' +
      str(estimated_parameters_decayfitnet[1]))
print('==== DecayFitNet: Estimated N values (linear scale): ====\n' +
      str(estimated_parameters_decayfitnet[2]))

# Get fitted EDC from estimated parameters
fitted_edc_decayfitnet = decay_model(torch.from_numpy(estimated_parameters_decayfitnet[0]),
                                     torch.from_numpy(estimated_parameters_decayfitnet[1]),
                                     torch.from_numpy(estimated_parameters_decayfitnet[2]),
                                     time_axis=time_axis,
                                     compensate_uli=True,
                                     backend='torch')

# Discard last 5% for MSE evaluation
true_edc = discard_last_n_percent(true_edc, 5)
fitted_edc_decayfitnet = discard_last_n_percent(fitted_edc_decayfitnet, 5)

# Calculate MSE between true EDC and fitted EDC
mse_per_frequencyband = calc_mse(true_edc, fitted_edc_decayfitnet)

# Plot
time_axis_excl5 = time_axis[0:round(0.95 * len(time_axis))]  # discard last 5 percent of plot time axis
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for band_idx in range(true_edc.shape[0]):
    plt.plot(time_axis_excl5, 10 * torch.log10(true_edc[band_idx, 0, :].squeeze()),
             colors[band_idx], label='Measured EDC, {} Hz'.format(filter_frequencies[band_idx]))
    plt.plot(time_axis_excl5, 10 * torch.log10(fitted_edc_decayfitnet[band_idx, 0, :].squeeze()),
             colors[band_idx] + '--', label='DecayFitNet fit, {} Hz'.format(filter_frequencies[band_idx]))

plt.xlabel('time [s]')
plt.ylabel('energy [dB]')
plt.subplots_adjust(right=0.6)
plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
plt.title('DecayFitNet')
plt.show()

# How to change the center frequencies manually (can also be set directly in init of DecayFitNet)
decayfitnet.set_filter_frequencies([0, 125, 250, 500, 1000, 2000, 4000, fs / 2])
estimated_parameters_decayfitnet, norm_vals_decayfitnet = decayfitnet.estimate_parameters(rir)
print('==== DecayFitNet: Estimated T values (in seconds, T=0 indicates an inactive slope): ====\n'
      + str(estimated_parameters_decayfitnet[0]))
print('==== DecayFitNet: Estimated A values (linear scale, A=0 indicates an inactive slope): ====\n'
      + str(estimated_parameters_decayfitnet[1]))
print('==== DecayFitNet: Estimated N values (linear scale): ====\n'
      + str(estimated_parameters_decayfitnet[2]))

# ===============================================================================
# Bayesian decay analysis

# Init Bayesian decay analysis
bda = BayesianDecayAnalysis(n_slopes, fs, parameter_ranges, n_iterations)

# Estimate decay parameters
estimated_parameters_bayesian, norm_vals_bayesian = bda.estimate_parameters(rir)

print('==== Bayesian decay analysis: Estimated T values (in seconds, T=0 indicates an inactive slope): ====\n' +
      str(estimated_parameters_bayesian[0]))
print('==== Bayesian decay analysis: Estimated A values (linear scale, A=0 indicates an inactive slope): ====\n' +
      str(estimated_parameters_bayesian[1]))
print('==== Bayesian decay analysis: Estimated N values (linear scale): ====\n' +
      str(estimated_parameters_bayesian[2]))

# Get EDC from estimated parameters
fitted_edc_bayesian = decay_model(estimated_parameters_bayesian[0],
                                  estimated_parameters_bayesian[1],
                                  estimated_parameters_bayesian[2],
                                  time_axis=time_axis,
                                  compensate_uli=True,
                                  backend='np')
fitted_edc_bayesian = torch.from_numpy(np.sum(fitted_edc_bayesian, 1, keepdims=True))  # sum up terms of the decay model

# Discard last 5% for MSE evaluation
fitted_edc_bayesian = discard_last_n_percent(fitted_edc_bayesian, 5)

# Calculate MSE between true EDC and fitted EDC
mse_per_frequencyband = calc_mse(true_edc, fitted_edc_bayesian)

# Plot
time_axis_excl5 = time_axis[0:round(0.95 * len(time_axis))]  # discard last 5 percent of plot time axis
colors = ['b', 'g', 'r', 'c', 'm', 'y']
filter_frequencies = decayfitnet.get_filter_frequencies()
for band_idx in range(true_edc.shape[0]):
    plt.plot(time_axis_excl5, 10 * torch.log10(true_edc[band_idx, 0, :].squeeze()),
             colors[band_idx], label='Measured EDC, {} Hz'.format(filter_frequencies[band_idx]))
    plt.plot(time_axis_excl5, 10 * torch.log10(fitted_edc_bayesian[band_idx, 0, :].squeeze()),
             colors[band_idx] + '--', label='Bayesian fit, {} Hz'.format(filter_frequencies[band_idx]))

plt.xlabel('time [s]')
plt.ylabel('energy [dB]')
plt.subplots_adjust(right=0.6)
plt.legend(loc='upper right', bbox_to_anchor=(1.8, 1))
plt.title('Bayesian decay analysis')
plt.show()
