# This demo shows how to convert parameters of the Schroeder model to parameters of a decaying noise model

import numpy as np
import matplotlib.pyplot as plt

from toolbox.core import decay_kernel, schroeder_to_envelope

np.random.seed(0)  # For reproducibility

schroeder_T = np.array([0.5, 1.5])  # seconds
schroeder_A = np.array([1, 0.01])  # decay model amplitude

fs = 48000  # in Hz
L = 3 * fs  # in samples

# Set up decay model
time_axis = np.linspace(0, (L - 1) / fs, L)
edf_model = decay_kernel(schroeder_T, time_axis)
edf_model = np.delete(edf_model, -1, axis=1)  # throw away noise term
edf_model = np.dot(edf_model, schroeder_A)

# Determine envelope from EDF model
envelopes_T, envelopes_A = schroeder_to_envelope(schroeder_T, schroeder_A, fs)

envelopes = decay_kernel(envelopes_T, time_axis)
envelopes = envelopes[:, :-1] * envelopes_A

# Set up decaying Gaussian noise
gaussian_noise = np.random.randn(L, len(schroeder_T))
decaying_gaussian_noise = np.sum(gaussian_noise * envelopes, axis=1)

# Calculate EDF from Gaussian noise
decaying_gaussian_noise_EDF = np.flipud(np.cumsum(np.flipud(decaying_gaussian_noise ** 2)))

# Plot
plt.figure()
plt.plot(time_axis, 10 * np.log10(edf_model))
plt.plot(time_axis, 10 * np.log10(decaying_gaussian_noise_EDF))
plt.legend(['EDF Model', 'Decaying Gaussian noise EDF'])
plt.show()

