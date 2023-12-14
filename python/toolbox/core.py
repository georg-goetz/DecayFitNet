import torch
import torch.nn as nn
import torchaudio.functional
from torch.utils.data import Dataset
import scipy
import scipy.stats
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import h5py


class DecayDataset(Dataset):
    """Decay dataset."""

    def __init__(self, n_slopes_max=3, edcs_per_slope=100000, testset_flag=False, exactly_n_slopes_mode=False):
        """
        Args:
        """
        self.testset_flag = testset_flag

        datasets_dir = '../data/'

        if not testset_flag:
            self.exactly_n_slopes_mode = exactly_n_slopes_mode
            if exactly_n_slopes_mode:
                n_slopes_str = '_{}slopes'.format(n_slopes_max)
            else:
                n_slopes_str = ''

            # Load EDCs
            f_edcs = h5py.File(datasets_dir + 'synthEDCs/edcs_100{}.mat'.format(n_slopes_str), 'r')
            edcs = np.array(f_edcs.get('edcs'))

            # Load noise values
            f_noise_levels = h5py.File(datasets_dir + 'synthEDCs/noiseLevels_100{}.mat'.format(n_slopes_str), 'r')
            noise_levels = np.array(f_noise_levels.get('noiseLevels'))

            # Get EDCs into pytorch format
            edcs = torch.from_numpy(edcs).float()
            self.edcs = edcs

            # Convert EDCs into dB
            edcs_db = 10 * torch.log10(self.edcs)
            assert not torch.any(torch.isnan(edcs_db)), 'NaN values in db EDCs'

            # Normalize dB values to lie between -1 and 1 (input scaling)
            self.edcs_db_normfactor = torch.max(torch.abs(edcs_db))
            edcs_db_normalized = 2 * edcs_db / self.edcs_db_normfactor
            edcs_db_normalized += 1

            assert not torch.any(torch.isnan(edcs_db_normalized)), 'NaN values in normalized EDCs'
            assert not torch.any(torch.isinf(edcs_db_normalized)), 'Inf values in normalized EDCs'
            self.edcs_db_normalized = edcs_db_normalized

            # Noise level values are used in training for the noise loss
            noise_levels = torch.from_numpy(noise_levels).float()
            self.noise_levels = noise_levels

            assert self.edcs.shape[1] == self.noise_levels.shape[1], 'More EDCs than noise_levels'

            # Only if n_slopes should be predicted by network: Generate vector that specifies how many slopes are in
            # every EDC
            if not exactly_n_slopes_mode:
                self.n_slopes = torch.zeros((1, self.edcs.shape[1]))
                for slope_idx in range(1, n_slopes_max + 1):
                    self.n_slopes[0, (slope_idx - 1) * edcs_per_slope:slope_idx * edcs_per_slope] = slope_idx - 1
                self.n_slopes = self.n_slopes.long()
        else:
            f_edcs = h5py.File(datasets_dir + 'motus/edcs_100.mat', 'r')
            edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 100).T

            self.edcs = edcs

    def __len__(self):
        return self.edcs.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.testset_flag:
            edcs = self.edcs[:, idx]
            return edcs
        else:
            edcs = self.edcs[:, idx]
            edcs_db_normalized = self.edcs_db_normalized[:, idx]
            noise_levels = self.noise_levels[:, idx]

            if self.exactly_n_slopes_mode:
                n_slopes = torch.empty(0)  # just return empty, because n_slopes is already fixed
            else:
                n_slopes = self.n_slopes[:, idx]

            return edcs, noise_levels, edcs_db_normalized, n_slopes


class DecayFitNet(nn.Module):
    def __init__(self, n_slopes, n_max_units, n_filters, n_layers, relu_slope, dropout, reduction_per_layer, device,
                 exactly_n_slopes_mode=False):
        super(DecayFitNet, self).__init__()

        self.n_slopes = n_slopes
        self.device = device

        self.activation = nn.LeakyReLU(relu_slope)
        self.dropout = nn.Dropout(dropout)

        # Base Network
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=13, padding=6)
        self.maxpool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(n_filters, n_filters*2, kernel_size=7, padding=3)
        self.maxpool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(n_filters*2, n_filters*2, kernel_size=7, padding=3)
        self.maxpool3 = nn.MaxPool1d(2)
        self.input = nn.Linear(2*n_filters*2, n_max_units)

        self.linears = nn.ModuleList([nn.Linear(round(n_max_units * (reduction_per_layer**i)),
                                                round(n_max_units * (reduction_per_layer**(i+1)))) for i in range(n_layers-1)])

        # T_vals
        self.final1_t = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_t = nn.Linear(50, n_slopes)

        # A_vals
        self.final1_a = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
        self.final2_a = nn.Linear(50, n_slopes)

        # Noise
        self.final1_n = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
        self.final2_n = nn.Linear(50, 1)

        # N Slopes
        self.exactly_n_slopes_mode = exactly_n_slopes_mode
        if not exactly_n_slopes_mode:
            self.final1_n_slopes = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers-1))), 50)
            self.final2_n_slopes = nn.Linear(50, n_slopes)

    def forward(self, edcs):
        """
        Args:

        Returns:
        """

        # Base network
        x = self.maxpool1(self.activation(self.conv1(edcs.unsqueeze(1))))
        x = self.maxpool2(self.activation(self.conv2(x)))
        x = self.maxpool3(self.activation(self.conv3(x)))
        x = self.activation(self.input(self.dropout(x.view(edcs.shape[0], -1))))
        for layer in self.linears:
            x = layer(x)
            x = self.activation(x)

        # T_vals
        t = self.activation(self.final1_t(x))
        t = torch.pow(self.final2_t(t), 2.0) + 0.01

        # A_vals
        a = self.activation(self.final1_a(x))
        a = torch.pow(self.final2_a(a), 2.0) + 1e-16

        # Noise
        n_exponent = self.activation(self.final1_n(x))
        n_exponent = self.final2_n(n_exponent)

        # N Slopes (if they should be predicted, otherwise, just leave this out)
        if not self.exactly_n_slopes_mode:
            n_slopes = self.activation(self.final1_n_slopes(x))
            n_slopes = self.final2_n_slopes(n_slopes)
        else:
            n_slopes = torch.empty(0)  # just return None, because n_slopes is already fixed

        return t, a, n_exponent, n_slopes


def edc_loss(t_vals_prediction, a_vals_prediction, n_exp_prediction, edcs_true, device, training_flag=True,
             plot_fit=False, apply_mean=True):
    fs = 10
    l_edc = 10

    # Generate the t values that would be discarded (last 5%) as well, otherwise the models do not match.
    t = (torch.linspace(0, l_edc * fs - 1, round((1 / 0.95) * l_edc * fs)) / fs).to(device)

    # Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
    n_exp_prediction = torch.clamp(n_exp_prediction, -32, 32)
    n_vals_prediction = torch.pow(10, n_exp_prediction)

    if training_flag:
        # use L1Loss in training
        loss_fn = nn.L1Loss(reduction='none')
    else:
        loss_fn = nn.MSELoss(reduction='none')

    # Use predicted values to generate an EDC
    edc_prediction = generate_synthetic_edc_torch(t_vals_prediction, a_vals_prediction, n_vals_prediction, t, device)

    # discard last 5 percent (i.e. the step which is already done for the true EDC and the test datasets prior to
    # saving them to the .mat files that are loaded in the beginning of this script
    edc_prediction = edc_prediction[:, 0:l_edc * fs]

    if plot_fit:
        for idx in range(0, edcs_true.shape[0]):
            plt.plot(10 * torch.log10(edcs_true[idx, :]))
            plt.plot(10 * torch.log10(edc_prediction[idx, :].detach()))
            plt.show()

    # Go to dB scale
    edc_true_db = 10 * torch.log10(edcs_true + 1e-16)
    edc_prediction_db = 10 * torch.log10(edc_prediction + 1e-16)

    # Calculate loss on dB scale
    if apply_mean:
        loss = torch.mean(loss_fn(edc_true_db, edc_prediction_db))
    else:
        loss = loss_fn(edc_true_db, edc_prediction_db)

    return loss


class FilterByOctaves(nn.Module):
    """Generates an octave wide filterbank and filters tensors.

    This is gpu compatible if using torch backend, but it is super slow and should not be used at all.
    The octave filterbanks is created using cascade Buttwerworth filters, which then are processed using
    the biquad function native to PyTorch.

    This is useful to get the decay curves of RIRs.
    """

    def __init__(self, center_frequencies=None, order=5, sample_rate=48000, backend='scipy'):
        super(FilterByOctaves, self).__init__()

        if center_frequencies is None:
            center_frequencies = [125, 250, 500, 1000, 2000, 4000]
        self._center_frequencies = center_frequencies
        self._order = order
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(center_frequencies, self._sample_rate, self._order)
        self.backend = backend

    def _forward_scipy(self, x):
        out = []
        for this_sos in self._sos:
            tmp = torch.clone(x).cpu().numpy()
            tmp = scipy.signal.sosfilt(this_sos, tmp, axis=-1)
            out.append(torch.from_numpy(tmp.copy()))
        out = torch.stack(out, dim=-2)  # Stack over frequency bands

        return out

    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_order(self, order):
        self._order = order
        self._sos = self._get_octave_filters(self._center_frequencies, self._sample_rate, self._order)

    def set_center_frequencies(self, center_freqs):
        center_freqs_np = np.asarray(center_freqs)
        assert not np.any(center_freqs_np < 0) and not np.any(center_freqs_np > self._sample_rate / 2), \
            'Center Frequencies must be greater than 0 and smaller than fs/2. Exceptions: exactly 0 or fs/2 ' \
            'will give lowpass or highpass bands'
        self._center_frequencies = np.sort(center_freqs_np).tolist()
        self._sos = self._get_octave_filters(center_freqs, self._sample_rate, self._order)

    def get_center_frequencies(self):
        return self._center_frequencies

    def forward(self, x):
        if self.backend == 'scipy':
            out = self._forward_scipy(x)
        else:
            raise NotImplementedError('No good implementation relying solely on the pytorch backend has been found yet')
        return out

    def get_filterbank_impulse_response(self):
        """Returns the impulse response of the filterbank."""
        impulse = torch.zeros(1, self._sample_rate * 20)
        impulse[0, self._sample_rate] = 1
        response = self.forward(impulse)
        return response

    @staticmethod
    def _get_octave_filters(center_freqs: List, fs: int, order: int = 5) -> List[torch.Tensor]:
        """
        Design octave band filters (butterworth filter).
        Returns a tensor with the SOS (second order sections) representation of the filter
        """
        sos = []
        for band_idx in range(len(center_freqs)):
            center_freq = center_freqs[band_idx]
            if abs(center_freq) < 1e-6:
                # Lowpass band below lowest octave band
                f_cutoff = (1 / np.sqrt(2)) * center_freqs[band_idx + 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='lowpass', analog=False, output='sos')
            elif abs(center_freq - fs / 2) < 1e-6:
                f_cutoff = np.sqrt(2) * center_freqs[band_idx - 1]
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='highpass', analog=False,
                                               output='sos')
            else:
                f_cutoff = center_freq * np.array([1 / np.sqrt(2), np.sqrt(2)])
                this_sos = scipy.signal.butter(N=order, Wn=f_cutoff, fs=fs, btype='bandpass', analog=False,
                                               output='sos')

            sos.append(torch.from_numpy(this_sos))

        return sos


class Normalizer(torch.nn.Module):
    """ Normalizes the data to have zero mean and unit variance for each feature."""

    def __init__(self, means, stds):
        super(Normalizer, self).__init__()
        self.means = means
        self.stds = stds
        self.eps = np.finfo(np.float32).eps

    def forward(self, x):
        out = x - self.means
        out = out / (self.stds + self.eps)

        return out


def discard_last_n_percent(edc: torch.Tensor, n_percent: float) -> torch.Tensor:
    # Discard last n%
    last_id = int(np.round((1 - n_percent / 100) * edc.shape[-1]))
    out = edc[..., 0:last_id]

    return out


def _discard_below(edc: torch.Tensor, threshold_val: float) -> torch.Tensor:
    # set all values below minimum to 0
    out = edc.detach().clone()
    out[out < threshold_val] = 0

    out = _discard_trailing_zeros(out)
    return out


def _discard_trailing_zeros(rir: torch.Tensor) -> torch.Tensor:
    # find first non-zero element from back
    last_above_thres = rir.shape[-1] - torch.argmax((rir.flip(-1) != 0).squeeze().int())

    # discard from that sample onwards
    out = rir[..., :last_above_thres]
    return out


def check_format(rir):
    rir = torch.as_tensor(rir).detach().clone()

    if len(rir.shape) == 1:
        rir = rir.reshape(1, -1)

    if rir.shape[0] > rir.shape[1]:
        rir = torch.swapaxes(rir, 0, 1)
        print(f'Swapped axes to bring rir into the format [{rir.shape[0]} x {rir.shape[1]}]. This should coincide '
              f'with [n_channels x rir_length], which is the expected input format to the function you called.')
    return rir


def rir_onset(rir):
    spectrogram_trans = torchaudio.transforms.Spectrogram(n_fft=64, win_length=64, hop_length=4)
    spectrogram = spectrogram_trans(rir)
    windowed_energy = torch.sum(spectrogram, dim=len(spectrogram.shape)-2)
    delta_energy = windowed_energy[..., 1:] / (windowed_energy[..., 0:-1]+1e-16)
    highest_energy_change_window_idx = torch.argmax(delta_energy)
    onset = int((highest_energy_change_window_idx-2) * 4 + 64)
    return onset


class PreprocessRIR(nn.Module):
    """ Preprocess a RIR to extract the EDC and prepare it for the neural network model.
        The preprocessing includes:

        # RIR -> Filterbank -> octave-band filtered RIR
        # octave-band filtered RIR -> backwards integration -> EDC
        # EDC -> delete last 5% of samples -> EDC_crop
        # EDC_crop -> downsample to the smallest number above 2400, i.e. by factor floor(original_length / 2400)
            -> EDC_ds1
        # EDC_ds1 -> as it might still be a little more than 2400 samples, just cut away everything after 2400 samples
            -> EDC_ds2
        # EDC_ds2 -> dB scale-> EDC_db
        # EDC_db -> normalization -> EDC_final that is the input to the network
    """

    def __init__(self, input_transform: Dict = None, sample_rate: int = 48000, output_size: int = None,
                 filter_frequencies: List = None):
        super(PreprocessRIR, self).__init__()

        self.input_transform = input_transform
        self.output_size = output_size
        self.sample_rate = sample_rate
        self.eps = 1e-10

        self.filterbank = FilterByOctaves(order=5, sample_rate=self.sample_rate, backend='scipy',
                                          center_frequencies=filter_frequencies)

    def set_filter_frequencies(self, filter_frequencies):
        self.filterbank.set_center_frequencies(filter_frequencies)

    def get_filter_frequencies(self):
        return self.filterbank.get_center_frequencies()

    def forward(self, input_rir: torch.Tensor, input_is_edc: bool = False, analyse_full_rir=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        input_rir = check_format(input_rir)

        if input_is_edc:
            norm_vals = torch.max(input_rir, dim=-1, keepdim=True).values  # per channel
            schroeder_decays = input_rir / norm_vals
            if len(input_rir.shape) == 2:
                schroeder_decays = schroeder_decays.unsqueeze(1)
        else:
            # Extract decays from RIR: Do backwards integration
            schroeder_decays, norm_vals = self.schroeder(input_rir, analyse_full_rir=analyse_full_rir)

        # Convert to dB
        schroeder_decays_db = 10 * torch.log10(schroeder_decays + self.eps)

        # N values have to be adjusted for downsampling
        n_adjust = schroeder_decays_db.shape[-1] / self.output_size

        # DecayFitNet: T value predictions have to be adjusted for the time-scale conversion
        if self.input_transform is not None:
            t_adjust = 10 / (schroeder_decays_db.shape[-1] / self.sample_rate)
        else:
            t_adjust = 1

        # DecayFitNet: Discard last 5%
        if self.input_transform is not None:
            schroeder_decays_db = discard_last_n_percent(schroeder_decays_db, 5)

        # Resample to self.output_size samples (if given, otherwise keep sampling rate)
        if self.output_size is not None:
            schroeder_decays_db = torch.nn.functional.interpolate(schroeder_decays_db, size=self.output_size,
                                                                  mode='linear', align_corners=True)

        # DecayFitNet: Normalize with input transform
        if self.input_transform is not None:
            schroeder_decays_db = 2 * schroeder_decays_db / self.input_transform["edcs_db_normfactor"]
            schroeder_decays_db = schroeder_decays_db + 1

        # Write adjust factors into one dict
        scale_adjust_factors = {"t_adjust": t_adjust, "n_adjust": n_adjust}

        # Calculate time axis: be careful, because schroeder_decays_db might be on a different time scale!
        time_axis = torch.linspace(0, (schroeder_decays.shape[2] - 1) / self.sample_rate, schroeder_decays_db.shape[2])

        # Reshape freq bands as batch size, shape = [batch * freqs, timesteps]
        schroeder_decays_db = schroeder_decays_db.view(-1, schroeder_decays_db.shape[-1]).type(torch.float32)

        return schroeder_decays_db, time_axis, norm_vals, scale_adjust_factors

    def schroeder(self, rir: torch.Tensor, analyse_full_rir=True) -> Tuple[torch.Tensor, torch.Tensor]:
        # Check that RIR is in correct format/shape and return it in correct format if it wasn't before
        rir = check_format(rir)

        if not analyse_full_rir:
            onset = rir_onset(rir)
            rir = rir[..., onset:]

        out = _discard_trailing_zeros(rir)

        # Filter
        out = self.filterbank(out)

        # Remove filtering artefacts (last 5 permille)
        out = discard_last_n_percent(out, 0.5)

        # Backwards integral
        out = torch.flip(out, [2])
        out = torch.cumsum(out ** 2, 2)
        out = torch.flip(out, [2])

        # Normalize to 1
        norm_vals = torch.max(out, dim=-1, keepdim=True).values  # per channel
        out = out / norm_vals

        return out, norm_vals.squeeze(2)


def _postprocess_parameters(t_vals, a_vals, n_vals, scale_adjust_factors, exactly_n_slopes_mode):
    # Process the estimated t, a, and n parameters

    # Adjust for downsampling
    n_vals = n_vals / scale_adjust_factors['n_adjust']

    # Only for DecayFitNet: T value predictions have to be adjusted for the time-scale conversion (downsampling)
    t_vals = t_vals / scale_adjust_factors['t_adjust']  # factors are 1 for Bayesian

    # In nSlope estimation mode: get a binary mask to only use the number of slopes that were predicted, zero others
    if not exactly_n_slopes_mode:
        mask = (a_vals == 0)

        # Assign NaN instead of zero for now, to sort inactive slopes to the end
        t_vals[mask] = np.nan
        a_vals[mask] = np.nan

    # Sort T and A values
    sort_idxs = np.argsort(t_vals, 1)
    for band_idx in range(t_vals.shape[0]):
        t_vals[band_idx, :] = t_vals[band_idx, sort_idxs[band_idx, :]]
        a_vals[band_idx, :] = a_vals[band_idx, sort_idxs[band_idx, :]]

    # In nSlope estimation mode: set nans to zero again
    if not exactly_n_slopes_mode:
        t_vals[np.isnan(t_vals)] = 0
        a_vals[np.isnan(a_vals)] = 0

    return t_vals, a_vals, n_vals


def decay_model(t_vals, a_vals, n_val, time_axis, compensate_uli=True, backend='np', device='cpu'):
    # t_vals, a_vals, n_vals can be either given as [n_vals, ] or as [n_batch or n_bands, n_vals]

    # Avoid div by zero for T=0: Write arbitary number (1) into T values that are equal to zero (inactive slope),
    # because their amplitude will be 0 as well (i.e. they don't contribute to the EDC)
    zero_t = (t_vals == 0)
    also_zero_a = (a_vals[zero_t] == 0)
    if backend == 'torch':
        also_zero_a = also_zero_a.numpy()
    assert (np.all(also_zero_a)), "T values equal zero detected, for which A values are nonzero. This " \
                                  "yields division by zero. For inactive slopes, set A to zero."
    t_vals[t_vals == 0] = 1

    if backend == 'np':
        edc_model = generate_synthetic_edc_np(t_vals, a_vals, n_val, time_axis, compensate_uli=compensate_uli)
        return edc_model
    elif backend == 'torch':
        edc_model = generate_synthetic_edc_torch(t_vals, a_vals, n_val, time_axis, device=device,
                                                 compensate_uli=compensate_uli)

        # Output should have the shape [n_bands, n_batches, n_samples]
        edc_model = torch.unsqueeze(edc_model, 1)
        return edc_model
    else:
        raise ValueError("Backend must be either 'np' or 'torch'.")


def generate_synthetic_edc_torch(t_vals, a_vals, noise_level, time_axis, device='cpu', compensate_uli=True) -> torch.Tensor:
    """ Generates an EDC from the estimated parameters."""
    # Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
    tau_vals = torch.log(torch.Tensor([1e6])).to(device) / t_vals

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    t_rep = time_axis.repeat(t_vals.shape[0], t_vals.shape[1], 1)
    tau_vals_rep = tau_vals.unsqueeze(2).repeat(1, 1, time_axis.shape[0])

    # Calculate exponentials from decay rates
    time_vals = -t_rep * tau_vals_rep
    exponentials = torch.exp(time_vals)

    # account for limited upper limit of integration, see: Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M.
    # "Evaluation of decay times in coupled spaces: Reliability analysis of Bayeisan decay time estimation."
    # J Acoust Soc Am 117, 3707–3715 (2005).
    if compensate_uli:
        exp_offset = exponentials[:, :, -1].unsqueeze(2).repeat(1, 1, time_axis.shape[0])
    else:
        exp_offset = 0

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    A_rep = a_vals.unsqueeze(2).repeat(1, 1, time_axis.shape[0])

    # Multiply exponentials with their amplitudes and sum all exponentials together
    edcs = A_rep * (exponentials - exp_offset)
    edc = torch.sum(edcs, 1)

    # Add noise
    noise = noise_level * torch.linspace(len(time_axis), 1, len(time_axis)).to(device)
    edc = edc + noise
    return edc


def generate_synthetic_edc_np(t_vals, a_vals, noise_level, time_axis, compensate_uli=True) -> np.ndarray:
    value_dim = len(t_vals.shape) - 1

    # get decay rate: decay energy should have decreased by 60 db after T seconds
    zero_a = (a_vals == 0)
    tau_vals = np.log(1e6) / t_vals
    tau_vals[zero_a] = 0

    # calculate decaying exponential terms
    time_vals = - np.tile(time_axis, (*t_vals.shape, 1)) * np.expand_dims(tau_vals, -1)
    exponentials = np.exp(time_vals)

    # account for limited upper limit of integration, see: Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M.
    # "Evaluation of decay times in coupled spaces: Reliability analysis of Bayeisan decay time estimation."
    # J Acoust Soc Am 117, 3707–3715 (2005).
    if compensate_uli:
        exp_offset = np.expand_dims(exponentials[..., -1], -1)
    else:
        exp_offset = 0

    # calculate final exponential terms
    exponentials = (exponentials - exp_offset) * np.expand_dims(a_vals, -1)

    # zero exponentials where T=A=0 (they are NaN now because div by 0, and NaN*0=NaN in python)
    exponentials[zero_a, :] = 0

    # calculate noise term
    noise = noise_level * np.linspace(len(time_axis), 1, len(time_axis))
    noise = np.expand_dims(noise, value_dim)

    edc_model = np.concatenate((exponentials, noise), value_dim)
    return edc_model
