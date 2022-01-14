import torch
import torch.nn as nn
from torch.utils.data import Dataset
import scipy
import scipy.stats
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, TypeVar, Callable, Any, List, Dict
import h5py

T = TypeVar('T', bound=Callable[..., Any])


# https://realpython.com/documenting-python-code/

class DecayDataset(Dataset):
    """Decay dataset."""

    def __init__(self, n_slopes_min=1, n_slopes_max=5, edcs_per_slope=10000, triton_flag=False, testset_flag=False,
                 testset_name='summer830'):
        """
        Args:
        """
        self.testset_flag = testset_flag

        if triton_flag:
            datasets_dir = '/scratch/elec/t40527-hybridacoustics/datasets/decayfitting/'
        else:
            datasets_dir = '/Volumes/ARTSRAM/GeneralDecayEstimation/decayFitting/'

        if not testset_flag:
            # Load EDCs
            f_edcs = h5py.File(datasets_dir + 'edcs_slim.mat', 'r')
            edcs = np.array(f_edcs.get('edcs'))

            # Load noise values
            f_noise_levels = h5py.File(datasets_dir + 'noiseLevels_slim.mat', 'r')
            noise_levels = np.array(f_noise_levels.get('noiseLevels'))

            # Get EDCs into pytorch format
            edcs = torch.from_numpy(edcs).float()
            self.edcs = edcs[:, (n_slopes_min - 1) * edcs_per_slope:n_slopes_max * edcs_per_slope]

            # Put EDCs into dB
            edcs_db = 10 * torch.log10(self.edcs)
            assert not torch.any(torch.isnan(edcs_db)), 'NaN values in db EDCs'

            # Normalize dB values to lie between -1 and 1 (input scaling)
            self.edcs_db_normfactor = torch.max(torch.abs(edcs_db))
            edcs_db_normalized = 2 * edcs_db / self.edcs_db_normfactor
            edcs_db_normalized += 1

            assert not torch.any(torch.isnan(edcs_db_normalized)), 'NaN values in normalized EDCs'
            assert not torch.any(torch.isinf(edcs_db_normalized)), 'Inf values in normalized EDCs'
            self.edcs_db_normalized = edcs_db_normalized

            # Generate vector that specifies how many slopes are in every EDC
            self.n_slopes = torch.zeros((1, self.edcs.shape[1]))
            for slope_idx in range(n_slopes_min, n_slopes_max + 1):
                self.n_slopes[0, (slope_idx - 1) * edcs_per_slope:slope_idx * edcs_per_slope] = slope_idx - 1
            self.n_slopes = self.n_slopes.long()

            # Noise level values are used in training for the noise loss
            noise_levels = torch.from_numpy(noise_levels).float()
            self.noise_levels = noise_levels[:, (n_slopes_min - 1) * edcs_per_slope:n_slopes_max * edcs_per_slope]

            assert self.edcs.shape[1] == self.noise_levels.shape[1], 'More EDCs than noise_levels'
        else:
            if testset_name == 'summer830':
                f_edcs = h5py.File(datasets_dir + 'summer830/edcs_slim.mat', 'r')
                edcs = torch.from_numpy(np.array(f_edcs.get('summer830edcs/edcs'))).float().view(-1, 100).T
            elif testset_name == 'roomtransition':
                f_edcs = h5py.File(datasets_dir + 'roomtransition/edcs_slim.mat', 'r')
                edcs = torch.from_numpy(np.array(f_edcs.get('roomTransitionEdcs/edcs'))).float().view(-1, 100).T
            else:
                raise NotImplementedError('Unknown testset.')

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
            n_slopes = self.n_slopes[:, idx]

            return edcs, noise_levels, edcs_db_normalized, n_slopes


class DecayFitNetLinear(nn.Module):
    def __init__(self, n_slopes, n_max_units, n_filters, n_layers, relu_slope, dropout, reduction_per_layer, device):
        super(DecayFitNetLinear, self).__init__()

        self.n_slopes = n_slopes
        self.device = device

        self.activation = nn.LeakyReLU(relu_slope)
        self.dropout = nn.Dropout(dropout)

        # Base Network
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=13, padding=6)
        self.maxpool1 = nn.MaxPool1d(5)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=7, padding=3)
        self.maxpool2 = nn.MaxPool1d(5)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 2, kernel_size=7, padding=3)
        self.maxpool3 = nn.MaxPool1d(2)
        self.input = nn.Linear(2 * n_filters * 2, n_max_units)

        self.linears = nn.ModuleList([nn.Linear(round(n_max_units * (reduction_per_layer ** i)),
                                                round(n_max_units * (reduction_per_layer ** (i + 1)))) for i in
                                      range(n_layers - 1)])

        # T_vals
        self.final1_t = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_t = nn.Linear(50, n_slopes)

        # A_vals
        self.final1_a = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_a = nn.Linear(50, n_slopes)

        # Noise
        self.final1_n = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_n = nn.Linear(50, 1)

        # N Slopes
        self.final1_n_slopes = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
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

        # N Slopes
        n_slopes = self.activation(self.final1_n_slopes(x))
        n_slopes = self.final2_n_slopes(n_slopes)

        return t, a, n_exponent, n_slopes


class DecayFitNetLinearExactlyNSlopes(nn.Module):
    def __init__(self, n_slopes, n_max_units, n_filters, n_layers, relu_slope, dropout, reduction_per_layer, device):
        super(DecayFitNetLinearExactlyNSlopes, self).__init__()

        self.n_slopes = n_slopes
        self.device = device

        self.activation = nn.LeakyReLU(relu_slope)
        self.dropout = nn.Dropout(dropout)

        # Base Network
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size=13, padding=6)
        self.maxpool1 = nn.MaxPool1d(10)
        self.conv2 = nn.Conv1d(n_filters, n_filters * 2, kernel_size=7, padding=3)
        self.maxpool2 = nn.MaxPool1d(8)
        self.conv3 = nn.Conv1d(n_filters * 2, n_filters * 4, kernel_size=7, padding=3)
        self.maxpool3 = nn.MaxPool1d(6)
        self.input = nn.Linear(5 * n_filters * 4, n_max_units)

        self.linears = nn.ModuleList([nn.Linear(round(n_max_units * (reduction_per_layer ** i)),
                                                round(n_max_units * (reduction_per_layer ** (i + 1)))) for i in
                                      range(n_layers - 1)])

        # T_vals
        self.final1_t = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_t = nn.Linear(50, n_slopes)

        # A_vals
        self.final1_a = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_a = nn.Linear(50, n_slopes)

        # Noise
        self.final1_n = nn.Linear(round(n_max_units * (reduction_per_layer ** (n_layers - 1))), 50)
        self.final2_n = nn.Linear(50, 1)

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

        return t, a, n_exponent


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
            tmp = scipy.signal.sosfiltfilt(this_sos, tmp, axis=-1)
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


def discard_lastNPercent(edc: torch.Tensor, nPercent: float) -> torch.Tensor:
    # Discard last n%
    last_id = int(np.round((1 - nPercent / 100) * edc.shape[-1]))
    out = edc[..., 0:last_id]

    return out


def discard_below(edc: torch.Tensor, threshold_val: float) -> torch.Tensor:
    # set all values below minimum to 0
    out = edc.detach().clone()
    out[out < threshold_val] = 0

    out = discard_trailing_zeros(out)
    return out


def discard_trailing_zeros(rir: torch.Tensor) -> torch.Tensor:
    out = rir.detach().clone()

    # find first non-zero element from back
    last_above_thres = out.shape[-1] - torch.argmax((out.flip(-1) != 0).squeeze().int())

    # discard from that sample onwards
    out = out[..., :last_above_thres]
    return out


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

    def __init__(self, input_transform: Dict = None, sample_rate: int = 48000, output_size: int = 100,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        # Extract decays: Do backwards integration
        schroeder_decays, norm_vals = self.schroeder(x)

        # Convert to dB
        schroeder_decays_db = 10 * torch.log10(schroeder_decays + self.eps)

        # N values have to be adjusted for downsampling
        n_adjust = schroeder_decays_db.shape[2] / self.output_size

        # DecayFitNet: T value predictions have to be adjusted for the time-scale conversion
        if self.input_transform is not None:
            t_adjust = 10 / (schroeder_decays_db.shape[2] / self.sample_rate)
        else:
            t_adjust = 1

        # DecayFitNet: Discard last 5%
        if self.input_transform is not None:
            schroeder_decays_db = discard_lastNPercent(schroeder_decays_db, 5)

        # Resample to self.output_size samples
        schroeder_decays_db_ds = torch.nn.functional.interpolate(schroeder_decays_db, size=self.output_size,
                                                                 mode='linear', align_corners=True)

        # DecayFitNet: Normalize with input transform
        if self.input_transform is not None:
            schroeder_decays_db_ds = 2 * schroeder_decays_db_ds / self.input_transform["edcs_db_normfactor"]
            schroeder_decays_db_ds = schroeder_decays_db_ds + 1

        # Write adjust factors into one dict
        scale_adjust_factors = {"t_adjust": t_adjust, "n_adjust": n_adjust}

        time_axis_ds = torch.linspace(0, (schroeder_decays.shape[2] - 1) / self.sample_rate, self.output_size)

        # Reshape freq bands as batch size, shape = [batch * freqs, timesteps]
        schroeder_decays_db_ds = schroeder_decays_db_ds.view(-1, schroeder_decays_db_ds.shape[-1]).type(torch.float32)

        return schroeder_decays_db_ds, time_axis_ds, norm_vals, scale_adjust_factors

    def schroeder(self, rir: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = discard_trailing_zeros(rir)

        # Filter
        out = self.filterbank(out)

        # Backwards integral
        out = torch.flip(out, [2])
        out = (1 / out.shape[2]) * torch.cumsum(out ** 2, 2)
        out = torch.flip(out, [2])

        # Normalize to 1
        norm_factors = torch.max(out, dim=-1, keepdim=True).values  # per channel
        out = out / norm_factors

        return out, norm_factors.squeeze(2)


def generate_synthetic_edc(T, A, noiseLevel, t, device) -> torch.Tensor:
    """ Generates an EDC from the estimated parameters."""
    # Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
    tau_vals = -torch.log(torch.Tensor([1e-6])).to(device) / T

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    t_rep = t.repeat(T.shape[0], T.shape[1], 1)
    tau_vals_rep = tau_vals.unsqueeze(2).repeat(1, 1, t.shape[0])

    # Calculate exponentials from decay rates
    time_vals = -t_rep * tau_vals_rep
    exponentials = torch.exp(time_vals)

    # Offset is required to make last value of EDC be correct
    exp_offset = exponentials[:, :, -1].unsqueeze(2).repeat(1, 1, t.shape[0])

    # Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    A_rep = A.unsqueeze(2).repeat(1, 1, t.shape[0])

    # Multiply exponentials with their amplitudes and sum all exponentials together
    edcs = A_rep * (exponentials - exp_offset)
    edc = torch.sum(edcs, 1)

    # Add noise
    noise = noiseLevel * torch.linspace(len(t), 1, len(t)).to(device)
    edc = edc + noise
    return edc


def postprocess_parameters(t_prediction, a_prediction, n_prediction, n_slopes_probabilities, device, sort_values=True):
    # Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
    n_prediction = torch.clamp(n_prediction, -32, 32)

    # Go from noise exponent to noise value
    n_prediction = torch.pow(10, n_prediction)

    # Get a binary mask to only use the number of slopes that were predicted, zero others
    _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
    n_slopes_prediction += 1  # because python starts at 0
    temp = torch.linspace(1, 3, 3).repeat(n_slopes_prediction.shape[0], 1).to(device)
    mask = temp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, 3))
    a_prediction[~mask] = 0

    if sort_values:
        # Sort T and A values:
        # 1) assign nans to sort the inactive slopes to the end
        t_prediction[~mask] = float('nan')  # nan as a placeholder, gets replaced in a few lines
        a_prediction[~mask] = float('nan')  # nan as a placeholder, gets replaced in a few lines

        # 2) sort and set nans to zero again
        t_prediction, sort_idxs = torch.sort(t_prediction)
        for batch_idx, a_this_batch in enumerate(a_prediction):
            a_prediction[batch_idx, :] = a_this_batch[sort_idxs[batch_idx]]
        t_prediction[torch.isnan(t_prediction)] = 0  # replace nan from above
        a_prediction[torch.isnan(a_prediction)] = 0  # replace nan from above

    return t_prediction, a_prediction, n_prediction, n_slopes_prediction


def adjust_timescale(t_prediction, n_prediction, scale_adjust_factors):
    # T value predictions have to be adjusted for the time-scale conversion (downsampling)
    t_prediction = t_prediction / scale_adjust_factors['t_adjust']

    # N value predictions have to be converted from exponent representation to actual value and adjusted for
    # the downsampling
    n_prediction = n_prediction / scale_adjust_factors['n_adjust']

    return t_prediction, n_prediction


def edc_loss(t_vals_prediction, a_vals_prediction, n_exp_prediction, edcs_true, device, training_flag=True,
             plot_fit=False, apply_mean=True):
    fs = 10
    l_edc = 10

    # Generate the t values that would be discarded as well, otherwise the models do not match.
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
    edc_prediction = generate_synthetic_edc(t_vals_prediction, a_vals_prediction, n_vals_prediction, t, device)

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


def decay_model(t_vals, a_vals, n_val, time_axis, compensate_uli=False):
    # get decay rate: decay energy should have decreased by 60 db after T seconds
    zero_t = (t_vals == 0)
    assert (np.all(a_vals[zero_t] == 0)), "T values equal zero detected, for which A values are nonzero. This yields " \
                                          "division by zero. For inactive slopes, set A to zero."
    tau_vals = np.log(1e6) / t_vals

    # calculate decaying exponential terms
    time_vals = - np.outer(time_axis, tau_vals)
    exponentials = np.exp(time_vals)

    # account for limited upper limit of integration, see: Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M. "Evaluation
    # of decay times in coupled spaces: Reliability analysis of Bayeisan decay time estimation." J Acoust Soc Am 117,
    # 3707–3715 (2005).
    if compensate_uli:
        exp_offset = exponentials[-1, :]
    else:
        exp_offset = 0

    # calculate final exponential terms
    exponentials = (exponentials - exp_offset) * a_vals

    # zero exponentials where T=A=0 (they are NaN now because div by 0)
    exponentials[:, zero_t] = 0

    # calculate noise term
    noise = n_val * np.linspace(len(time_axis), 1, len(time_axis))
    noise = np.expand_dims(noise, 1)

    edc_model = np.concatenate((exponentials, noise), 1)
    return edc_model
