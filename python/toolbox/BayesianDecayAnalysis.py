import numpy as np
import copy
import scipy.signal
import scipy.special
from typing import List, Tuple

from .core import PreprocessRIR, _postprocess_parameters, decay_model


def evaluate_likelihood(true_edc_db, t_vals, a_vals, n_val, time_axis):
    # Formulas following Xiang, N., Goggans, P., Jasa, T. & Robinson, P. "Bayesian characterization of multiple-
    # slope sound energy decays in coupled-volume systems." J Acoust Soc Am 129, 741–752 (2011)

    # Calculate model EDC
    model_edc = decay_model(t_vals, a_vals, n_val, time_axis, compensate_uli=True, backend='np')
    model_edc = np.sum(model_edc, 0)

    # Convert to dB (true EDC is already in db)
    model_edc = 10 * np.log10(model_edc)

    # exclude last 5% from likelihood (makes analysis more robust to edge effects caused by octave filtering)
    k = round(0.95 * len(time_axis))

    # Evaluate likelihood
    e = 0.5 * np.sum((true_edc_db[:k] - model_edc[:k]) ** 2)
    likelihood = 0.5 * scipy.special.gamma(k / 2) * (2 * np.pi * e) ** (-k / 2)

    return likelihood


class BayesianDecayAnalysis:
    def __init__(self, n_slopes: int = 0, sample_rate: int = 48000, parameter_ranges: dict = None,
                 n_iterations: int = 50, filter_frequencies: List = None):
        self._version = '0.1.2'
        self._n_slopes = n_slopes
        self._n_points_per_dim = 100
        self.n_iterations = n_iterations

        self._sample_rate = sample_rate

        if parameter_ranges is None:
            parameter_ranges = {'t_range': [0.1, 3.5],
                                'a_range': [-3, 0],
                                'n_range': [-10, -2]}
        self._parameter_ranges = parameter_ranges

        # Init parameter ranges
        t_range, a_range, n_range = self.get_parameter_ranges()
        self._t_range = t_range
        self._a_range = a_range
        self._n_range = n_range

        # Init parameter space
        t_space, a_space, n_space = self.get_parameter_space()
        self._t_space = t_space
        self._a_space = a_space
        self._n_space = n_space

        self._preprocess = PreprocessRIR(input_transform=None,
                                         sample_rate=self._sample_rate,
                                         output_size=100,
                                         filter_frequencies=filter_frequencies)

    def set_filter_frequencies(self, filter_frequencies):
        self._preprocess.set_filter_frequencies(filter_frequencies)

    def get_filter_frequencies(self):
        return self._preprocess.get_filter_frequencies()

    def set_output_size(self, output_size):
        self._preprocess.output_size = output_size

    def get_output_size(self):
        return self._preprocess.output_size

    def set_n_slopes(self, n_slopes):
        assert n_slopes <= 3, 'Maximum number of supported slopes is 3.'
        self._n_slopes = n_slopes

    def get_max_n_slopes(self):
        if self._n_slopes == 0:
            max_n_slopes = 3
        else:
            max_n_slopes = self._n_slopes
        return max_n_slopes

    def get_parameter_space(self):
        t_space = np.linspace(self._t_range[0], self._t_range[1], self._n_points_per_dim)
        a_space = np.logspace(self._a_range[0], self._a_range[1], self._n_points_per_dim)
        n_space = np.logspace(self._n_range[0], self._n_range[1], self._n_points_per_dim)
        return t_space, a_space, n_space

    def set_parameter_ranges(self, parameter_ranges):
        self._parameter_ranges = parameter_ranges

        t_range, a_range, n_range = self.get_parameter_ranges()
        self._t_range = t_range
        self._a_range = a_range
        self._n_range = n_range

    def get_parameter_ranges(self):
        assert ('t_range' in self._parameter_ranges) and ('a_range' in self._parameter_ranges) and (
                'n_range' in self._parameter_ranges), \
            'Parameter ranges must constist of the fields t_range, a_range, and n_range.'

        t_range = self._parameter_ranges['t_range']
        a_range = self._parameter_ranges['a_range']
        n_range = self._parameter_ranges['n_range']

        assert (len(t_range) == 2) and (len(a_range) == 2) and (len(n_range) == 2), \
            't_range, a_range, and n_range must be given as lists with two elements [min_val, max_val].'
        assert (t_range[1] > t_range[0]) and (a_range[1] > a_range[0]) and (n_range[1] > n_range[0]), \
            'First value in t_range, a_range, and n_range must be smaller than second value.'

        return t_range, a_range, n_range

    def set_n_points_per_dim(self, n_points_per_dim):
        self._n_points_per_dim = n_points_per_dim

        # Re-Init parameter space
        t_space, a_space, n_space = self.get_parameter_space()
        self._t_space = t_space
        self._a_space = a_space
        self._n_space = n_space

    def estimate_parameters(self, input: np.ndarray, input_is_edc: bool = False) -> Tuple[List[np.ndarray], np.ndarray]:
        """ Estimates the parameters for this impulse response. The resulting fitted EDC is normalized to 0dB and can be
        re-normalized to the original level with norm_vals

        Args:
            input: [rir_length, 1], rir to be analyzed
            input_is_edc: bool that indicates if input is edc or rir

        The estimation returns:
            Tuple of:
            1) List with:
            -- t_prediction : [batch, 3] : time_values for the 3 slopes
            -- a_prediction : [batch, 3] : amplitude_values for the 3 slopes
            -- n_prediction :  [batch, 1] : noise floor
            2) norm_vals : [batch, n_bands] : EDCs are normalized to 0dB, as customary for most decay analysis problems,
                                              but if the initial level is required, norm_vals will return it

        """
        # Pre-process RIR to get EDCs
        edcs, time_axis_ds, norm_vals, scale_adjust_factors = self._preprocess(input, input_is_edc)
        edcs = edcs.detach().numpy()
        time_axis_ds = time_axis_ds.detach().numpy()
        norm_vals = norm_vals.detach().numpy()

        # Init arrays
        n_bands = edcs.shape[0]
        t_vals = np.zeros((n_bands, self.get_max_n_slopes()))
        a_vals = np.zeros((n_bands, self.get_max_n_slopes()))
        n_vals = np.zeros((n_bands, 1))

        # Do Bayesian
        for band_idx, edc_this_band in enumerate(edcs):
            t_prediction, a_prediction, n_prediction = self._estimation(edc_this_band, time_axis_ds)
            n_slopes_prediction = t_prediction.shape[0]
            t_vals[band_idx, 0:n_slopes_prediction] = t_prediction
            a_vals[band_idx, 0:n_slopes_prediction] = a_prediction
            n_vals[band_idx, 0] = n_prediction

        # Postprocess parameters
        exactly_n_slopes_mode = (self._n_slopes != 0)
        t_vals, a_vals, n_vals = _postprocess_parameters(t_vals, a_vals, n_vals, scale_adjust_factors,
                                                         exactly_n_slopes_mode)

        return [t_vals, a_vals, n_vals], norm_vals

    def _estimation(self, edc_db, time_axis):
        # Following Xiang, N., Goggans, P., Jasa, T. & Robinson, P. "Bayesian characterization of multiple-slope sound
        # energy decays in coupled-volume systems." J Acoust Soc Am 129, 741–752 (2011).
        assert (edc_db.shape[0] == time_axis.shape[0]), "Time axis does not match EDC."

        if self._n_slopes == 0:
            model_orders = [1, 2, 3]  # estimate number of slopes according to BIC
        else:
            model_orders = [self._n_slopes]

        all_max_likelihood_params = []
        all_bics = []
        # Go through all possible model orders and find max likelihood
        for this_model_order in model_orders:
            # Do slice sampling to determine likelihoods for multiple parameter combinations
            tested_parameters, likelihoods = self._slice_sampling(edc_db, this_model_order, time_axis)

            # Find maximum likelihood and corresponding parameter combination
            max_likelihood_idx = np.argmax(likelihoods)
            max_likelihood = likelihoods[max_likelihood_idx]
            max_likelihood_params = tested_parameters[max_likelihood_idx, :]
            all_max_likelihood_params.append(max_likelihood_params)

            # Determine BIC for this maximum likelihood and model order: this is used to estimate the model order
            # if desired
            bic = 2*np.log(max_likelihood) - (2*this_model_order + 1)*np.log(len(time_axis))  # Eq. (15)
            all_bics.append(bic)

        # Find model with highest BIC: model that describes data best with most concise model
        best_model_order_idx = np.argmax(all_bics)
        best_model_order = model_orders[best_model_order_idx]
        best_model_params = all_max_likelihood_params[best_model_order_idx]

        t_vals = self._t_space[best_model_params[0:best_model_order]]
        a_vals = self._a_space[best_model_params[best_model_order:2*best_model_order]]
        n_val = self._n_space[best_model_params[2*best_model_order]]

        return t_vals, a_vals, n_val

    def _slice_sampling(self, edc_db, model_order, time_axis):
        # Following Jasa, T. & Xiang, N. "Efficient estimation of decay parameters in acoustically coupled-spaces using
        # slice sampling." J Acoust Soc Am 126, 1269–1279 (2009).
        assert self._t_space.shape[0] == self._a_space.shape[0] and self._t_space.shape[0] == self._n_space.shape[0], \
            "There must be an equal number of T, A, and N values in the parameter space."
        n_parameters = model_order * 2 + 1

        tested_paramters = np.zeros((self.n_iterations, n_parameters), dtype=np.uint16)
        likelihoods = np.zeros(self.n_iterations)

        # randomly draw first parameter values (indices)
        x0 = np.random.randint(self._n_points_per_dim, size=n_parameters)

        # evaluate likelihood for these paramters, and multiply with a random number between 0...1 to determine a
        # likelihood threshold
        y0 = np.random.rand() * evaluate_likelihood(edc_db,
                                                    self._t_space[x0[0:model_order]],
                                                    self._a_space[x0[model_order:2 * model_order]],
                                                    self._n_space[x0[2 * model_order]],
                                                    time_axis)

        # start to iterate
        for sample_idx in range(self.n_iterations):
            # determine which variable is varied: variables are varied in turn
            param_idx = sample_idx % n_parameters

            # ======
            # 1: Vary parameter until slice is established : the slice is the region, for which the likelihood is
            # higher than the previously found likelihood threshold y0

            # Find left edge of slice: decrease parameter value until likelihood is below threshold
            this_x0_left = copy.copy(x0)
            while this_x0_left[param_idx] > 0:
                this_x0_left[param_idx] -= 1
                this_y0_left = evaluate_likelihood(edc_db,
                                                   self._t_space[this_x0_left[0:model_order]],
                                                   self._a_space[this_x0_left[model_order:2 * model_order]],
                                                   self._n_space[this_x0_left[2 * model_order]],
                                                   time_axis)

                if this_y0_left < y0:
                    break

            # Find right edge of slice: increase parameter value until likelihood is below threshold
            this_x0_right = copy.copy(x0)
            while this_x0_right[param_idx] < self._n_points_per_dim - 1:
                this_x0_right[param_idx] += 1
                this_y0_right = evaluate_likelihood(edc_db,
                                                    self._t_space[this_x0_right[0:model_order]],
                                                    self._a_space[this_x0_right[model_order:2 * model_order]],
                                                    self._n_space[this_x0_right[2 * model_order]],
                                                    time_axis)

                if this_y0_right < y0:
                    break

            # ======
            # 2: Draw new parameter value from the slice, to find a new and higher likelihood (and threshold)
            while True:
                # copy old parameter values
                x1 = copy.copy(x0)

                # randomly draw varied parameter (index) from the slice
                # +1 to avoid randi(0), which gives error. (-1 not required, because randint excludes upper limit)
                x1[param_idx] = np.random.randint(this_x0_right[param_idx] -
                                                  this_x0_left[param_idx] + 1) + this_x0_left[param_idx]

                # evaluate likelihood for drawn parameter
                y1 = evaluate_likelihood(edc_db,
                                         self._t_space[x1[0:model_order]],
                                         self._a_space[x1[model_order:2 * model_order]],
                                         self._n_space[x1[2 * model_order]],
                                         time_axis)

                if y1 > y0:
                    # higher likelihood found, continue with next iteration step
                    break
                else:
                    # drawn value is not actually in slice due to sampling rate error (slice is established as
                    # multiples of a step), therefore adapt slice interval

                    # find out which edge of the slice was wrong
                    if np.linalg.norm(x1-this_x0_left) < np.linalg.norm(x1-this_x0_right):
                        this_x0_left = copy.copy(x1)
                    else:
                        this_x0_right = copy.copy(x1)

            # Save tested parameters and likelihood of this iteration
            tested_paramters[sample_idx, :] = copy.copy(x1)
            likelihoods[sample_idx] = y1

            # prepare for next iteration: new threshold
            y0 = np.random.rand() * y1
            x0 = copy.copy(x1)

        return tested_paramters, likelihoods
