# Pre processing:
# 	Load impulse  or path to impulse
# 	Filter bank
# 	Backwards integration
# 	Resampling to 100 samples
# 	Ampltiude2DB
# 	Normalization (wtih training data parameters)
#
# Reshaping freq bands to batch size.
#
# Output:
# 	1 fitting per octave band ( 3 t , 3 a, 1 noise)
# 	Reshape output if multiple freq bands
# 	MSE of EDC fit (generate_synthetic_edc() net --> EDC)
#
# Plots:
# 	Input EDCs
# 	Output EDCs

import torch
import numpy as np
import pickle
import onnx
import onnxruntime
import os
from pathlib import Path
from typing import Union, List, Tuple

from .core import PreprocessRIR, _postprocess_parameters


class DecayFitNetToolbox:
    def __init__(self, sample_rate: int = 48000, backend: str = 'pytorch', device: torch.device = torch.device('cpu'),
                 filter_frequencies: List = None):
        self._version = '0.1.0'
        self.backend = backend
        self.device = device

        self._sample_rate = sample_rate

        PATH_ONNX = Path.joinpath(Path(__file__).parent.parent.parent, 'model')

        self._onnx_model = onnx.load(os.path.join(PATH_ONNX, "DecayFitNet_v10.onnx"))
        onnx.checker.check_model(self._onnx_model)
        self._session = onnxruntime.InferenceSession(os.path.join(PATH_ONNX, "DecayFitNet_v10.onnx"))
        self._input_transform = pickle.load(open(os.path.join(PATH_ONNX, 'input_transform.pkl'), 'rb'))

        self._preprocess = PreprocessRIR(input_transform=self._input_transform,
                                         sample_rate=self._sample_rate,
                                         output_size=100,
                                         filter_frequencies=filter_frequencies)

    def __repr__(self):
        frmt = f'DecayFitNetToolbox {self._version}  \n'
        frmt += f'Input fs = {self._sample_rate} \n'
        frmt += f'Output_size = {self.get_output_size()} \n'

        return frmt

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def set_filter_frequencies(self, filter_frequencies):
        self._preprocess.set_filter_frequencies(filter_frequencies)

    def get_filter_frequencies(self):
        return self._preprocess.get_filter_frequencies()

    def set_output_size(self, output_size):
        self._preprocess.output_size = output_size

    def get_output_size(self):
        return self._preprocess.output_size

    def preprocess(self, signal: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess an input signal to extract EDCs"""
        edcs, norm_vals = self._preprocess(signal)
        return edcs, norm_vals

    def estimate_parameters(self, signal: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """ Estimates the parameters for this impulse response. The resulting fitted EDC is normalized to 0dB and can be
        re-normalized to the original level with norm_vals

        Args:
            signal: [rir_length, 1], rir to be analyzed

        The estimation returns:
            Tuple of:
            1) List with:
            -- t_prediction : [batch, 3] : time_values for the 3 slopes
            -- a_prediction : [batch, 3] : amplitude_values for the 3 slopes
            -- n_prediction :  [batch, 1] : noise floor
            2) norm_vals : [batch, n_bands] : EDCs are normalized to 0dB, as customary for most decay analysis problems,
                                              but if the initial level is required, norm_vals will return it

        """
        edcs, __, norm_vals, scale_adjust_factors = self._preprocess(signal)

        ort_inputs = {self._session.get_inputs()[0].name: DecayFitNetToolbox._to_numpy(edcs)}
        ort_outs = self._session.run(None, ort_inputs)
        # ort_outs = [torch.from_numpy(jj) for jj in ort_outs]
        t_prediction, a_prediction, n_exp_prediction, n_slopes_probabilities = ort_outs

        # Clamp noise to reasonable values to avoid numerical problems
        n_exp_prediction = np.clip(n_exp_prediction, -32, 32)
        # Go from noise exponent to noise value
        n_prediction = np.power(10, n_exp_prediction)

        # Get a binary mask to only use the number of slopes that were predicted, zero others
        n_slopes_prediction = np.argmax(n_slopes_probabilities, 1)
        n_slopes_prediction += 1  # because python starts at 0
        temp = np.tile(np.linspace(1, 3, 3, dtype=np.uint8), (n_slopes_prediction.shape[0], 1))
        mask = (np.tile(np.expand_dims(n_slopes_prediction, 1), (1, 3)) < temp)
        a_prediction[mask] = 0

        n_slopes_estimation_mode = True  # TODO: change this after nSlopes are part of python toolbox
        t_prediction, a_prediction, n_prediction = _postprocess_parameters(t_prediction, a_prediction, n_prediction,
                                                                           scale_adjust_factors,
                                                                           n_slopes_estimation_mode)

        return [t_prediction, a_prediction, n_prediction], norm_vals
