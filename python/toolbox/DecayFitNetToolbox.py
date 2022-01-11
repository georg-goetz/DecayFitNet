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

from .core import PreprocessRIR, generate_synthetic_edc, postprocess_parameters, adjust_timescale


class DecayFitNetToolbox:
    output_size = 100  # Timesteps of downsampled RIRs
    __version = '0.1.0'

    def __init__(self, sample_rate: int = 48000, filter_frequencies=None,
                 backend: str = 'pytorch', device: torch.device = torch.device('cpu')):

        if filter_frequencies is None:
            filter_frequencies = [125, 250, 500, 1000, 2000, 4000]

        self.backend = backend
        self.fs = sample_rate
        self.device = device
        self._filter_frequencies = filter_frequencies

        PATH_ONNX = Path.joinpath(Path(__file__).parent.parent.parent, 'model')

        self._onnx_model = onnx.load(os.path.join(PATH_ONNX, "DecayFitNet_v10.onnx"))
        onnx.checker.check_model(self._onnx_model)
        self._session = onnxruntime.InferenceSession(os.path.join(PATH_ONNX, "DecayFitNet_v10.onnx"))
        self.input_transform = pickle.load(open(os.path.join(PATH_ONNX, 'input_transform.pkl'), 'rb'))

        self._preprocess = PreprocessRIR(input_transform=self.input_transform,
                                         sample_rate=self.fs,
                                         filter_frequencies=self._filter_frequencies,
                                         output_size=self.output_size)

    def __repr__(self):
        frmt = f'DecayFitNetToolbox {self.__version}  \n'
        frmt += f'Input fs = {self.fs} \n'
        frmt += f'Output_size = {self.output_size} \n'
        frmt += f'Filter freqs = {self._filter_frequencies} \n'

        return frmt

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def set_filter_frequencies(self, filter_frequencies):
        self._filter_frequencies = filter_frequencies
        self._preprocess.set_filter_frequencies(filter_frequencies)

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
        ort_outs = [torch.from_numpy(jj) for jj in ort_outs]

        t_prediction, a_prediction, n_prediction, __ = postprocess_parameters(ort_outs[0], ort_outs[1], ort_outs[2],
                                                                              ort_outs[3], self.device)

        t_prediction, n_prediction = adjust_timescale(t_prediction, n_prediction, scale_adjust_factors)

        return [t_prediction, a_prediction, n_prediction], norm_vals

    def generate_EDCs(self,
                      estimated_T: Union[torch.Tensor, List[float]],
                      estimated_A: Union[torch.Tensor, List[float]],
                      estimated_n: Union[torch.Tensor, List[float]],
                      time_axis: Union[torch.Tensor, np.ndarray] = None) -> torch.Tensor:
        """Generates an EDC from the estimated parameters.

        estimated_T, estimated_A, n_exp_prediction should be [batch, dim]
        Example:
                >> fs = 10
                >> l_edc = 10
                >> t = np.linspace(0, l_edc * fs - 1, l_edc * fs) / fs
                >> edc = estimate_EDC([1.3, 0.7], [0.7, 0.2], 1e-7, 0.1, t, "cpu")  # TODO
        """
        # TODO fix comment example

        # Avoid inf/nan for T=0: Write arbitary number (1) into T values that are equal to zero (inactive slope),
        # because their amplitude will be 0 as well (i.e. they don't contribute to the EDC)
        estimated_T[estimated_T == 0] = 1

        if time_axis is None:
            fs = 10
            l_edc = 10
            time_axis = (torch.linspace(0, l_edc * fs - 1, round((1/0.95)*l_edc * fs)) / fs).to(self.device)

        out = []
        if len(estimated_T.shape) > 1:  # if there are more than 1 RIRs
            for idx in range(estimated_T.shape[0]):
                tmp = generate_synthetic_edc(T=estimated_T[idx:idx+1, :],
                                             A=estimated_A[idx:idx+1, :],
                                             noiseLevel=estimated_n[idx],
                                             t=time_axis,
                                             device=self.device)
                # discard last 5 percent
                tmp = tmp[:, 0:round(0.95*tmp.shape[1])]
                out.append(tmp)
            out = torch.stack(out, dim=0)  # Stack over batch
        else:
            out = generate_synthetic_edc(T=estimated_T,
                                         A=estimated_A,
                                         noiseLevel=estimated_n,
                                         t=time_axis,
                                         device=self.device)
            # discard last 5 percent
            out = out[:, 0:round(0.95*out.shape[1])]

        return out
