# Pre processing:
# 	Load impulse  or path to impulse
# 	Filter bank
# 	Backwards integration (cut arway at -140 db)
# 	Downsampling to 2400 samples  (pad with last value)
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
from typing import Union, List

from .core import PreprocessRIR, generate_synthetic_edc, postprocess_parameters, adjust_timescale


class DecayFitNetToolbox:
    output_size = 2400  # Timesteps of downsampled RIRs
    filter_frequencies = [125, 250, 500, 1000, 2000, 4000]
    __version = '0.0.3'

    def __init__(self, sample_rate: int = 48000,  normalization: bool = True, backend: str = 'pytorch',
                 device: torch.device = torch.device('cpu')):
        self.backend = backend
        self.fs = sample_rate
        self.normalization = normalization
        self.device = device

        PATH_ONNX = Path.joinpath(Path(__file__).parent.parent.parent, 'model')

        self._onnx_model = onnx.load(os.path.join(PATH_ONNX, "DecayFitNet.onnx"))
        onnx.checker.check_model(self._onnx_model)
        self._session = onnxruntime.InferenceSession(os.path.join(PATH_ONNX, "DecayFitNet.onnx"))
        self.input_transform = pickle.load(open(os.path.join(PATH_ONNX, 'input_transform.pkl'), 'rb'))

        self._preprocess = PreprocessRIR(input_transform=self.input_transform,
                                         normalization=normalization,
                                         sample_rate=self.fs,
                                         filter_frequencies=self.filter_frequencies,
                                         output_size=self.output_size)

    def __repr__(self):
        frmt = f'DecayFitNetToolbox {self.__version}  \n'
        frmt += f'Input fs = {self.fs} \n'
        frmt += f'Output_size = {self.output_size} \n'
        frmt += f'Normalization = {self.normalization} \n'
        frmt += f'Filter freqs = {self.filter_frequencies} \n'
        #frmt += f'Using model: {self._onnx_model}'

        return frmt

    @staticmethod
    def _to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    def preprocess(self, signal: torch.Tensor) -> torch.Tensor:
        """Preprocess an input signal to extract EDCs"""
        edcs = self._preprocess(signal)
        return edcs

    def estimate_parameters(self, signal: torch.Tensor, do_preprocess: bool = True, do_scale_adjustment: bool = True) \
            -> List[torch.Tensor]:
        """ Estimates the parameters for the impulse.
        TODO input parameter description

        The estimation returns:
            -- t_vals : [batch, 3] : time_values for the 3 slopes
            -- a_vals : [batch, 3] : amplitude_values for the 3 slopes
            -- noise_vals :  [batch, 1] : noise floor
            -- n_slopes : [batch, 3] : probabilities for each slope

        """
        if do_preprocess:
            edcs = self._preprocess(signal)
        else:
            edcs = signal

        ort_inputs = {self._session.get_inputs()[0].name: DecayFitNetToolbox._to_numpy(edcs)}
        ort_outs = self._session.run(None, ort_inputs)
        ort_outs = [torch.from_numpy(jj) for jj in ort_outs]

        t_prediction, a_prediction, n_prediction, __ = postprocess_parameters(ort_outs[0], ort_outs[1], ort_outs[2],
                                                                              ort_outs[3], self.device)

        if do_scale_adjustment:
            t_prediction, n_prediction = adjust_timescale(t_prediction, n_prediction, signal.shape[1], self.fs)

        return [t_prediction, a_prediction, n_prediction]

    def generate_EDCs(self,
                      estimated_T: Union[torch.Tensor, List[float]],
                      estimated_A: Union[torch.Tensor, List[float]],
                      estimated_n: Union[torch.Tensor, List[float]],
                      time_axis: Union[torch.Tensor, np.ndarray] = None) -> torch.Tensor:
        """Generates an EDC from the estimated parameters.

        estimated_T, estimated_A, n_exp_prediction should be [batch, dim]
        Example:
                >> fs = 240
                >> l_edc = 10
                >> t = np.linspace(0, l_edc * fs - 1, l_edc * fs) / fs
                >> edc = estimate_EDC([1.3, 0.7], [0.7, 0.2], 1e-7, 0.1, t, "cpu")  # TODO
        """
        # TODO fix comment example

        # Write arbitary number (1) into T values that are equal to zero (inactive slope), because their amplitude will
        # be 0 as well (i.e. they don't contribute to the EDC)
        estimated_T[estimated_T == 0] = 1

        if time_axis is None:
            fs = 240
            l_edc = 10
            time_axis = (torch.linspace(0, l_edc * fs - 1, round((1/0.95)*l_edc * fs)) / fs).to(self.device)

        out = []
        if len(estimated_T.shape) > 1:
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
