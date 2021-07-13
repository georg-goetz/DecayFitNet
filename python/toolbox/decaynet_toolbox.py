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
import scipy
import pickle
import onnx
import onnxruntime
import os
from pathlib import Path
from typing import Union, List

from core import PreprocessRIR_new, generate_synthetic_edc


class DecaynetToolbox():
    output_size = 2400  # Timesteps of downsampled RIRs
    filter_frequencies = [125, 250, 500, 1000, 2000, 4000]
    __version = '0.0.3'

    def __init__(self, sample_rate: int = 48000,  normalization: bool = True, backend: str = 'pytorch'):
        self.backend = backend
        self.fs = sample_rate
        self.normalization = normalization

        PATH_ONNX = Path.joinpath(Path(__file__).parent.parent.parent, 'data')

        self._onnx_model = onnx.load(os.path.join(PATH_ONNX, "decaynet.onnx"))
        onnx.checker.check_model(self._onnx_model)
        self._session = onnxruntime.InferenceSession(os.path.join(PATH_ONNX, "decaynet.onnx"))
        self.input_transform = pickle.load(open(os.path.join(PATH_ONNX, 'input_transform_final.pkl'), 'rb'))

        self._preprocess = PreprocessRIR_new(input_transform=self.input_transform,
                                             normalization=normalization,
                                             sample_rate=self.fs,
                                             filter_frequencies=self.filter_frequencies,
                                             output_size=self.output_size)

    def __repr__(self):
        frmt = f'DecaynetToolbox {self.__version}  \n'
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

    def estimate_parameters(self, signal: torch.Tensor, do_preprocess: bool = True) -> List[torch.Tensor]:
        """ Estimates the parameters for the impulse.

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
        # TODO: re order freqs and batches

        ort_inputs = {self._session.get_inputs()[0].name: DecaynetToolbox._to_numpy(edcs)}
        ort_outs = self._session.run(None, ort_inputs)

        return [torch.from_numpy(jj) for jj in ort_outs]

    def estimate_EDC(self,
                     estimated_T: Union[torch.Tensor, List[float]],
                     estimated_A: Union[torch.Tensor, List[float]],
                     estimated_n: Union[torch.Tensor, List[float]],
                     n_slopes_probabilities: Union[torch.Tensor, List[float]],
                     time_axis: Union[torch.Tensor, np.ndarray] = None,
                     device: str = 'cpu') -> torch.Tensor:
        """Generates an EDC from the estimated parameters.

        estimated_T, estimated_A, n_exp_prediction should be [batch, dim]
        Example:
                >> fs = 240
                >> l_edc = 10
                >> t = np.linspace(0, l_edc * fs - 1, l_edc * fs) / fs
                >> edc = estimate_EDC([1.3, 0.7], [0.7, 0.2], 1e-7, 0.1, t, "cpu")  # TODO
        """

        # TODO fix comment example

        # Only use the number of slopes that were predicted, zero others
        _, n_slopes_prediction = torch.max(n_slopes_probabilities, 1)
        n_slopes_prediction += 1  # because python starts at 0
        temp = torch.linspace(1, 3, 3).repeat(n_slopes_prediction.shape[0], 1).to(device)
        mask = temp.less_equal(n_slopes_prediction.unsqueeze(1).repeat(1, 3))
        estimated_A[~mask] = 0

        # Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
        estimated_n = torch.clamp(estimated_n, -32, 32)
        estimated_n_fixed = torch.pow(10, estimated_n)

        if time_axis is None:
            fs = 240
            l_edc = 10
            # time_axis = (torch.linspace(0, l_edc * fs - 1, l_edc * fs) / fs).to(device)  # TODO remove old version
            time_axis = (torch.linspace(0, l_edc * fs - 1, round((1/0.95)*l_edc * fs)) / fs).to(device)

        out = []
        if len(estimated_T.shape) > 1:
            for idx in range(estimated_T.shape[0]):
                tmp = generate_synthetic_edc(T=estimated_T[idx:idx+1, :],
                                             A=estimated_A[idx:idx+1, :],
                                             noiseLevel=estimated_n_fixed[idx],
                                             t=time_axis,
                                             device=device)
                # discard last 5 percent 
                tmp = tmp[:, 0:l_edc*fs]
                out.append(tmp)
            out = torch.stack(out, dim=0)  # Stack over batch
        else:
            out = generate_synthetic_edc(T=estimated_T,
                                         A=estimated_A,
                                         noiseLevel=estimated_n_fixed[idx],
                                         t=time_axis,
                                         device=device)
            # discard last 5 percent 
            out = out[:, 0:l_edc*fs]

        return out




