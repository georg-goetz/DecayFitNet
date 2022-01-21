import os
import pickle
from pathlib import Path

import torch.onnx
import toolbox.core as core
import toolbox.utils as utils

# Exports DecayFitNet to ONNX
# Important: MATLAB only works with ONNX 9 and pickle v2, Python works with ONNX 10 and regular pickle

# Training hyperparameters
UNITS_PER_LAYER = 400
N_LAYERS = 3
N_FILTER = 64
N_SLOPES = 1
EXACTLY_N_SLOPE_MODE = True


def get_net():
    """ Loads the pretrained model and weights. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = core.DecayFitNet(n_slopes=N_SLOPES, n_max_units=UNITS_PER_LAYER, n_filters=N_FILTER, n_layers=N_LAYERS,
                           relu_slope=0, dropout=0, reduction_per_layer=1, device=device,
                           exactly_n_slopes_mode=EXACTLY_N_SLOPE_MODE)
    utils.load_model(net, os.path.join(path_onnx, network_name), device)
    net.eval()

    return net


def export_onnx(onnx_version=10):
    """ Exports the pretrained model into ONNX format. """
    batch_size = 1
    timesteps = 100
    net = get_net()

    # Input to the model
    x = torch.randn(batch_size, timesteps, requires_grad=True)  # (batch_size, 100), no channels
    __ = net(x)  # just dummy call so that ONNX can pass data through the network once

    # Export the model
    print('=== Exporting model to ONNX')
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      os.path.join(path_onnx, f"DecayFitNet{n_slope_str}_v{onnx_version}.onnx"),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=onnx_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def export_input_transform2matlab(protocol=2):
    """ Helper function to dump the input transform in a different pickle protocol.
        This is useful for reading in other platforms (e.g. Matlab)
        """
    with open(os.path.join(path_onnx, f'input_transform{n_slope_str}.pkl'), 'rb') as f:
        data = pickle.load(f)

    with open(os.path.join(path_onnx, f'input_transform{n_slope_str}_p{protocol}.pkl'), 'wb') as f:
        # Parse to dictionary without tensors
        data = {k: v.numpy().tolist() for k, v in data.items()}
        pickle.dump(data, f, protocol=protocol)


if __name__ == '__main__':
    if EXACTLY_N_SLOPE_MODE:
        n_slope_str = f'_{N_SLOPES}slopes'
    else:
        n_slope_str = ''

    # Path to weights and export directory
    network_name = f'DecayFitNet{n_slope_str}.pth'
    path_onnx = Path.joinpath(Path(__file__).parent.parent, 'model')

    export_onnx(onnx_version=9)
    export_onnx(onnx_version=10)
    print('Finished exporting DecayFitNet to ONNX file.')
    export_input_transform2matlab(protocol=2)
    print('Finished exporting input transform file.')

    if os.path.exists(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.m')) or \
            os.path.exists(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.mat')):

        answer = input('Matlab files of previous model were found. They must be deleted for the new network to work. '
                       'Delete them? [y/n] \n')

        if answer.lower() == 'y':
            if os.path.exists(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.m')):
                os.remove(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.m'))
            if os.path.exists(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.mat')):
                os.remove(Path.joinpath(path_onnx, f'DecayFitNet{n_slope_str}_model.mat'))




