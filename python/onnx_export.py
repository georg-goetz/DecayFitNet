import os
import pickle
from pathlib import Path

import torch.onnx
import toolbox.core as core
import toolbox.utils as utils

# Exports DecayFitNet with exactly N Slopes to ONNX
# Important: MATLAB only works with ONNX 9 and pickle v2, Python works with ONNX 10 and regular pickle

# Hyperparemeters fron the training
UNITS_PER_LAYER = 1500
DROPOUT = 0.0
N_LAYERS = 3
N_FILTER = 128

# Path to weights and export directory
NETWORK_NAME = 'DecayFitNet.pth'
PATH_ONNX = Path.joinpath(Path(__file__).parent.parent, 'model')


def get_net():
    """ Loads the pretrained model and weights. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = core.DecayFitNetLinear(3, UNITS_PER_LAYER, N_FILTER, N_LAYERS, 0, DROPOUT, 1, device)
    utils.load_model(net, os.path.join(PATH_ONNX, NETWORK_NAME), device)
    net.eval()

    return net


def export_onnx(onnx_version=10):
    """ Exports the pretrained model into ONNX format. """
    batch_size = 1
    channels = 1
    timesteps = 2400
    net = get_net()

    # Input to the model
    x = torch.randn(batch_size, timesteps, requires_grad=True)  # (batch_size, 2400), no channels
    torch_out = net(x)

    # Export the model
    print('=== Exporting model to ONNX')
    torch.onnx.export(net,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      os.path.join(PATH_ONNX, f"DecayFitNet_v{onnx_version}.onnx"),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=onnx_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def export_input_transform2matlab(protocol=4):
    """ Helper function to dump the input transform in a different pickle protocol.
        This is useful for reading in other platforms (e.g. Matlab)
        """
    with open(os.path.join(PATH_ONNX, 'input_transform.pkl'), 'rb') as f:
        data = pickle.load(f)

    with open(os.path.join(PATH_ONNX, f'input_transform_p{protocol}.pkl'), 'wb') as f:
        # Parse to dictionary without tensors
        data = {k: v.numpy().tolist() for k, v in data.items()}
        pickle.dump(data, f, protocol=protocol)

if __name__ == '__main__':
    export_onnx(onnx_version=9)
    export_onnx(onnx_version=10)
    print('Finished exporting DecayFitNet to ONNX file.')
    export_input_transform2matlab(protocol=2)
    print('Finished exporting input transform file.')



