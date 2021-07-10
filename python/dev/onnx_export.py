import os
from pathlib import Path

import torch.onnx
from python.dev import DecayFitNet
import tools


## NOTES:
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
#
#


# Hyperparemeters fron the training
UNITS_PER_LAYER = 1500
DROPOUT = 0.0
N_LAYERS = 3
N_FILTER = 128

# Path to weights and export directory
NETWORK_NAME = 'DecayFitNet_final_red100_sl3_3layers_128f_relu0_1500units_do0_b2048_lr5e3_sch20_e100_wd1e3.pth'
PATH_ONNX = Path.joinpath(Path(__file__).parent.parent.parent, 'data')


def get_net():
    """ Loads the pretrained model and weights. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DecayFitNet.DecayFitNetLinear(3, UNITS_PER_LAYER, N_FILTER, N_LAYERS, 0, DROPOUT, 1, device)
    tools.load_model(net, os.path.join(PATH_ONNX, NETWORK_NAME), device)
    net.eval()

    return net


def export_onnx():
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
                      os.path.join(PATH_ONNX, "decaynet.onnx"),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


if __name__ == '__main__':
    export_onnx()
    print('Finished exporting DecayFitNet to ONNX file.')



