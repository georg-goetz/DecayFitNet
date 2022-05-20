# DecayFitNet
This toolbox accompanies the following paper:
>Georg Götz, Ricardo Falcón Pérez, Sebastian J. Schlecht, and Ville Pulkki, *"Neural network for multi-exponential sound energy decay analysis"*, submitted to Journal of the Acoustical Society of America, 2022.
 
The paper is currently under review at the Journal of the Acoustical Society of America. A preprint can be found at https://doi.org/10.48550/arXiv.2205.09644

## External dependencies (MATLAB)
The following toolboxes must be installed in order to run the MATLAB version of the toolbox:
- https://se.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format
- Deep Learning Toolbox

Furthermore, you must have at least MATLAB 2020b to run the toolbox.

## External dependencies (Python)
The following toolboxes must be installed in order to run the Python version of the toolbox:
- numpy
- scipy
- pytorch
- onnx
- matplotlib
- h5py

For example, install the dependencies like this:
```
conda create -n decayfitnet anaconda
conda activate decayfitnet
conda install pytorch torchaudio -c pytorch
pip install onnx onnxruntime
```
