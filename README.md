# DecayFitNet
This toolbox accompanies the following paper:
>Georg Götz, Ricardo Falcón Pérez, Sebastian J. Schlecht, and Ville Pulkki, *"Neural network for multi-exponential sound energy decay analysis"*, The Journal of the Acoustical Society of America, 152(2), pp. 942-953, 2022, https://doi.org/10.1121/10.0013416.
 
The paper was published in The Journal of the Acoustical Society of America. It can be found at https://doi.org/10.1121/10.0013416

Please refer to the demo files for a tutorial on how to use this toolbox. A more thorough documentation might be added at a later point, if required. 

## External dependencies (MATLAB)
The following toolboxes must be installed in order to run the MATLAB version of the toolbox:
- https://se.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format
- Deep Learning Toolbox

Furthermore, you must have at least MATLAB 2020b to run the toolbox. You will also need a Python installation on your machine, and you have to let MATLAB know where it is located, e.g. using ```pyversion(<your_python_path>)```.

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

## ToDos: 
Here is a list of ToDos and planned features. If you wrote your own code and feel that it could be part of this toolbox, or you encounter any problems, please reach out to me or submit an issue/pull request. 
- Efficient batch processing: while the original pyTorch models can be easily used for batch processing of multiple EDFs, the current architecture of MATLAB and python wrappers does not support this yet. Of course this feature would be very desirable in the future, so it will be definitely worked on at some point.
- Optimization of the octave filters: the octave-band filtering for sound energy decay analysis is an interesting problem, which needs some more work. There is a trade-off between filter selectivity and artefacts (pre-ringing, etc.). We have already spent some time working on this, but we have not come up with a satisfactory solution yet. 
- Speeding up the Bayesian analysis: at this point the Bayesian analysis is somewhat slow. This may very well be due to our implementation. If you are an expert in Bayesian decay analysis and have your own code that you want to contribute, please let us know.
