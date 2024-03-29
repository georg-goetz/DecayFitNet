classdef DecayFitNetToolbox < handle
    
    properties        
    end
    properties (SetAccess = private)
        version = '0.1.2'
        outputSize = 100  % Timesteps of resampled RIRs
        nSlopes
        onnxPath
        onnxModel
        networkName
        inputTransform
        preprocessing
        sampleRate
    end
    
    methods
        function obj = DecayFitNetToolbox(nSlopes, sampleRate, filterFrequencies)
            if nargin < 1
                nSlopes = 0; % estimate number of slopes from data
            end
            if nargin < 2
                sampleRate = 48000;
            end
            if nargin < 3
                filterFrequencies = [];
            end
                
            obj.sampleRate = sampleRate;
            obj.nSlopes = nSlopes;
            
            % Load onnx model
            [thisDir, ~, ~] = fileparts(mfilename('fullpath'));
            obj.onnxPath = fullfile(thisDir, '..', 'model');
            addpath(obj.onnxPath);
            
            if obj.nSlopes == 0
                % infer number of slopes with network
                slopeMode = '';
            elseif obj.nSlopes == 1 || obj.nSlopes == 2 || obj.nSlopes == 3
                % fit exactly n slopes plus noise
                slopeMode = sprintf('%dslopes_', obj.nSlopes);
            else
                error('Please specify a valid number of slopes to be predicted by the network (nSlopes=1,2,3 for 1,2,3 slopes plus noise, respectively, or nSlopes=0 to let the network infer the number of slopes [max 3 slopes]).');
            end
            
            obj.networkName = sprintf('DecayFitNet_%s', slopeMode);
            
            pe = pyenv;
            if strcmp(pe.Version, "") || isempty(pe.Version)
                error(sprintf("MATLAB does not know where your Python executable is located. \n Please run pyenv('Version', <path_to_your_python_executable>). \n\n More info: https://www.mathworks.com/help/releases/R2022a/matlab/matlab_external/install-supported-python-implementation.html"));
            end

            % Load ONNX model:
            if exist(fullfile(obj.onnxPath, [obj.networkName, 'model.mat']), 'file')
                fprintf('Loading precompiled model %smodel.mat\n', obj.networkName)
                obj.onnxModel = load(fullfile(obj.onnxPath, [obj.networkName, 'model.mat'])).tmp;
            else
                obj.onnxModel = importONNXFunction(fullfile(obj.onnxPath, [obj.networkName, 'v9.onnx']), fullfile(obj.onnxPath, [obj.networkName, 'model']));
                tmp = obj.onnxModel;
                save(fullfile(obj.onnxPath, [obj.networkName, 'model.mat']), 'tmp');
            end
            [~, msgid] = lastwarn;
            if strcmp(msgid, 'MATLAB:load:cannotInstantiateLoadedVariable')
                error('Could not load the ONNX model. Please make sure you are running MATLAB 2020b or later and installed the following toolbox to load ONNX models in MATLAB: https://se.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format');
            end
            
            disp(obj.onnxModel)
            
            % Load input transform for preprocessing the network inputs
            fid = py.open(fullfile(obj.onnxPath, sprintf('input_transform_%sp2.pkl', slopeMode)),'rb');
            obj.inputTransform = py.pickle.load(fid);
            
            % Init preprocessing
            obj.preprocessing = PreprocessRIR(obj.inputTransform, sampleRate, filterFrequencies, obj.outputSize);
        end
                        
        function setFilterFrequencies(obj, filterFrequencies)
            obj.preprocessing.filterFrequencies = filterFrequencies;
        end
        
        function filterFrequencies = getFilterFrequencies(obj)
            filterFrequencies = obj.preprocessing.filterFrequencies;
        end
        
        function [tPrediction, aPrediction, nPrediction, normVals] = estimateParameters(obj, input, inputIsEDC)
            % Output is [nBands, nSlopes]
            
            if ~exist('inputIsEDC', 'var')
                inputIsEDC = false;
            end
            
            [edcs, ~, normVals, scaleAdjustFactors] = obj.preprocessing.preprocess(input, inputIsEDC);
            
            % Forward pass of the DecayFitNet
            net = str2func([obj.networkName, 'model']);
            [tPrediction, aPrediction, nExpPrediction, nSlopesProbabilities] = net(edcs, obj.onnxModel);
            
            % Transpose results to make them [batchSize, nSlopes]
            tPrediction = tPrediction.';
            aPrediction = aPrediction.';
            nExpPrediction = nExpPrediction.';
            nSlopesProbabilities = nSlopesProbabilities.';
            
            % Clamp noise to reasonable values to avoid numerical problems
            nExpPrediction = min(max(nExpPrediction, -32), 32);

            % Go from noise exponent to noise value
            nPrediction = 10 .^ nExpPrediction;

            % In nSlope inference mode: Get a binary mask to only use the number of slopes that were predicted, zero others
            nSlopeEstimationMode = (obj.nSlopes == 0);
            if nSlopeEstimationMode
                [~, nSlopesPrediction] = max(nSlopesProbabilities, [], 2);
                tmp = repmat(linspace(1,3,3), [length(nSlopesPrediction), 1]);
                mask = tmp <= repmat(nSlopesPrediction, [1, 3]);
                aPrediction(~mask) = 0;
            end
            
            % Postprocess parameters and also convert to original
            % time-scale
            [tPrediction, aPrediction, nPrediction] = postprocessDecayParameters(tPrediction, aPrediction, nPrediction, scaleAdjustFactors, nSlopeEstimationMode);
        end
    end
    
    methods(Static)            
        function output = discardLast5(signal)
            % Discard last 5% samples of a signal
            assert(size(signal,1) > size(signal,2), 'The signal should be in [timesteps, channels]')
            last5 = round(0.95 * size(signal,1));
            
            output = zeros(last5, size(signal,2));
            for channel = 1:size(signal,2)
                tmp = signal(:,channel);
                tmp(last5+1:end) = [];
                output(:,channel) = tmp;
            end
        end
    end
    
end
