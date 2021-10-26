classdef DecayFitNetToolbox < handle
    
    properties
        filter_frequencies = [125, 250, 500, 1000, 2000, 4000]
        sample_rate
    end
    properties (SetAccess = private)
        version = '0.0.7'
        output_size = 2400  % Timesteps of downsampled RIRs
        nSlopes
        PATH_ONNX
        onnx_model
        networkName
        input_transform
    end
    
    methods
        function obj = DecayFitNetToolbox(nSlopes, sampleRate)
            if nargin < 1
                nSlopes = 0;
            end
            if nargin < 2
                sampleRate = 48000;
            end
                
            obj.sample_rate = sampleRate;
            obj.nSlopes = nSlopes;
            
            % Load onnx model
            [thisDir, ~, ~] = fileparts(mfilename('fullpath'));
            obj.PATH_ONNX = fullfile(thisDir, '..', 'model');
            
            if obj.nSlopes == 0
                % infer number of slopes with network
                slopeMode = '';
            elseif obj.nSlopes == 1
                % fit exactly 1 slope plus noise
                slopeMode = '1slopes_';
            elseif obj.nSlopes == 2
                % fit exactly 2 slopes plus noise
                slopeMode = '2slopes_';
            elseif obj.nSlopes == 3
                % fit exactly 3 slopes plus noise
                slopeMode = '3slopes_';
            else
                error('Please specify a valid number of slopes to be predicted by the network (nSlopes=1,2,3 for 1,2,3 slopes plus noise, respectively, or nSlopes=0 to let the network infer the number of slopes [max 3 slopes]).');
            end
            obj.networkName = sprintf('DecayFitNet_%soffset_', slopeMode);
            
            % FAILS:
            % ONNX network with multiple outputs is not supported. Instead, use 'importONNXLayers' with 'ImportWeights' set to true.
            % obj.onnx_model = importONNXNetwork(fullfile(obj.PATH_ONNX, 'DecayFitNet.onnx'),'OutputLayerType','regression');
            
            % FAILS:
            % Warning: Unable to import some ONNX operators, because they are not supported. They have been replaced by placeholder layers. To find these layers, call the
            % function findPlaceholderLayers on the returned object.
            % obj.onnx_model = importONNXLayers(fullfile(obj.PATH_ONNX, 'DecayFitNet_v9.onnx'),'ImportWeights',true)  % Fails due to unsupported functions
            
            % Load ONNX model:
            if exist([obj.networkName, 'model.mat'], 'file')
                fprintf('Loading precompiled model %smodel.mat\n', obj.networkName)
                obj.onnx_model = load([obj.networkName, 'model.mat']).tmp;
            else
                obj.onnx_model = importONNXFunction(fullfile(obj.PATH_ONNX, [obj.networkName, 'v9.onnx']), [obj.networkName, 'model']);
                tmp = obj.onnx_model;
                save([obj.networkName, 'model.mat'], 'tmp');
            end
            [~, msgid] = lastwarn;
            if strcmp(msgid, 'MATLAB:load:cannotInstantiateLoadedVariable')
                error('Could not load the ONNX model. Please make sure you are running MATLAB 2020b or later and installed the following toolbox to load ONNX models in MATLAB: https://se.mathworks.com/matlabcentral/fileexchange/67296-deep-learning-toolbox-converter-for-onnx-model-format');
            end
            
            disp(obj.onnx_model)
            %[output, x66, x69, x72, state] = test_DecayFitNet(signal, '');
            
            fid = py.open(fullfile(obj.PATH_ONNX, sprintf('input_transform_%soffset_p2.pkl', slopeMode)),'rb');
            obj.input_transform = py.pickle.load(fid);
        end
                
        function [edcs, scaleAdjustFactors] = preprocess(obj, signal, includeResidualBands)
            if nargin < 3
                includeResidualBands = false;
            end
            
            nBands = length(obj.filter_frequencies);
            if includeResidualBands == true
                nBands = nBands + 2;
            end
            
            nRirs = 1;
            edcs = zeros(obj.output_size, nRirs, nBands);
            tAdjustFactors = zeros(1, nRirs, nBands);
            nAdjustFactors = zeros(1, nRirs, nBands);

            % Extract decays
            schroederDecays = rir2decay(signal, obj.sample_rate, obj.filter_frequencies, true, true, true, includeResidualBands);
            
            rirIdx = 1;
            for bandIdx = 1:nBands
                % Do backwards integration and remove trailing zeroes
                thisDecay = schroederDecays(:, bandIdx);
                                
                % Convert to dB and clamp at -140dB
                thisDecay = pow2db(thisDecay);
                thisLength = find(thisDecay < -140, 1);
                if ~isempty(thisLength)
                    thisDecay = thisDecay(1:thisLength);
                end
                
                % Calculate adjustment factors for t and n predictions
                tAdjustFactors(:, rirIdx, bandIdx) = 10/(length(thisDecay)/obj.sample_rate);
                nAdjustFactors(:, rirIdx, bandIdx) = length(thisDecay) / 2400;
                
                % Discard last 5%
                thisDecay = DecayFitNetToolbox.discardLast5(thisDecay);

                % Downsample to obj.output_size (default = 2400) samples
                dsFactor = floor(length(thisDecay)/obj.output_size);
                thisDecay_ds = downsample(thisDecay, dsFactor);
                edcs(1:obj.output_size, rirIdx, bandIdx) = thisDecay_ds(1:obj.output_size);

                tmp = 2 * edcs(1:obj.output_size, rirIdx, bandIdx) ./ obj.input_transform{'edcs_db_normfactor'};
                edcs(1:obj.output_size, rirIdx, bandIdx) = tmp + 1;
            end
            
            edcs = squeeze(edcs);
            scaleAdjustFactors.tAdjust = squeeze(tAdjustFactors);
            scaleAdjustFactors.nAdjust = squeeze(nAdjustFactors);
        end
        
        function [t_prediction, a_prediction, n_prediction] = estimate_parameters(obj, rir, do_preprocess, do_scale_adjustment, includeResidualBands)
            if nargin < 3
                do_preprocess = true;
            end
            if nargin < 4
                do_scale_adjustment = false;
            end
            if nargin < 5
                includeResidualBands = false;
            end
            
            if do_preprocess
                [edcs, scaleAdjustFactors] = obj.preprocess(rir, includeResidualBands);
            end
            
            % Forward pass of the DecayFitNet
            net = str2func([obj.networkName, 'model']);
            [t_prediction, a_prediction, n_prediction, n_slopes_probabilities] = net(edcs, obj.onnx_model, 'InputDataPermutation', [2,1]);
            
            if do_scale_adjustment
                [t_prediction, a_prediction, n_prediction] = obj.postprocess_parameters(t_prediction, a_prediction, n_prediction, n_slopes_probabilities, true, scaleAdjustFactors);
            end
            
        end
        
        function [t_prediction, a_prediction, n_prediction, n_slopes_prediction] = postprocess_parameters(obj, t_prediction, a_prediction, n_prediction, n_slopes_probabilities, sort_values, scaleAdjustFactors)
        %% Process the estimated t, a, and n parameters (output of the decayfitnet) to meaningful values
            if ~exist('sort_values', 'var')
                sort_values = true;
            end

            % Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
            n_prediction = min(max(n_prediction, -32), 32);

            % Go from noise exponent to noise value
            n_prediction = 10 .^ n_prediction;

            % In nSlope inference mode: Get a binary mask to only use the number of slopes that were predicted, zero others
            if obj.nSlopes == 0
                [~, n_slopes_prediction] = max(n_slopes_probabilities, [], 1);
                tmp = repmat(linspace(1,3,3), [size(n_slopes_prediction,2), 1])';
                mask = tmp <= repmat(n_slopes_prediction, [3,1]);
                a_prediction(~mask) = 0;
            end

            t_prediction = t_prediction ./ scaleAdjustFactors.tAdjust.';
            n_prediction = n_prediction ./ scaleAdjustFactors.nAdjust.';

           if sort_values
                % Sort T and A values:
                
                % 1) only in nSlope inference mode: assign nans to sort the inactive slopes to the end
                if obj.nSlopes == 0
                    t_prediction(~mask) = NaN;
                    a_prediction(~mask) = NaN;
                end

                % 2) sort
                [t_prediction, sort_idxs] = sort(t_prediction, 1);
                for batchIdx = 1: size(a_prediction, 2)
                    a_this_batch = a_prediction(:, batchIdx);
                    a_prediction(:, batchIdx) = a_this_batch(sort_idxs(:, batchIdx));
                end
                
                % 3) only in nSlope inference mode: set nans to zero again
                if obj.nSlopes == 0
                    t_prediction(isnan(t_prediction)) = 0;  
                    a_prediction(isnan(a_prediction)) = 0; 
                end
           end
        end
    end
    
    methods(Static)    
        function edc = generate_synthetic_edcs(T, A, noiseLevel, t)
            %% Generates an EDC from the estimated parameters.  
            assert(size(T, 2) == size(A, 2) && size(T, 2) == size(noiseLevel, 2), 'Wrong size in the input (different batch size in T, A, N)')
            [nSlopes, batchSize] = size(T);
            nSamples = length(t);
            
            % Permute to [batch_size, n_slopes], for consistency with the toolbox.
            T = permute(T, [2,1]);
            A = permute(A, [2,1]);
            noiseLevel = permute(noiseLevel, [2,1]);
            
            % Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
            tau_vals = -log(1e-6) ./ T;
            
            % Repeat values such that end result will be (sampled_idx, batch_size, n_slopes)
            %t_rep = repmat(t, [1, size(T, 2), size(T, 1)]);
            %tau_vals_rep = repmat(permute(tau_vals, [2,1]), [size(t, 1), 1, 1 ]);
            % Repeat values such that end result will be (batch_size, n_slopes, sampled_idx)
            t_rep = repmat(permute(t, [1, 3, 2]), [batchSize, nSlopes, 1]);
            tau_vals_rep = repmat(permute(tau_vals, [1,2,3]), [1, 1, nSamples]);
            assert(all(size(t_rep) == size(tau_vals_rep)), 'Wrong size in tau')
            
            % Calculate exponentials from decay rates
            time_vals = -t_rep .* tau_vals_rep;
            exponentials = exp(time_vals);
            
            % Zero exponentials where T=A=0
            for batchIdx = 1:batchSize
                zeroT = (T(batchIdx, :) == 0);
                exponentials(batchIdx, zeroT, :) = 0;
            end
            
            % Offset is required to make last value of EDC be correct
            exp_offset = repmat(exponentials(:, :, end), [1, 1, nSamples]);
            
            % Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
            A_rep = repmat(A, [1, 1, nSamples]);
            
            % Multiply exponentials with their amplitudes and sum all exponentials together
            edcs = A_rep .* (exponentials - exp_offset);
            edc = reshape(sum(edcs, 2), batchSize, nSamples);

            % Add noise
            noise = noiseLevel .* linspace(nSamples, 1, nSamples);
            edc = edc + noise;
        end
        
        function output = discardLast5(signal)
            % Discard last 5% samples of a signal
            assert(size(signal,1) > size(signal,2), 'The signal should be in [timesteps, channels]')
            last5 = round(0.95 * size(signal,1));
            
            output = zeros(last5, size(signal,2));
            for channel = [1 : size(signal,2)]
                tmp = signal(:,channel);
                tmp(last5+1:end) = [];
                output(:,channel) = tmp;
            end
        end
    end
    
end
