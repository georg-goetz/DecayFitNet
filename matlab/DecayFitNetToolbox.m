classdef DecayFitNetToolbox < handle
    
    properties
        output_size = 2400  % Timesteps of downsampled RIRs
        filter_frequencies = [125, 250, 500, 1000, 2000, 4000]
        version = '0.0.3'
        sample_rate
        normalization
        PATH_ONNX
        onnx_model
        input_transform
    end
    
    methods
        function obj = DecayFitNetToolbox(sample_rate, normalization)
            if nargin < 1
                sample_rate = 48000;
            end
            if nargin < 2
                normalization = true;
            end
                
            obj.sample_rate = sample_rate;
            obj.normalization = normalization;
            
            % Load onnx model
            obj.PATH_ONNX = fullfile('..', 'model');
            
            % FAILS:
            % ONNX network with multiple outputs is not supported. Instead, use 'importONNXLayers' with 'ImportWeights' set to true.
            % obj.onnx_model = importONNXNetwork(fullfile(obj.PATH_ONNX, 'DecayFitNet.onnx'),'OutputLayerType','regression');
            
            % FAILS:
            % Warning: Unable to import some ONNX operators, because they are not supported. They have been replaced by placeholder layers. To find these layers, call the
            % function findPlaceholderLayers on the returned object.
            % obj.onnx_model = importONNXLayers(fullfile(obj.PATH_ONNX, 'DecayFitNet_v9.onnx'),'ImportWeights',true)  % Fails due to unsupported functions
            
            % FAILS :
            % The value of 'params' is invalid. It must satisfy the function: @(x)isa(x,'ONNXParameters').
            obj.onnx_model = importONNXFunction(fullfile(obj.PATH_ONNX, 'DecayFitNet_v9.onnx'), 'DecayFitNet_model')
            %[output, x66, x69, x72, state] = test_DecayFitNet(signal, '');
            
            fid = py.open(fullfile(obj.PATH_ONNX, 'input_transform_p2.pkl'),'rb');
            obj.input_transform = py.pickle.load(fid);
        end
        
        function signal = preprocess_old(obj, signal)
            % Resampling    
            p = size(signal, 1);
            q = obj.output_size;
            signal = resample(signal, q, p);
        end
        
        function edcs = preprocess(obj, signal)    
            nbands = length(obj.filter_frequencies);
            nRirs = 1;
            edcs = zeros(obj.output_size, nRirs, nbands);
            edcs_5dbCut = zeros(obj.output_size, nRirs, nbands);
            tAdjustFactors = zeros(1, nRirs, nbands);
            tAdjustFactors_5dbCut = zeros(1, nRirs, nbands);

            % Extract decays
            schroederDecays = rir2decay(signal, obj.sample_rate, obj.filter_frequencies, true, true);
            warning('Ignoring onset detection') % TODO: Fix this
            
            rirIdx = 1;
            for bandIdx = 1:nbands
                % Do backwards integration and remove trailing zeroes
                thisDecay = schroederDecays(:, bandIdx);
                thisLength = find(flipud(thisDecay), 1);
                thisDecay = thisDecay(1:end-thisLength+1);

                % Normalize to 1
                thisDecay = thisDecay / max(thisDecay);

                % Discard last 5%
                thisDecay = DecayFitNetToolbox.discarLast5(thisDecay);

                % Downsample to obj.output_size (default = 2400) samples
                tAdjustFactors(:, rirIdx, bandIdx) = 10/(length(thisDecay)/obj.sample_rate);
                thisDecay_ds = downsample(thisDecay, floor(length(thisDecay)/obj.output_size));
                edcs(1:obj.output_size, rirIdx, bandIdx) = thisDecay_ds(1:obj.output_size);
                
                % Cut at -140dB
                this_edc = pow2db(edcs(1:obj.output_size, rirIdx, bandIdx));
                thisLength = find(this_edc < -140, 1);
                if ~isempty(thisLength)
                    edcs(1:obj.output_size, rirIdx, bandIdx) = this_edc(1:thisLength);
                else
                    edcs(1:obj.output_size, rirIdx, bandIdx) = this_edc;
                end

                if obj.normalization
                    tmp = 2 * this_edc ./ obj.input_transform{'edcs_db_normfactor'};
                    edcs(1:obj.output_size, rirIdx, bandIdx) = tmp + 1;
                end
                
                % This is not needed (Legacy)
                % Go to dB and cut away everything above -5dB
                thisDecay_db = pow2db(thisDecay);
                below5dbIdx = find(thisDecay_db < -5, 1);
                thisDecay_5dbCut = thisDecay(below5dbIdx:end);
                thisDecay_5dbCut = thisDecay_5dbCut / max(thisDecay_5dbCut); % renormalize

                tAdjustFactors_5dbCut(:, rirIdx, bandIdx) = 10/(length(thisDecay_5dbCut)/obj.sample_rate);
                thisDecay_5dbCut_ds = downsample(thisDecay_5dbCut, ceil(length(thisDecay_5dbCut)/obj.output_size));
                edcs_5dbCut(1:length(thisDecay_5dbCut_ds), rirIdx, bandIdx) = thisDecay_5dbCut_ds;
            end
            
            edcs = squeeze(edcs);
            % Returns edcs
        end
        
        function [t_prediction, a_prediction, n_prediction] = estimate_parameters(obj, rir, do_preprocess, do_scale_adjustment)
            if nargin < 3
                do_preprocess = true;
            end
            if nargin < 4
                do_scale_adjustment = false;
            end
            
            % TODO: fix preprocessing
            if do_preprocess
                edcs = obj.preprocess(rir);
            end
            
            [t_prediction, a_prediction, n_prediction, n_slopes_probabilities, state] = DecayFitNet_model(edcs, obj.onnx_model, 'InputDataPermutation', [2,1]);
            % TODO: do postprocessing of parameters
            if do_scale_adjustment
                [t_prediction, a_prediction, n_prediction] = DecayFitNetToolbox.postprocess_parameters(t_prediction, a_prediction, n_prediction, n_slopes_probabilities, false);
            end
            
        end
    end
    
    methods(Static)
        function [t_prediction, a_prediction, n_prediction, n_slopes_prediction] = postprocess_parameters(t_prediction, a_prediction, n_prediction, n_slopes_probabilities, sort_values)
            %% Process the estimated t, a, and n parameters (output of the decayfitnet) to meaningful values
            if ~exist('sort_values', 'var')
                sort_values = true;
            end

            % Clamp noise to reasonable values to avoid numerical problems and go from exponent to actual noise value
            n_prediction = min(max(n_prediction, -32), 32);

            % Go from noise exponent to noise value
            n_prediction = n_prediction .^ 10;

            % Get a binary mask to only use the number of slopes that were predicted, zero others
            [~, n_slopes_prediction] = max(n_slopes_probabilities, [], 1);
            tmp = repmat(linspace(1,3,3), [size(n_slopes_prediction,2), 1])';
            mask = tmp <= repmat(n_slopes_prediction, [3,1]);
            a_prediction(~mask) = 0;

           if sort_values
                % Sort T and A values:
                % 1) assign nans to sort the inactive slopes to the end
                t_prediction(~mask) = NaN;
                a_prediction(~mask) = NaN;

                % 2) sort and set nans to zero again
                [t_prediction, sort_idxs] = sort(t_prediction);
                for batch_idx = [1: 1: size(a_prediction,1)]
                    a_this_batch = a_prediction(batch_idx, :);
                    a_prediction(batch_idx, :) = a_this_batch(sort_idxs(batch_idx));
                end
                t_prediction(isnan(t_prediction)) = 0;  
                a_prediction(isnan(a_prediction)) = 0; 
           end
        end
        
        function edc = generate_synthetic_edcs(T, A, noiseLevel, t)
            %% Generates an EDC from the estimated parameters.
            %if size(t,2) > size(t,1)
            %    t = permute(t, [2,1]);  %Timesteps first format
            %end
            
            % Permute to [batch_size, n_slopes], for consistency with the toolbox.
            T = permute(T, [2,1]);
            A = permute(A, [2,1]);
            noiseLevel = permute(noiseLevel, [2,1]);
            assert(size(T, 1) == size(A,1) && size(T,1) == size(noiseLevel, 1), 'Wrong size in the input')
                
            
            % Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
            tau_vals = -log(1e-6) ./ T;
            
            % Repeat values such that end result will be (sampled_idx, batch_size, n_slopes)
            %t_rep = repmat(t, [1, size(T, 2), size(T, 1)]);
            %tau_vals_rep = repmat(permute(tau_vals, [2,1]), [size(t, 1), 1, 1 ]);
            % Repeat values such that end result will be (batch_size, n_slopes, sampled_idx)
            t_rep = repmat(permute(t, [1, 3, 2]), [size(T, 1), size(T, 2),1]);
            tau_vals_rep = repmat(permute(tau_vals, [1,2,3]), [1, 1, size(t, 2) ]);
            assert(all(size(t_rep) == size(tau_vals_rep)), 'Wrong size in tau')
            
            % Calculate exponentials from decay rates
            time_vals = -t_rep .* tau_vals_rep;
            exponentials = exp(time_vals);
            
            % Offset is required to make last value of EDC be correct
            exp_offset = repmat(exponentials(:, :, end), [1,1, size(t, 2)]);
            
            % Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
            A_rep = repmat(A, [1, 1, size(t,2)]);
            
            % Multiply exponentials with their amplitudes and sum all exponentials together
            edcs = A_rep .* (exponentials - exp_offset);
            edc = squeeze(sum(edcs, 2));

            % Add noise
            noise = noiseLevel .* linspace(length(t), 1, length(t));
            edc = edc + noise;
            
        end
        
        function output = discarLast5(signal)
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
