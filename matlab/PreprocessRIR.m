classdef PreprocessRIR < handle
    
    properties
        sampleRate
        
        % filter frequency = 0 will give a lowpass band below the lowest
        % octave band, filter frequency = sample rate / 2 will give the
        % highpass band above the highest octave band
        filterFrequencies
        
        inputTransform
        outputSize
    end
    
    methods
        function obj = PreprocessRIR(inputTransform, sampleRate, filterFrequencies, outputSize)
            if nargin < 2
                sampleRate = 48000;
            end
            if nargin < 3 || isempty(filterFrequencies)
                filterFrequencies = [125, 250, 500, 1000, 2000, 4000];
            end
            if nargin < 4
                outputSize = 100;
            end
            
            obj.inputTransform = inputTransform;
            obj.sampleRate = sampleRate;
            obj.filterFrequencies = filterFrequencies;
            obj.outputSize = outputSize;
        end
        
        function set.filterFrequencies(obj, filterFrequencies)
            assert(~any(filterFrequencies < 0) && ~any(filterFrequencies > obj.sampleRate/2), 'Invalid band frequency. Octave band frequencies must be bigger than 0 and smaller than fs/2. Set frequency=0 for a lowpass band and frequency=fs/2 for a highpass band.');
            filterFrequencies = sort(filterFrequencies);
            obj.filterFrequencies = filterFrequencies;
        end
        
        function [edcs, timeAxis_ds, normVals, scaleAdjustFactors] = preprocess(obj, rir)
            % Extract decays: Do backwards integration
            [schroederDecays, normVals] = rir2decay(rir, obj.sampleRate, obj.filterFrequencies, true, true, true); % doBackwardsInt, analyseFullRIR, normalize
            
            % Init arrays
            nBands = length(obj.filterFrequencies);
            edcs = zeros(obj.outputSize, nBands);
            % -- DecayFitNet: T value predictions have to be adjusted for the time-scale conversion
            tAdjustFactors = ones(1, nBands);
            % -- N values have to be adjusted for downsampling
            nAdjustFactors = ones(1, nBands);
            
            % Go through all bands and process
            for bandIdx = 1:nBands
                thisDecay = schroederDecays(:, bandIdx);
                                
                % Convert to dB
                thisDecay_db = pow2db(thisDecay+eps);
                                                
                % N values have to be adjusted for downsampling
                nAdjustFactors(:, bandIdx) = length(thisDecay_db) / obj.outputSize;
                
                % DecayFitNet only: 
                if ~isempty(obj.inputTransform)
                    % T value predictions have to be adjusted for the time-scale conversion
                    tAdjustFactors(:, bandIdx) = 10/(length(thisDecay_db)/obj.sampleRate);
                    
                    % Discard last 5%
                    thisDecay_db = DecayFitNetToolbox.discardLast5(thisDecay_db);
                end
                
                % Resample to obj.outputSize (default = 100) samples
                thisDecay_db_ds = resample(thisDecay_db, obj.outputSize, length(thisDecay_db), 0, 5);
                
                % DecayFitNet only: Apply input transform
                if ~isempty(obj.inputTransform)
                    tmp = 2 * thisDecay_db_ds ./ obj.inputTransform{'edcs_db_normfactor'};
                    tmp = tmp + 1;
                    thisDecay_db_ds = tmp;
                end
                
                edcs(:, bandIdx) = thisDecay_db_ds;
            end
            
            scaleAdjustFactors.tAdjust = tAdjustFactors;
            scaleAdjustFactors.nAdjust = nAdjustFactors;
            
            timeAxis_ds = linspace(0, (length(schroederDecays)-1)/obj.sampleRate, obj.outputSize).';
        end
    end
end