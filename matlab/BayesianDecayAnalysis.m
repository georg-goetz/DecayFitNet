classdef BayesianDecayAnalysis < handle
    
    properties        
        nPointsPerDim = 100; % defines parameter space and slice window (increment in steps of 1)
        nIterations = 50;
        
        nSlopes
    end
    properties (SetAccess = private)
        version = '0.1.0'
        sampleRate
        outputSize = 100  % Timesteps of resampled RIRs
        preprocessing
        
        tRange
        aRange % 10^aRange
        nRange % 10^nRange
        tSpace
        aSpace
        nSpace
    end
    properties (Dependent)
        maxNSlopes
    end
    
    methods
        function obj = BayesianDecayAnalysis(nSlopes, sampleRate, parameterRanges, nIterations, filterFrequencies)
            if nargin < 1
                nSlopes = 0; % estimate number of slopes from data
            end
            if nargin < 2
                sampleRate = 48000;
            end
            if nargin < 3
                parameterRanges.tRange = [0.1, 3.5];
                parameterRanges.aRange = [-3, 0]; % 10^aRange
                parameterRanges.nRange = [-10, -2]; % 10^nRange
            end
            if nargin < 4
                nIterations = 50;
            end
            if nargin < 5
                filterFrequencies = [];
            end
                            
            obj.sampleRate = sampleRate;
            obj.nSlopes = nSlopes;
            
            obj.setParameterRanges(parameterRanges);
            obj.initParameterSpace();
            
            obj.nIterations = nIterations;
            
            % Init preprocessing
            obj.preprocessing = PreprocessRIR([], sampleRate, filterFrequencies, obj.outputSize);
            
        end
        
        function setFilterFrequencies(obj, filterFrequencies)
            obj.preprocessing.filterFrequencies = filterFrequencies;
        end
        
        function filterFrequencies = getFilterFrequencies(obj)
            filterFrequencies = obj.preprocessing.filterFrequencies;
        end
        
        function set.nSlopes(obj, nSlopes)
            assert(nSlopes <= 3, 'Maximum number of supported slopes: 3');
            obj.nSlopes = nSlopes;
        end
        
        function maxNSlopes = get.maxNSlopes(obj)
            if obj.nSlopes == 0
                maxNSlopes = 3;
            else
                maxNSlopes = obj.nSlopes;
            end
        end
        
        function set.nPointsPerDim(obj, nPointsPerDim)
            obj.nPointsPerDim = nPointsPerDim;
            obj.initParameterSpace();
        end
        
        function setParameterRanges(obj, parameterRanges)
            assert(isfield(parameterRanges, 'tRange') ...
                && isfield(parameterRanges, 'aRange') ...
                && isfield(parameterRanges, 'nRange'), ...
                'parameterRanges must be a struct that has the fields tRange, aRange, and nRange');
            tRange_tmp = parameterRanges.tRange;
            aRange_tmp = parameterRanges.aRange;
            nRange_tmp = parameterRanges.nRange;

            assert(length(tRange_tmp)==2 && length(aRange_tmp)==2 && length(nRange_tmp)==2, ...
                'tRange, aRange, and nRange must be given as an array [minVal, maxVal]');
            assert(diff(tRange_tmp) > 0 && diff(aRange_tmp) > 0 && diff(nRange_tmp) > 0, ...
                'First value in tRange, aRange, and nRange must be smaller than second value, i.e., array should be [minVal, maxVal]');
            
            obj.tRange = tRange_tmp;
            obj.aRange = aRange_tmp;
            obj.nRange = nRange_tmp;
        end
        
        function initParameterSpace(obj)
            obj.tSpace = linspace(obj.tRange(1), obj.tRange(2), obj.nPointsPerDim);
            obj.aSpace = logspace(obj.aRange(1), obj.aRange(2), obj.nPointsPerDim);
            obj.nSpace = logspace(obj.nRange(1), obj.nRange(2), obj.nPointsPerDim);
        end
        
        function [tVals, aVals, nVals] = estimateParameters(obj, rir)
            [edcs, timeAxis_ds, normVals, scaleAdjustFactors] = obj.preprocessing.preprocess(rir);
 
            % in nSlope estimation mode: max number of slopes is hard-coded
            % in get method (3 is usually enough)
            nBands = size(edcs, 2);
            tVals = zeros(obj.maxNSlopes, nBands);
            aVals = zeros(obj.maxNSlopes, nBands);
            nVals = zeros(1, nBands);
            
            % go over all frequency bands
            for bandIdx=1:nBands
                [tPrediction, aPrediction, nPrediction] = obj.estimation(edcs(:, bandIdx), timeAxis_ds);
                nSlopesPrediction = size(tPrediction, 1);
                tVals(1:nSlopesPrediction, bandIdx) = tPrediction;
                aVals(1:nSlopesPrediction, bandIdx) = aPrediction;
                nVals(1, bandIdx) = nPrediction;
            end
            
            % Postprocess parameters: scale adjustment, zero inactive
            % slopes, and sort
            nSlopeEstimationMode = (obj.nSlopes == 0);
            [tVals, aVals, nVals] = postprocessDecayParameters(tVals, aVals, nVals, scaleAdjustFactors, nSlopeEstimationMode);
        end
                        
        function [tVals, aVals, nVal] = estimation(obj, edc_db, timeAxis)
            % Following Xiang, N., Goggans, P., Jasa, T. & Robinson, P. "Bayesian characterization of multiple-slope sound energy decays in coupled-volume systems." J Acoust Soc Am 129, 741–752 (2011).
            
            assert(length(edc_db) == length(timeAxis), 'Time axis does not match EDC.');
            
            if obj.nSlopes == 0
                modelOrders = [1, 2, 3]; % estimate number of slopes according to BIC
            else
                modelOrders = obj.nSlopes;
            end
            
            allMaxLikelihoodParams = cell(length(modelOrders), 1);
            allBICs = zeros(length(modelOrders), 1);
            
            % go through all possible model orders and find max likelihood
            for thisModelOrderIdx = 1:length(modelOrders)
                thisModelOrder = modelOrders(thisModelOrderIdx);
                
                % Do slice sampling to determine likelihoods for multiple
                % parameter combinations
                [testedParameters, likelihoods] = obj.sliceSampling(edc_db, thisModelOrder, timeAxis);
                
                % Find maximum likelihood and corresponding parameter
                % combination
                [maxLikelihood, maxLikelihoodIdx] = max(likelihoods);
                allMaxLikelihoodParams{thisModelOrderIdx} = testedParameters(maxLikelihoodIdx, :);
                
                % Determine BIC for this maximum likelihood and model
                % order: this is used to estimate the model order if
                % desired
                allBICs(thisModelOrderIdx) = 2*log(maxLikelihood) - (2*thisModelOrder + 1)*log(length(timeAxis)); % Eq. (15)
            end

            % Find model with highest BIC: model that describes data best
            % with most concise model
            [~, bestModelOrderIdx] = max(allBICs);
            bestModelOrder = modelOrders(bestModelOrderIdx);
            bestModelParams = allMaxLikelihoodParams{bestModelOrderIdx};

            tVals = obj.tSpace(bestModelParams(1:bestModelOrder)).';
            aVals = obj.aSpace(bestModelParams(bestModelOrder+1:2*bestModelOrder)).';
            nVal = obj.nSpace(bestModelParams(2*bestModelOrder+1:end)).';
        end
        
        function [testedParameters, likelihoods] = sliceSampling(obj, edc_db, modelOrder, timeAxis)
            % Following Jasa, T. & Xiang, N. "Efficient estimation of decay parameters in acoustically coupled-spaces using slice sampling." J Acoust Soc Am 126, 1269–1279 (2009).

            assert(length(obj.tSpace)==length(obj.aSpace) && length(obj.tSpace)==length(obj.nSpace), 'There must be an equal number of T, A, and N values in the parameter space.');
            nParameters = modelOrder*2+1;

            testedParameters = zeros(obj.nIterations, nParameters);
            likelihoods = zeros(obj.nIterations, 1);
            
            % randomly draw first parameter values (indices)
            x0 = randi(obj.nPointsPerDim, nParameters, 1);
            
            % evaluate likelihood for these parameters, and multiply with a
            % random number between 0...1 to determine a likelihood threshold
            y0 = rand * obj.evaluateLikelihood(edc_db, obj.tSpace(x0(1:modelOrder)).', obj.aSpace(x0(modelOrder+1:2*modelOrder)).', obj.nSpace(x0(2*modelOrder+1)), timeAxis);
            
            % start to iterate
            for sampleIdx=1:obj.nIterations
                % determine which variable is varied: variables are varied 
                % in turn
                paramIdx = mod(sampleIdx-1, nParameters) + 1; 
                
                % =======
                % 1: Vary parameter until slice is established: the slice 
                % is the region, for which the likelihood is higher than 
                % the previously found likelihood threshold y0

                % Find left edge of slice: decrease parameter value until
                % likelihood is below threshold
                thisX0Left = x0;
                while(thisX0Left(paramIdx)>1)
                    thisX0Left(paramIdx) = thisX0Left(paramIdx) - 1;
                    thisY0Left = obj.evaluateLikelihood(edc_db, obj.tSpace(thisX0Left(1:modelOrder)).', obj.aSpace(thisX0Left(modelOrder+1:2*modelOrder)).', obj.nSpace(thisX0Left(2*modelOrder+1)), timeAxis);

                    if(thisY0Left < y0)
                        break;
                    end
                end

                % Find right edge of slice: increase parameter value until
                % likelihood is below threshold
                thisX0Right = x0;
                while(thisX0Right(paramIdx)<obj.nPointsPerDim-1)
                    thisX0Right(paramIdx) = thisX0Right(paramIdx) + 1;
                    thisY0Right = obj.evaluateLikelihood(edc_db, obj.tSpace(thisX0Right(1:modelOrder)).', obj.aSpace(thisX0Right(modelOrder+1:2*modelOrder)).', obj.nSpace(thisX0Right(2*modelOrder+1)), timeAxis);

                    if(thisY0Right < y0)
                        break;
                    end
                end
                
                % =======
                % 2: Draw new parameter value from the slice, to find a new 
                % and higher likelihood (and threshold)
                while(true)
                    % copy old parameter values
                    x1 = x0;
                    
                    % randomly draw varied parameter (index) from the slice
                    % +1 -1 to avoid randi(0), which gives error
                    x1(paramIdx) = randi(thisX0Right(paramIdx)-thisX0Left(paramIdx) + 1) + thisX0Left(paramIdx) - 1;
                    
                    % evaluate likelihood for drawn parameter
                    y1 = obj.evaluateLikelihood(edc_db, obj.tSpace(x1(1:modelOrder)).', obj.aSpace(x1(modelOrder+1:2*modelOrder)).', obj.nSpace(x1(2*modelOrder+1)), timeAxis);

                    if(y1>y0)
                        % higher likelihood found, continue with next 
                        % iteration step
                        break;
                    else
                        % drawn value is not actually in slice due to 
                        % sampling rate error (slice is established as 
                        % multiples of a step), therefore adapt slice 
                        % interval
                        
                        % find out which edge of the slice was wrong
                        if vecnorm(x1-thisX0Left) < vecnorm(x1-thisX0Right)
                            thisX0Left = x1;
                        else
                            thisX0Right = x1;
                        end
                    end
                end
                
                % Save tested parameters and likelihood of this iteration
                testedParameters(sampleIdx, :) = x1;
                likelihoods(sampleIdx) = y1;

                % prepare for next iteration: new threshold
                y0 = rand*y1;
                x0 = x1;
            end
        end
    end
    methods(Static)
        function likelihood = evaluateLikelihood(edc_db, T, A, N, timeAxis)
            % Following Xiang, N., Goggans, P., Jasa, T. & Robinson, P. "Bayesian characterization of multiple-slope sound energy decays in coupled-volume systems." J Acoust Soc Am 129, 741–752 (2011).

            % Calculate model EDC
            modelEDC = decayModel(T, A, N, timeAxis, true); % compensateULI=true
            modelEDC = sum(modelEDC, 2);

            % Convert to dB (true EDC is already db)
            modelEDC = 10*log10(modelEDC);

            % Exclude last 5% from likelihood (makes analysis more robust
            % to edge effects caused by octave filtering)
            K = round(0.95 * length(timeAxis)); % Eq. (1), 
            
            % Evaluate Likelihood
            E = 0.5 * sum((edc_db(1:K) - modelEDC(1:K)).^2); % Eq. (13)
            likelihood = 0.5 * gamma(K/2) * (2*pi*E)^(-K/2); % Eq. (12)
        end
    end
end