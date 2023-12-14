% DecayFitNet offers a mode, in which the number of active decay
% slopes is determined directly by the network [i.e., by using
% DecayFitNetToolbox(0)]. This usually works well and yields good model
% fits, but the model order tends to be overestimated. 
%
% We found an alternative way of estimating the model order. First, we
% estimate the model parameters with multiple different DecayFitNets, each
% of which predicts a predefined number of slopes [i.e., by using
% DecayFitNetToolbox(1), DecayFitNetToolbox(2), and DecayFitNetToolbox(3),
% for 1, 2, and 3 slopes, respectively]. Second, we choose the model with 
% the lowest order that still describes the data sufficiently well, using 
% the Inverse Bayesian Information Criterion (IBIC). 
%
% This functions implements the above IBIC-based alternative approach for 
% the full decay analysis (parameters + model order). 
%
% Georg Götz, October 2023, Aalto University, georg.gotz@aalto.fi
classdef ibicBasedDecayFitNet < handle
    properties (SetAccess = private)
        version = '0.1.2'
        outputSize = 100  % Timesteps of resampled RIRs
        maxOrder
        sampleRate
        nets
        filterFrequency
    end

    methods
        function obj = ibicBasedDecayFitNet(maxOrder, sampleRate, filterFrequency)
            obj.maxOrder = maxOrder;
            obj.sampleRate = sampleRate;

            assert(numel(filterFrequency)==1, 'Multiple frequency bands are not supported yet.');
            obj.filterFrequency = filterFrequency;
            
            nets = cell(obj.maxOrder, 1);
            for oIdx=1:maxOrder
                nets{oIdx} = DecayFitNetToolbox(oIdx, obj.sampleRate, filterFrequency);
            end
            obj.nets = nets;
        end

        function [tPrediction, aPrediction, nPrediction, normVal] = estimateParameters(obj, input, inputIsEDF)
            % Initialize time axis and calculate true EDF from input (required for
            % IBIC calculations)
            timeAxis = linspace(0, (size(input, 1)-1)/obj.sampleRate, size(input, 1));
            if inputIsEDF
                edf = input;
            else
                edf = rir2decay(input, obj.sampleRate, obj.filterFrequency, true, true, true); % doBackwardsInt, analyseFullRIR, normalize
            end
            edf_db = pow2db(edf);
            edf_db_ds = resample(edf_db, obj.outputSize, length(edf_db), 0, 5);
            
            % Init arrays for results of all networks
            allIBICs = zeros(obj.maxOrder, 1);
            allTVals = zeros(obj.maxOrder);
            allAVals = zeros(obj.maxOrder);
            allNVals = zeros(obj.maxOrder, 1);

            % Go through all orders, use exactlyNSlopes mode of DecayFitNet to predict
            % parameters, save them for later
            for oIdx=1:obj.maxOrder
                % Predict parameters using DecayFitNet in exavtlyNSlopes mode
                [tVals, aVals, nVal, normVal] = obj.nets{oIdx}.estimateParameters(input, inputIsEDF);
                allTVals(oIdx, 1:length(tVals)) = tVals;
                allAVals(oIdx, 1:length(aVals)) = aVals;
                allNVals(oIdx) = nVal;
                
                % Get model EDF from parameters
                fittedEDF = generateSyntheticEDCs(tVals, aVals, nVal, timeAxis).';
                fittedEDF_db_ds = resample(pow2db(fittedEDF), obj.outputSize, length(fittedEDF), 0, 5);
                
                % Calculate IBIC for this model fit
                exclLast5PercentIdx = round(0.95*obj.outputSize); % exclude last 5% of EDF samples
                thisSummedError = sum((fittedEDF_db_ds(1:exclLast5PercentIdx) - edf_db_ds(1:exclLast5PercentIdx)).^2);
                thisLikelihood = 0.5* gamma(exclLast5PercentIdx/2) * (pi*thisSummedError)^(-exclLast5PercentIdx/2);
                allIBICs(oIdx) = 2*log(thisLikelihood) - (2*oIdx + 1)*log(exclLast5PercentIdx);
            end
            
            % Determine best model by maximizing IBIC
            [~, idxBestFitIBIC] = max(allIBICs);
            tPrediction = allTVals(idxBestFitIBIC, 1:idxBestFitIBIC);
            aPrediction = allAVals(idxBestFitIBIC, 1:idxBestFitIBIC);
            nPrediction = allNVals(idxBestFitIBIC);
        end
    end
end
