function [tVals, aVals, nVals] = postprocessDecayParameters(tVals, aVals, nVals, scaleAdjustFactors, nSlopeEstimationMode)
    % Process the estimated t, a, and n parameters

    % Adjust for downsampling
    nVals = nVals ./ scaleAdjustFactors.nAdjust;
    
    % Only relevant for DecayFitNet: T value predictions have to be adjusted for the time-scale conversion
    tVals = tVals ./ scaleAdjustFactors.tAdjust; % factors are 1 for Bayesian

    % In nSlope estimation mode: Get a binary mask to only use the 
    % number of slopes that were predicted, others are zeroed
    if nSlopeEstimationMode
        mask = (aVals == 0);

        % Assign NaN instead of zero for now, to sort inactive
        % slopes to the end
        tVals(mask) = NaN;
        aVals(mask) = NaN;
    end

    % Sort T and A values
    [tVals, sortIdxs] = sort(tVals, 1);
    for bandIdx = 1:size(tVals, 2)
        aThisBand = aVals(:, bandIdx);
        aVals(:, bandIdx) = aThisBand(sortIdxs(:, bandIdx));
    end

    % 3) only in nSlope estimation mode: set nans to zero again
    if nSlopeEstimationMode
        tVals(isnan(tVals)) = 0;  
        aVals(isnan(aVals)) = 0; 
    end
end