function [tVals, aVals, nVals] = bayesianDecayAnalysis_estimateParameters(edc_db, modelOrders, tCandidates, aCandidates, nCandidates, nIterations, t)
assert(length(edc_db) <= 1000, 'Please downsample the EDC prior to Bayesian analysis. 100 sample EDCs work well.');        
assert(length(edc_db) == length(t), 'Time axis does not match EDC.');

allMaxLikelihoodParams = cell(length(modelOrders), 1);
allBICs = zeros(length(modelOrders), 1);

for thisModelOrderIdx = 1:length(modelOrders)
    thisModelOrder = modelOrders(thisModelOrderIdx);
    [testedParameters_model, likelihoods_model] = bayesianDecayAnalysis_sliceSampling(edc_db, thisModelOrder, tCandidates, aCandidates, nCandidates, nIterations, t);
    [maxLikelihood_model, maxLikelihoodIdx_model] = max(likelihoods_model, [], 'all', 'linear');
    allMaxLikelihoodParams{thisModelOrderIdx} = testedParameters_model(maxLikelihoodIdx_model, :);
    allBICs(thisModelOrderIdx) = 2*log(maxLikelihood_model) - (2*thisModelOrder + 1)*log(length(t)); % Eq. (15)
end

% Evaluate BIC
[~, bestModelOrderIdx] = max(allBICs);
bestModelOrder = modelOrders(bestModelOrderIdx);
bestModelParams = allMaxLikelihoodParams{bestModelOrderIdx};

tVals = tCandidates(bestModelParams(1:bestModelOrder)).';
aVals = aCandidates(bestModelParams(bestModelOrder+1:2*bestModelOrder)).';
nVals = nCandidates(bestModelParams(2*bestModelOrder+1:end)).';
end