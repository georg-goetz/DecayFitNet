function [tVals, aVals, nVals] = bayesianDecayAnalysis_estimateParameters(edc_db, modelOrders, tCandidates, aCandidates, nCandidates, nIterations, t)
assert(length(edc_db) <= 1000, 'Please downsample the EDC prior to Bayesian analysis. 100 sample EDCs work well.');        
assert(length(edc_db) == length(t), 'Time axis does not match EDC.');

allMaxLikelihoodParams = cell(length(modelOrders), 1);
allBICs = zeros(length(modelOrders), 1);

for thisModelOrder = modelOrders(1:end)
    [testedParameters_model, likelihoods_model] = bayesianDecayAnalysis_sliceSampling(edc_db, thisModelOrder, tCandidates, aCandidates, nCandidates, nIterations, t);
    [maxLikelihood_model, maxLikelihoodIdx_model] = max(likelihoods_model, [], 'all', 'linear');
    allMaxLikelihoodParams{thisModelOrder} = testedParameters_model(maxLikelihoodIdx_model, :);
    allBICs(thisModelOrder) = 2*log(maxLikelihood_model) - (2*thisModelOrder + 1)*log(length(t)); % Eq. (15)
end

% Evaluate BIC
[~, bestModelOrder] = max(allBICs);
bestModelParams = allMaxLikelihoodParams{bestModelOrder};

tVals = tCandidates(bestModelParams(1:bestModelOrder)).';
aVals = aCandidates(bestModelParams(bestModelOrder+1:2*bestModelOrder)).';
nVals = nCandidates(bestModelParams(2*bestModelOrder+1:end)).';
end