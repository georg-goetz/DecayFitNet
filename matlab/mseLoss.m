function mse = mseLoss(trueVals, predictedVals)
% calculates the mean squared error between two arrays of values. Can also
% be replaced by MATLAB's internal function immse (Image Processing
% Toolbox)
% 
% Inputs:
%   trueVals          - array of ground-truth values
%   predictedVals     - array of predicted values, must be same size as
%                       trueVals
%
% Outputs:
%   mse   - mean squared error between both arrays [1, 1]
%
% (c) Georg Götz, Aalto University, 2021

assert(all(size(trueVals) == size(predictedVals)), 'Arrays of true values and predicted values must have the same dimensions');
mse = mean((trueVals-predictedVals).^2, 'all');

end