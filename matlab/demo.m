close all; clear variables; clc; lastwarn('');
%% External dependencies

%% Parameters
audioPath = '../data/exampleRIRs';  %local path
rirFName = 'doubleslope_RIR_200cm.wav';

nSlopes = 0; % 0 = network/bayesian estimates number of active slopes; otherwise, if you know number of slopes before, put it here (max. 3)
cutFromEnd = 0;  % specify how much you want to cut away from the end of the RIR in seconds, for example if there are lots of trailing zeros or a fade-out winow

% Bayesian parameters
parameterRanges.tRange = [0.15, 3.75];
parameterRanges.aRange = [-4.5, 0]; % 10^aRange
parameterRanges.nRange = [-14, -2]; % 10^nRange
nIterations = 1000;

%% Load an impulse
[rir, fs] = audioread(fullfile(audioPath, rirFName));
channels = size(rir, 2);

% Keep only the first channel
if size(rir, 1) > 1 && length(size(rir)) > 1
    disp(size(rir))
    rir = rir(:,1);
end

% Delete potential fade-out windows
if cutFromEnd > 0
    rir = rir(1:end + round(-cutFromEnd*fs), :);
end

fprintf('The impulse has %d channels (before selecting the first one).\n', channels)
fprintf('The impulse has %d timesteps at %d kHz sampling rate = %f seconds.\n', size(rir,1), fs, size(rir,1) / fs)
%sound(rir, fs)

% rir = [rir; zeros(24000, 1)];

%% Load model and estimate parameters
net = DecayFitNetToolbox(nSlopes, fs);
[tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet, normVals_decayFitNet] = net.estimateParameters(rir);
disp('==== DecayFitNet: Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(tVals_decayfitnet)
disp('==== DecayFitNet: Estimated A values (linear scale, A=0 indicates an inactive slope): ====') 
disp(aVals_decayfitnet)
disp('==== DecayFitNet: Estimated N values (linear scale): ====') 
disp(nVals_decayFitNet)

% Estimate true EDC
trueEDCs = rir2decay(rir, fs, [125, 250, 500, 1000, 2000, 4000], true, true, true).'; % doBackwardsInt=true, analyseFullRIR=true, normalize=true
timeAxis = linspace(0, (size(trueEDCs, 2) - 1) / fs, size(trueEDCs, 2) );
estimatedEDCs_decayfitnet = generateSyntheticEDCs(tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet, timeAxis);

f = figure;
f.Position = [150, 200, 1600, 600];
subplot(1, 2, 1);
hold on;
cmap = parula(size(trueEDCs, 1));
legendStr = {'Measured EDC, 125 Hz', 'DecayFitNet fit, 125 Hz', ...
    'Measured EDC, 250 Hz', 'DecayFitNet fit, 250 Hz',...
    'Measured EDC, 500 Hz', 'DecayFitNet fit, 500 Hz',...
    'Measured EDC, 1 kHz', 'DecayFitNet fit, 1 kHz',...
    'Measured EDC, 2 kHz', 'DecayFitNet fit, 2 kHz',...
    'Measured EDC, 4 kHz', 'DecayFitNet fit, 4 kHz'};
L = round(0.95*size(trueEDCs, 2)); % discard last 5 percent
allMSE = zeros(size(trueEDCs, 1), 1);
for bandIdx=1:size(trueEDCs, 1)
    plot(timeAxis(1:L), pow2db(trueEDCs(bandIdx, 1:L)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '-');
    plot(timeAxis(1:L), pow2db(estimatedEDCs_decayfitnet(bandIdx, 1:L)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '--');
    ylim([-60, 0]);
    
    allMSE(bandIdx) = mseLoss(pow2db(trueEDCs(bandIdx, 1:L)), pow2db(estimatedEDCs_decayfitnet(bandIdx, 1:L)));
end
legend(legendStr, 'Location', 'EastOutside');
title('DecayFitNet: Measured vs. estimated EDC fits');
drawnow;

fprintf('==== Average MSE between input EDCs and estimated fits: %.02f ====\n', mean(allMSE));
fprintf('MSE between input EDC and estimated fit for different frequency bands:\n 125 Hz: %.02f, 250 Hz: %.02f, 500 Hz: %.02f, 1 kHz: %.02f, 2 kHz: %.02f, 4 kHz: %.02f\n', allMSE(:));

% DecayFitNet can also be directly applied to the EDC/EDF
inputIsEDC = true;
[tVals_decayfitnet2, aVals_decayfitnet2, nVals_decayfitnet2, normVals_decayfitnet2] = net.estimateParameters(trueEDCs, inputIsEDC);
estimatedEDCs_decayfitnet2 = generateSyntheticEDCs(tVals_decayfitnet2, aVals_decayfitnet2, nVals_decayfitnet2, timeAxis);

%% Use Bayesian decay analysis with slice sampling
bda = BayesianDecayAnalysis(nSlopes, fs, parameterRanges, nIterations);
[tVals_bayesian, aVals_bayesian, nVals_bayesian, normVals_bayesian] = bda.estimateParameters(rir);

disp('==== Bayesian analysis: Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(tVals_bayesian)
disp('==== Bayesian analysis: Estimated A values (linear scale, A=0 indicates an inactive slope): ====') 
disp(aVals_bayesian)
disp('==== Bayesian analysis: Estimated N values (linear scale): ====') 
disp(nVals_bayesian)

estimatedEDCs_bayesian = generateSyntheticEDCs(tVals_bayesian, aVals_bayesian, nVals_bayesian, timeAxis);

subplot(1, 2, 2);
hold on;
cmap = parula(size(trueEDCs, 1));
legendStr = {'Measured EDC, 125 Hz', 'Bayesian fit, 125 Hz', ...
    'Measured EDC, 250 Hz', 'Bayesian fit, 250 Hz',...
    'Measured EDC, 500 Hz', 'Bayesian fit, 500 Hz',...
    'Measured EDC, 1 kHz', 'Bayesian fit, 1 kHz',...
    'Measured EDC, 2 kHz', 'Bayesian fit, 2 kHz',...
    'Measured EDC, 4 kHz', 'Bayesian fit, 4 kHz'};
L = round(0.95*size(trueEDCs, 2)); % discard last 5 percent
allMSE = zeros(size(trueEDCs, 1), 1);
for bandIdx=1:size(trueEDCs, 1)
    plot(timeAxis(1:L), pow2db(trueEDCs(bandIdx, 1:L)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '-');
    plot(timeAxis(1:L), pow2db(estimatedEDCs_bayesian(bandIdx, 1:L)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '--');
    ylim([-60, 0]);
    
    allMSE(bandIdx) = mseLoss(pow2db(trueEDCs(bandIdx, 1:L)), pow2db(estimatedEDCs_bayesian(bandIdx, 1:L)));
end
legend(legendStr, 'Location', 'EastOutside');
title('Bayesian decay analysis: Measured vs. estimated EDC fits');

% Bayesian analysis can also be directly applied to the EDC/EDF
inputIsEDC = true;
[tVals_bayesian2, aVals_bayesian2, nVals_bayesian2, normVals_bayesian2] = bda.estimateParameters(trueEDCs, inputIsEDC);
estimatedEDCs_bayesian2 = generateSyntheticEDCs(tVals_bayesian2, aVals_bayesian2, nVals_bayesian2, timeAxis);

% Bayesian analysis can use either the evidence or the IBIC for the model
% order selection. The evidence-based analysis leverages the full potential
% of the slice sampling algorithm, while the IBIC might be faster in this
% implementation.
inputIsEDC = false;
modelEstimationMode = 'ibic'; % default is 'evidence'
[tVals_bayesian3, aVals_bayesian3, nVals_bayesian3, normVals_bayesian3] = bda.estimateParameters(rir, inputIsEDC, modelEstimationMode);
estimatedEDCs_bayesian3 = generateSyntheticEDCs(tVals_bayesian3, aVals_bayesian3, nVals_bayesian3, timeAxis);
