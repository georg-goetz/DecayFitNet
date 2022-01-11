close all; clear variables; clc; lastwarn('');
%% External dependencies

%% Parameters
audioPath = '../model';  %local path
rirFName = '0001_1_sh_rirs.wav';  % First measurement of Motus dataset

fadeoutLength = 0;  % in secs
nSlopes = 0; % 0 = estimate number of active slopes

% Bayesian parameters
parameterRanges.tRange = [0.1, 3.5];
parameterRanges.aRange = [-3, 0]; % 10^aRange
parameterRanges.nRange = [-10, -2]; % 10^nRange
nIterations = 100;

%% Load an impulse
[rir, fs] = audioread(fullfile(audioPath, rirFName));
channels = size(rir, 2);

% Keep only the first channel
if size(rir, 1) > 1 && length(size(rir)) > 1
    disp(size(rir))
    rir = rir(:,1);
end

% Delete potential fade-out windows
if fadeoutLength > 0
    rir = rir(1:end + round(-fadeoutLength*fs), :);
end

fprintf('The impulse has %d channels (before selecting the first one).\n', channels)
fprintf('The impulse has %d timesteps at %d kHz sampling rate = %f seconds.\n', size(rir,1), fs, size(rir,1) / fs)
%sound(rir, fs)

% rir = [rir; zeros(24000, 1)];

%% Load model and estimate parameters
net = DecayFitNetToolbox(nSlopes, fs);
[tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet] = net.estimateParameters(rir);
disp('==== DecayFitNet: Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(tVals_decayfitnet)
disp('==== DecayFitNet: Estimated A values (linear scale, A=0 indicates an inactive slope): ====') 
disp(aVals_decayfitnet)
disp('==== DecayFitNet: Estimated N values (linear scale): ====') 
disp(nVals_decayFitNet)

% Estimate true EDC
trueEDCs = rir2decay(rir, fs, [125, 250, 500, 1000, 2000, 4000], true, true, true); % doBackwardsInt=true, analyseFullRIR=true, normalize=true
timeAxis = linspace(0, (size(trueEDCs,1) - 1) / fs, size(trueEDCs,1) );
estimatedEDCs_decayfitnet = generateSyntheticEDCs(tVals_decayfitnet, aVals_decayfitnet, nVals_decayFitNet, timeAxis).';

f = figure;
f.Position = [150, 200, 1600, 600];
subplot(1, 2, 1);
hold on;
cmap = parula(size(trueEDCs, 2));
legendStr = {'Measured EDC, 125 Hz', 'DecayFitNet fit, 125 Hz', ...
    'Measured EDC, 250 Hz', 'DecayFitNet fit, 250 Hz',...
    'Measured EDC, 500 Hz', 'DecayFitNet fit, 500 Hz',...
    'Measured EDC, 1 kHz', 'DecayFitNet fit, 1 kHz',...
    'Measured EDC, 2 kHz', 'DecayFitNet fit, 2 kHz',...
    'Measured EDC, 4 kHz', 'DecayFitNet fit, 4 kHz'};
L = round(0.95*size(trueEDCs, 1)); % discard last 5 percent
allMSE = zeros(size(trueEDCs, 2), 1);
for bandIdx=1:size(trueEDCs, 2)
    plot(timeAxis(1:L), pow2db(trueEDCs(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '-');
    plot(timeAxis(1:L), pow2db(estimatedEDCs_decayfitnet(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '--');
    ylim([-60, 0]);
    
    allMSE(bandIdx) = mseLoss(pow2db(trueEDCs(1:L, bandIdx)), pow2db(estimatedEDCs_decayfitnet(1:L, bandIdx)));
end
legend(legendStr, 'Location', 'EastOutside');
title('DecayFitNet: Measured vs. estimated EDC fits');

fprintf('==== Average MSE between input EDCs and estimated fits: %.02f ====\n', mean(allMSE));
fprintf('MSE between input EDC and estimated fit for different frequency bands:\n 125 Hz: %.02f, 250 Hz: %.02f, 500 Hz: %.02f, 1 kHz: %.02f, 2 kHz: %.02f, 4 kHz: %.02f\n', allMSE(:));

%% Use Bayesian decay analysis with slice sampling
bda = BayesianDecayAnalysis(nSlopes, fs, parameterRanges, nIterations);
[tVals_bayesian, aVals_bayesian, nVals_bayesian] = bda.estimateParameters(rir);

disp('==== Bayesian analysis: Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(tVals_bayesian)
disp('==== Bayesian analysis: Estimated A values (linear scale, A=0 indicates an inactive slope): ====') 
disp(aVals_bayesian)
disp('==== Bayesian analysis: Estimated N values (linear scale): ====') 
disp(nVals_bayesian)

estimatedEDCs_bayesian = generateSyntheticEDCs(tVals_bayesian, aVals_bayesian, nVals_bayesian, timeAxis).';

subplot(1, 2, 2);
hold on;
cmap = parula(size(trueEDCs, 2));
legendStr = {'Measured EDC, 125 Hz', 'Bayesian fit, 125 Hz', ...
    'Measured EDC, 250 Hz', 'Bayesian fit, 250 Hz',...
    'Measured EDC, 500 Hz', 'Bayesian fit, 500 Hz',...
    'Measured EDC, 1 kHz', 'Bayesian fit, 1 kHz',...
    'Measured EDC, 2 kHz', 'Bayesian fit, 2 kHz',...
    'Measured EDC, 4 kHz', 'Bayesian fit, 4 kHz'};
L = round(0.95*size(trueEDCs, 1)); % discard last 5 percent
allMSE = zeros(size(trueEDCs, 2), 1);
for bandIdx=1:size(trueEDCs, 2)
    plot(timeAxis(1:L), pow2db(trueEDCs(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '-');
    plot(timeAxis(1:L), pow2db(estimatedEDCs_bayesian(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '--');
    ylim([-60, 0]);
    
    allMSE(bandIdx) = mseLoss(pow2db(trueEDCs(1:L, bandIdx)), pow2db(estimatedEDCs_bayesian(1:L, bandIdx)));
end
legend(legendStr, 'Location', 'EastOutside');
title('Bayesian decay analysis: Measured vs. estimated EDC fits');