close all; clear variables; clc; lastwarn('');
%% External dependencies
addpath(genpath('octave'));

%% Parameters
audio_path = '../model';  %local path
rir_fname = '0001_1_sh_rirs.wav';  % First measurement of Motus dataset

fadeout_length = 0;  % in secs
%% Load an impulse

[rir, fs] = audioread(fullfile(audio_path, rir_fname));
channels = size(rir, 2);

% Keep only the first channel
if size(rir, 1) > 1 && length(size(rir)) > 1
    disp(size(rir))
    rir = rir(:,1);
end

% Delete potential fade-out windows
if fadeout_length > 0
    rir = rir(1:end + round(-fadeout_length*fs), :);
end

fprintf('The impulse has %d channels (before selecting the first one).\n', channels)
fprintf('The impulse has %d timesteps at %d kHz sampling rate = %f seconds.\n', size(rir,1), fs, size(rir,1) / fs)
%sound(rir, fs)


%% Load model and estimate parameters
net = DecayFitNetToolbox();
[t_values, a_values, n_values] = net.estimate_parameters(rir, false); % flags: includeResidualBands
disp('==== Estimated T values (in seconds, T=0 indicates an inactive slope): ====') 
disp(t_values)
disp('==== Estimated A values (linear scale, A=0 indicates an inactive slope): ====') 
disp(a_values)
disp('==== Estimated N values (linear scale): ====') 
disp(n_values)

true_edcs = rir2decay(rir, fs, [125, 250, 500, 1000, 2000, 4000], true, true, true); 
time_axis = linspace(0, (size(true_edcs,1) - 1) / fs, size(true_edcs,1) );
estimated_edcs = net.generate_synthetic_edcs(t_values, a_values, n_values, time_axis).';

figure;
hold on;
cmap = parula(size(true_edcs, 2));
legendStr = {'Measured EDC, 125 Hz', 'DecayFitNet fit, 125 Hz', ...
    'Measured EDC, 250 Hz', 'DecayFitNet fit, 250 Hz',...
    'Measured EDC, 500 Hz', 'DecayFitNet fit, 500 Hz',...
    'Measured EDC, 1 kHz', 'DecayFitNet fit, 1 kHz',...
    'Measured EDC, 2 kHz', 'DecayFitNet fit, 2 kHz',...
    'Measured EDC, 4 kHz', 'DecayFitNet fit, 4 kHz'};
L = round(0.95*size(true_edcs, 1)); % discard last 5 percent
allMSE = zeros(size(true_edcs, 2), 1);
for bandIdx=1:size(true_edcs, 2)
    plot(time_axis(1:L), pow2db(true_edcs(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '-');
    plot(time_axis(1:L), pow2db(estimated_edcs(1:L, bandIdx)), 'Color', cmap(bandIdx, :), 'LineWidth', 2, 'LineStyle', '--');
    ylim([-60, 0]);
    
    allMSE(bandIdx) = mseLoss(pow2db(true_edcs(1:L, bandIdx)), pow2db(estimated_edcs(1:L, bandIdx)));
end
legend(legendStr, 'Location', 'EastOutside');
fprintf('==== Average MSE between input EDCs and estimated fits: %.02f ====\n', mean(allMSE));
fprintf('MSE between input EDC and estimated fit for different frequency bands:\n 125 Hz: %.02f, 250 Hz: %.02f, 500 Hz: %.02f, 1 kHz: %.02f, 2 kHz: %.02f, 4 kHz: %.02f\n', allMSE(:));