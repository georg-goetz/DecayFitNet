%% Parameters
audio_path = '/m/cs/work/falconr1/datasets/MusicSamples';
audio_path = '/Volumes/scratch/work/falconr1/datasets/MusicSamples';
%rir_fname = 'Single_503_1_RIR.wav';  % Single slope clean
%rir_fname = 'Single_502_1_RIR.wav';
%rir_fname = 'Single_EE_lobby_1_RIR.wav';  % Long tail

audio_path = '/Volumes/scratch/elec/t40527-hybridacoustics/datasets/summer830/raw_rirs';
%rir_fname = '0825_1_raw_rirs.wav';
%rir_fname = '0825_4_raw_rirs.wav';
%rir_fname = '0001_4_raw_rirs.wav';  % Huge
rir_fname = '0001_1_raw_rirs.wav';  % First measurement


audio_path = '../model/';  %local path
%rir_fname = '0825_1_raw_rirs.wav';
%rir_fname = '0825_4_raw_rirs.wav';
%rir_fname = '0001_4_raw_rirs.wav';  % Huge
rir_fname = '0001_1_raw_rirs.wav';  % First measurement

fadeout_length = 1;  % in secs
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

disp(sprintf('The impulse has %d channels (before selecting the first one.', channels))
disp(sprintf('The impulse has %d timesteps at %d sampling rate = %f seconds.', size(rir,1), fs, size(rir,1) / fs))
%sound(rir, fs)


%% Load model and estimate parameters

time_axis = linspace(0, (size(rir,1) - 1) / fs, size(rir,1) );

net = DecayFitNetToolbox();
[t_values, a_values, n_values] = net.estimate_parameters(rir, true, true);
disp('==== Estimated T values (in seconds, T=0 indicates an inactive slope): ====\n') 
disp(t_values)
disp('==== Estimated A values (linear scale, A=0 indicates an inactive slope): ====\n') 
disp(a_values)
disp('==== Estimated N values (linear scale): ====\n') 
disp(n_values)
estimated_edc = net.generate_synthetic_edcs(t_values, a_values, n_values, time_axis);

% Get true EDC to compare
true_edc = rir2decay(rir, fs, [125, 250, 500, 1000, 4000, 8000], true, true);
true_edc = DecayFitNetToolbox.discarLast5(true_edc);

f = figure(2);
subplot(3,1,1);
utils.plot_waveform(rir, fs, 'Original RIR', f)
subplot(3,1,2);
ylim([-1, 1])
utils.plot_waveform(10 .* log10(true_edc), fs, 'True EDC', f)
ax = gca;
ax.YScale = 'log';
ylim([-100, 0])
subplot(3,1,3);
utils.plot_waveform(estimated_edc, fs, 'Estimated EDC', f)
ax = gca;
ax.YScale = 'log';