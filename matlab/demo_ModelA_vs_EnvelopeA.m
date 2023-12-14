close all; clear variables; clc;
%% This demo shows how to convert parameters of the Schroeder model to parameters of a decaying noise model
schroederT = [0.5, 1.5]; % seconds
schroederA = [1, 0.01]; % decay model amplitude

fs = 48000; % in Hz
L = 3*fs; % in samples

%% Set up decay model
timeAxis = linspace(0, (L-1)/fs, L).';
edfModel = decayKernel(schroederT, timeAxis);
edfModel(:, end) = []; % throw away noise term
edfModel = edfModel * schroederA.';

%% Determine envelope from EDF model
[envelopesT, envelopesA] = schroeder2Envelope(schroederT, schroederA, fs);

envelopes = decayKernel(envelopesT, timeAxis);
envelopes = envelopes(:, 1:end-1) .* envelopesA;

%% Set up decaying Gaussian noise
gaussianNoise = randn(L, length(schroederT));
decayingGaussianNoise = sum(gaussianNoise .* envelopes, 2);

%% Calculate EDF from Gaussian noie
decayingGaussianNoiseEDF = flipud(cumsum(flipud(decayingGaussianNoise.^2)));

%% Plot
figure;
hold on;
plot(timeAxis, pow2db(edfModel));
plot(timeAxis, pow2db(decayingGaussianNoiseEDF));
legend('EDF Model', 'Decaying Gaussian noise EDF');