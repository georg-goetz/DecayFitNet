function [envelopeT, envelopeA] = schroeder2Envelope(schroederT, schroederA, fs)
assert(length(size(schroederT)) == length(size(schroederA)), 'Dimensions mismatch between T and A.');
assert(sum(size(schroederT) == size(schroederA)) > 0, 'schroederT and schroederA have different sizes.');

% Go from decay time (~T60) to decay rate of exp(-decayRate*t) with t being
% time in samples
decayRate = decayTime2DecayRate(schroederT, fs);

% Exponential exp(-decayRate*t) = [exp(-decayRate)]^t = decayPerSample^t
decayPerSample = exp(-decayRate);

% Use geometric sum to calculate sum_t=0^inf {decayPerSample^t}
decayEnergy = 1 ./ (1 - decayPerSample);

% We want sum_t=0^inf {gaussianNoise(t) * envelope(t) * scaling} = 1, so we
% can scale the decay model amplitudes with scaling. Gaussian noise has
% RMS=1, so we can ignore it.
scaling = sqrt(1./decayEnergy);
envelopeA = sqrt(schroederA).*scaling;

envelopeT = schroederT2EnvelopeT(schroederT);
end