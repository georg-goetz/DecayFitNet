function decayingNoiseEDC = generateSyntheticDecayingNoiseEDCFiltered(T, A, noiseLevel, t, octaveFilterbank, octaveBandIdx)
% Generate exponential decays
tauVals = -log(1e-6) ./ T;
timeVals = -t * tauVals.';
exponentials = exp(timeVals);

% Generate noise
A = [A; noiseLevel];
decayingNoiseEDC = randn(length(t), length(A));

% Filter noise in octave bands and normalize to zero mean unit variance
decayingNoiseEDC_ofb = octaveFilterbank(decayingNoiseEDC);
decayingNoiseEDC = squeeze(decayingNoiseEDC_ofb(:, octaveBandIdx, :));
decayingNoiseEDC = decayingNoiseEDC - mean(decayingNoiseEDC);
decayingNoiseEDC = decayingNoiseEDC ./ std(decayingNoiseEDC);

% Square, shape gaussian noise with exponentials, and sum all terms
% together, sqrt again to get into linear scale again
decayingNoiseEDC = decayingNoiseEDC.^2;
decayingNoiseEDC(:, 1:length(A)-1) = decayingNoiseEDC(:, 1:length(A)-1) .* exponentials;
decayingNoiseEDC = decayingNoiseEDC * A;
decayingNoiseEDC = sqrt(decayingNoiseEDC);
end