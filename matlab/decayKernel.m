function D = decayKernel(tVals, timeAxis)
if size(tVals, 1) > size(tVals, 2), tVals = tVals.'; end % should be row vector

% Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
tauVals = -log(1e-6) ./ tVals;

% Calculate exponentials from decay rates
timeVals = -timeAxis .* tauVals;
exponentials = exp(timeVals);

% Calculate noise
L = length(timeAxis);
noise = linspace(1, 1/L, L).';

% Assemble decay kernel
D = [exponentials, noise];
end