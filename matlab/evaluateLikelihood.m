function likelihood = evaluateBayesianLikelihood(trueEDC_db, T, A, N, t)
% Formulas following Xiang, N., Goggans, P., Jasa, T. & Robinson, P. "Bayesian characterization of multiple-slope sound energy decays in coupled-volume systems." J Acoust Soc Am 129, 741–752 (2011).

% Calculate model EDC
tauVals = -log(1e-6) ./ T;
timeVals = -t * tauVals.';
exponentials = exp(timeVals);
expOffset = exp(-t(end)*tauVals.');
modelEDC = (exponentials - expOffset) * A;
noise = N .* (length(t):-1:1).';
modelEDC = modelEDC + noise;

% Convert to dB (true EDC is already db)
modelEDC = 10*log10(modelEDC);

% Evaluate Likelihood
K = length(t); % Eq. (1)
E = 0.5 * sum((trueEDC_db - modelEDC).^2); % Eq. (13)
likelihood = 0.5 * gamma(K/2) * (2*pi*E)^(-K/2); % Eq. (12)
% likelihood = 2 / E; % not in the paper, but works as well, because it's just a scaling issue
end
