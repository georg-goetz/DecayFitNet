function [decayFBands, normvals] = rir2decay(rir, fs, fBands, doBackwardsInt, analyseFullRIR, normalize)
% calculates energy decay curves in octave-bands from a room impulse response
%
% Inputs:
%   rir             - room impulse response [lenRir, 1]
%   fs              - sampling frequency, in Hz
%   fBands          - frequency bands for which RT should be calculated, in Hz
%                   [numBands, 1]
%   doBackwardsInt  - boolean that indicates whether decay should be
%                   backwards integrated (Schroeder decay) or not
%   analyseFullRIR  - if true: analyse RIR as is, if false: cut away
%                   everything in the RIR before the onset/direct sound
%   normalize       - normalize the decay to a maximum of 1. NOTE: for the
%                   Schroeder decay, this means it starts with 1, but for
%                   the squared RIR, the 1 may be at a later point. This is
%                   because the analysis is carried out starting from the
%                   maximum of the full impulse response and not from the
%                   maxima in the frequency bands
%
% Outputs:
%   decayFBands     - energy decay curves in specified octave-bands and the
%                   lowpass/highpass band if includeResidualBands=true, in
%                   linear scale [lenDecay, numBands]
%   normvals        - (optional) scaling values in case "normalize" is true
%
% (c) Georg Götz, Aalto University, 2020

if ~exist('fBands', 'var')
    fBands = [63; 125; 250; 500; 1000; 2000; 4000; 8000];
end
if ~exist('doBackwardsInt', 'var')
    doBackwardsInt = false;
end
if ~exist('analyseFullRIR', 'var')
    analyseFullRIR = true;
end
if ~exist('normalize', 'var')
    normalize = false;
end

numBands = numel(fBands);

% Remove trailing zeros
nTrailingZeros = find(flipud(rir), 1);
rir = rir(1:end-nTrailingZeros+1);

% Apply octave band filters to RIR, order=5
rirFBands = octaveFiltering(rir, fs, fBands);

% Remove filtering artefacts (last 5 permille) 
nDiscard = round(5e-3*length(rir));
rirFBands = rirFBands(1:end-nDiscard, :); 

% detect peak in rir, because the decay will be calculated from that point
% onwards
if analyseFullRIR
    t0 = 1;
else
    t0 = rirOnset(rir);
    %     [~, t0] = max(rir.^2); % very approximate alternative, if rirOnset does not give a good result
end

% get octave filtered rir from rir peak onwards
rirFBands = rirFBands(t0:end, :);

% calculate decay curves for every band
decayFBands = zeros(size(rirFBands));
for bIdx = 1:numBands
    thisRIR = rirFBands(:, bIdx);
    if doBackwardsInt == true
        decayFBands(:, bIdx) = schroederInt(thisRIR);
    else
        decayFBands(:, bIdx) = thisRIR.^2;
    end
end

% normalize to max 1 and store normalization values
normvals = [];
if normalize == true
    normvals = max(abs(decayFBands));
    decayFBands = decayFBands ./ normvals; % normalize to maximum 1
end

end
