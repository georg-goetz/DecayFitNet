function decay = rir2decay(rir, fs, fBands, doBackwardsInt, ignoreOnset, normalize)
% calculates energy decay curves in octave-bands from a room impulse response
%
% Inputs:
%   rir             - room impulse response [lenRir, 1]
%   fs              - sampling frequency, in Hz
%   fBands          - frequency bands for which RT should be calculated, in Hz
%                   [numBands, 1]
%   doBackwardsInt  - boolean that indicates whether decay should be
%                   backwards integrated (Schroeder decay) or not
%   normalize       - normalize the decay to a maximum of 1. NOTE: for the
%                   Schroeder decay, this means it starts with 1, but for
%                   the squared RIR, the 1 may be at a later point. This is 
%                   because the analysis is carried out starting from the 
%                   maximum of the full impulse response and not from the 
%                   maxima in the frequency bands
%
% Outputs:
%   decay           - energy decay curves in octave-bands, linear scale [lenDecay, numBands]
%
% (c) Georg Götz, Aalto University, 2020

if ~exist('fBands', 'var')
    fBands = [63; 125; 250; 500; 1000; 2000; 4000; 8000];
end
if ~exist('doBackwardsInt', 'var')
    doBackwardsInt = false;
end
if ~exist('ignoreOnset', 'var')
    ignoreOnset = false;
end
if ~exist('normalize', 'var')
    normalize = false;
end

numBands = numel(fBands);

% Apply octave band filters to RIR, order=3
rirFBands = zeros(size(rir));
for bandIdx=1:numBands
    [B, A] = octdsgn(fBands(bandIdx), fs, 3);
    rirFBands(:, bandIdx) = filter(B, A, rir);
end

% detect peak in rir, because the decay will be calculated from that point
% onwards
if ignoreOnset
    t0 = 1;
else
    t0 = rirOnset(rir);
end
% [~, t0] = max(rir.^2);

% get octave filtered rir from rir peak onwards
rirFBands = rirFBands(t0:end, :);

% calculate decay curves for every band
decay = zeros(size(rirFBands));
for bIdx = 1:numBands
    this_rir = rirFBands(:, bIdx); 
    this_rir = this_rir / max(abs(this_rir)); % normalize to maximum 1
    if doBackwardsInt == true
        decay(:, bIdx) = schroederInt(this_rir);
    else
        decay(:, bIdx) = this_rir.^2;
    end
end

if normalize == true
    decay = decay ./ max(decay);
end

end
    