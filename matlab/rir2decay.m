function [decayFBands, normvals,rirFBands] = rir2decay(rir, fs, fBands, doBackwardsInt, analyseFullRIR, normalize, includeResidualBands)
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
%   includeResidualBands    - set this to true if you want the function to
%                           also return EDCs for the lowpass band below the
%                           first octave band and the highpass band above
%                           the last octave band
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
    analyseFullRIR = false;
end
if ~exist('normalize', 'var')
    normalize = false;
end
if ~exist('includeResidualBands', 'var')
    includeResidualBands = false;
end

numBands = numel(fBands);
fprintf('Processing the band with center frequency: ');
% Apply octave band filters to RIR, order=3
rirFBands = zeros(length(rir), numBands);
for bIdx = 1:numBands
    %     thisBand = fBands(bIdx) .* [1/sqrt(2), sqrt(2)];
    thisBand = fBands(bIdx) .* [1/sqrt(1.5), sqrt(1.5)];
    fprintf('%d Hz ', fBands(bIdx));
    
    %     % FIR
    %     bCoeffsFilt = fir1(fs/4, thisBand/fs*2);
    %     rirFBands(:,bIdx) = filtfilt(bCoeffsFilt, 1, rir);
    
    % IIR
    [z, p, k] = butter(5, thisBand/fs*2, 'bandpass');
    %     [z, p, k] = cheby1(3, 1, thisBand/fs*2, 'bandpass');
    
    % Variant 1: Zero phase filtering
    %     [sos, g] = zp2sos(z, p, k);
    %     rirFBands(:, bIdx) = filtfilt(sos, g, rir);
    
    % Variant 2: Regular IIR filtering
    % do the filtering in reverse direction - makes the starting peak less
    sos = zp2sos(z, p, k);
    rirFBands(:, bIdx) = flipud(sosfilt(sos, flipud(rir)));
end

if includeResidualBands == true
    disp('Processing lowpass and highpass bands.');
    
    %     % FIR
    %     bCoeffsLowpass = fir1(fs/4, (1/sqrt(2))*fBands(1)/fs*2, 'low');
    %     bCoeffsHighpass = fir1(fs/4, sqrt(2)*fBands(end)/fs*2, 'high');
    %     rirLowpass = filtfilt(bCoeffsLowpass, 1, rir);
    %     rirHighpass = filtfilt(bCoeffsHighpass, 1, rir);
    
    % IIR
    [zLow, pLow, kLow] = butter(5, (1/sqrt(2))*fBands(1)/fs*2);
    [zHigh, pHigh, kHigh] = butter(5, sqrt(2)*fBands(end)/fs*2, 'high');
    %     [zLow, pLow, kLow] = cheby1(3, 1, (1/sqrt(2))*fBands(1)/fs*2);
    %     [zHigh, pHigh, kHigh] = cheby1(3, 1, sqrt(2)*fBands(end)/fs*2, 'high');
    
    %     % Variant 1: Zero phase filtering
    %     [sosLow, gLow] = zp2sos(zLow, pLow, kLow);
    %     [sosHigh, gHigh] = zp2sos(zHigh, pHigh, kHigh);
    %     rirLowpass = filtfilt(sosLow, gLow, rir);
    %     rirHighpass = filtfilt(sosHigh, gHigh, rir);
    %
    % Variant2: Regular filtering
    sosLow = zp2sos(zLow, pLow, kLow);
    sosHigh = zp2sos(zHigh, pHigh, kHigh);
    rirLowpass = sosfilt(sosLow, rir);
    rirHighpass = sosfilt(sosHigh, rir);
    
    rirFBands = [rirLowpass, rirFBands, rirHighpass];
    numBands = numBands + 2;
end

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
    
    % compensate the filter energy loss
    % This the ratio between the ideal bandpass signal and a filtered bandpass
    % signal.
    filterFactor = db2mag([15.7158  -13.4540   -6.4259   -5.8049   -5.0090   -4.8093   -4.6998   -4.6814    0.5196  -10.0050]);
    
    if includeResidualBands == true
    else
        filterFactor = filterFactor(2:end-1);
    end
    
    normvals = normvals ./ filterFactor;
end



end
