function [fCenter, fLower, fUpper] = getBandFrequencies(n, fs, bandWidth, fRef)
% Get band frequencies for 1/n-octave band filter
if nargin < 3
    fRef = 1000; % Hz
end

bIdxLowest = ceil(log(bandWidth(1)/fRef) / log(2^(1/n)));
bIdxHighest = floor(log(bandWidth(2)/fRef) / log(2^(1/n)));
bIdx = bIdxLowest:bIdxHighest;

fCenter = round((2^(1/n)).^bIdx .* fRef, 3);
fLower = fCenter;
fUpper = fCenter;

end