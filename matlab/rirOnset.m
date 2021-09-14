function onsetIdx = rirOnset(rir)
% This onset detection approximately follows the "Ds approach" outlined in 
%
%   Defrance, G., Daudet, L. & Polack, J.-D. "Finding the onset of a room 
%   impulse response: Straightforward? J Acoust Soc Am 124, 
%   EL248–EL254 (2008).
%
% (c) Georg Götz, Aalto University, 2021
  
RIR = spectrogram(rir, 64, 60, 64);
RIR = sum(abs(RIR), 1);
E = RIR(2:end) ./ RIR(1:end-1);
[~, onsetIdx] = max(E);
onsetIdx = (onsetIdx-2) * 4 + 64; % 4 is the hopsize of spectrogram, 2 because we want one hopsize as safety margin
end