function decay = schroederInt(rir, upperLim)
% calculates Schroeder decay from a room impulse response by doing
% backwards integration
% 
% Inputs:
%   rir          - room impulse response [lenRir, 1]
%   upperLim     - upper limit of integration, should be chosen such that
%                noise floor is not included in integration
%
% Outputs:
%   decay   - Schroeder decay, linear scale [lenRir, 1]
%
% (c) Georg Götz, Aalto University, 2020

if ~exist('upperLim', 'var')
    upperLim = length(rir);
else
    assert(upperLim <= length(rir), 'Upper limit of integration must be smaller than length of RIR.');
end

irFlipped = flipud(rir(1:upperLim,:)); % flip, because of backwards integration
irSquared = irFlipped.^2;
irIntegrated = (1/length(rir)) * cumsum(irSquared); % integrate with trapezoids
decay = flipud(irIntegrated); % flip again, so that decay is right way around, and get rid of 0 value (would throw error when calculating log)

end