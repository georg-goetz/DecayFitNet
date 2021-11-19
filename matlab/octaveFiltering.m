function outBands = octaveFiltering(inputSignal, fs, fBands, includeResidualBands)
numBands = numel(fBands);
outBands = zeros(length(inputSignal), numBands);

for bIdx = 1:numBands
    thisBand = fBands(bIdx) .* [1/sqrt(2), sqrt(2)];
   
    % IIR
    [z, p, k] = butter(5, thisBand/fs*2, 'bandpass');
  
    % Zero phase filtering
    [sos, g] = zp2sos(z, p, k);
    outBands(:, bIdx) = filtfilt(sos, g, inputSignal);
end

% Residual bands = everything below and above the octave bandpass
% filters
if includeResidualBands == true    
    % IIR
    [zLow, pLow, kLow] = butter(5, (1/sqrt(2))*fBands(1)/fs*2);
    [zHigh, pHigh, kHigh] = butter(5, sqrt(2)*fBands(end)/fs*2, 'high');
        
    % Zero phase filtering
    [sosLow, gLow] = zp2sos(zLow, pLow, kLow);
    [sosHigh, gHigh] = zp2sos(zHigh, pHigh, kHigh);
    rirLowpass = filtfilt(sosLow, gLow, inputSignal);
    rirHighpass = filtfilt(sosHigh, gHigh, inputSignal);
     
    outBands = [rirLowpass, outBands, rirHighpass];
end
end