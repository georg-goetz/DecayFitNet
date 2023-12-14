function [allDecays, normVals] = rir2decayBatch(rirs, fs, fBands, doBackwardsInt, analyseFullRIR, normalize)
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

% Initialize arrays
nBands = length(fBands);
nRIRs = size(rirs, 1);
allDecays = cell(nRIRs, 1);
normVals = ones(nRIRs, nBands);

bb=0; % for progress bar
for rirIdx=1:nRIRs
    % Progress bar
    pstr = sprintf('Progress: Measurement %d / %d [%.02f %%%%]. \n', rirIdx, nRIRs, 100*rirIdx/nRIRs);
    fprintf([repmat('\b',[1 bb]),pstr]) %erase current line with backspaces
    bb = length(pstr)-1;
    
    [theseDecays, theseNormVals] = rir2decay(rirs(rirIdx, :).', fs, fBands, doBackwardsInt, analyseFullRIR, normalize);
    allDecays{rirIdx} = theseDecays;
    if normalize, normVals(rirIdx, :) = theseNormVals; end
end
end