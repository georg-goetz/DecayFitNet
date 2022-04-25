function edcs = generateSyntheticEDCs(T, A, noiseLevel, t)
% Generates an EDC from the estimated parameters: this is in principal the same as decayModel(), but it is more convenient for batch processing
    assert(size(T, 1) == size(A, 1) && size(T, 1) == size(noiseLevel, 1), 'Wrong size in the input (different batch size in T, A, N)')
    assert(size(T, 2) == size(A, 2), 'Wrong size in the input (different number of A and T values)');
    if size(t,1)>size(t,2), t=t.'; end
    batchSize = size(T, 1);
    nSamples = length(t);
    
    edcs = zeros(batchSize, nSamples);
    for bIdx=1:batchSize
        thisDecayModel = decayModel(T(bIdx,:) , A(bIdx,:) , noiseLevel(bIdx,:) , t, true); % compensateULI
        edcs(bIdx, :) = sum(thisDecayModel, 2);
    end
end