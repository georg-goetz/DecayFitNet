function edc = generateSyntheticEDCs(T, A, noiseLevel, t)
%% Generates an EDC from the estimated parameters: this is in principal the same as decayModel(), but it is more convenient for batch processing
    assert(size(T, 2) == size(A, 2) && size(T, 2) == size(noiseLevel, 2), 'Wrong size in the input (different batch size in T, A, N)')
    [nSlopes, batchSize] = size(T);
    nSamples = length(t);

    % Permute to [batch_size, n_slopes], for consistency with the toolbox.
    T = permute(T, [2,1]);
    A = permute(A, [2,1]);
    noiseLevel = permute(noiseLevel, [2,1]);

    % Calculate decay rates, based on the requirement that after T60 seconds, the level must drop to -60dB
    tauVals = -log(1e-6) ./ T;

    % Repeat values such that end result will be (sampled_idx, batch_size, n_slopes)
    %t_rep = repmat(t, [1, size(T, 2), size(T, 1)]);
    %tau_vals_rep = repmat(permute(tau_vals, [2,1]), [size(t, 1), 1, 1 ]);
    % Repeat values such that end result will be (batch_size, n_slopes, sampled_idx)
    t_rep = repmat(permute(t, [1, 3, 2]), [batchSize, nSlopes, 1]);
    tauVals_rep = repmat(permute(tauVals, [1,2,3]), [1, 1, nSamples]);
    assert(all(size(t_rep) == size(tauVals_rep)), 'Wrong size in tau')

    % Calculate exponentials from decay rates
    timeVals = -t_rep .* tauVals_rep;
    exponentials = exp(timeVals);

    % Zero exponentials where T=A=0
    for batchIdx = 1:batchSize
        zeroT = (T(batchIdx, :) == 0);
        exponentials(batchIdx, zeroT, :) = 0;
    end

    % Offset is required to make last value of EDC be correct
    expOffset = repmat(exponentials(:, :, end), [1, 1, nSamples]);

    % Repeat values such that end result will be (batch_size, n_slopes, sample_idx)
    A_rep = repmat(A, [1, 1, nSamples]);

    % Multiply exponentials with their amplitudes and sum all exponentials together
    edcs = A_rep .* (exponentials - expOffset);
    edc = reshape(sum(edcs, 2), batchSize, nSamples);

    % Add noise
    noise = noiseLevel .* linspace(nSamples, 1, nSamples);
    edc = edc + noise;
end