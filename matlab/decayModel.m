function edcModel = decayModel(T, A, N, timeAxis, compensateULI)
    if nargin < 5
        compensateULI = false;
    end
    
    % get decay rate: decay energy should have decreased by 60 dB after T
    % seconds
    zeroT = (T == 0);
    assert(all(A(zeroT)==0), 'T values equal zero detected, for which A values are nonzero. This yields division by zero. For inactive slopes, set A to zero.');
    tauVals = log(1e6) ./ T; 
    
    % calculate decaying exponential terms
    timeVals = -timeAxis * tauVals.';
    exponentials = exp(timeVals);
    
    % account for limited upper limit of integration, see:
    % Xiang, N., Goggans, P. M., Jasa, T. & Kleiner, M. "Evaluation of 
    % decay times in coupled spaces: Reliability analysis of Bayeisan decay 
    % time estimation." J Acoust Soc Am 117, 3707–3715 (2005).
    if compensateULI
        expOffset = exponentials(end,:); 
    else
        expOffset = 0;
    end
    
    % calculate final exponential terms
    exponentials = (exponentials - expOffset) .* A.';
    
    % zero exponentials where T=A=0 (they are NaN now because div by 0)
    exponentials(:, zeroT) = 0;
    
    % calculate noise term
    noise = N .* (length(timeAxis):-1:1).';
    
    % set up edc model, where each column is either an exponential or the
    % noise term
    edcModel = [exponentials, noise];
end