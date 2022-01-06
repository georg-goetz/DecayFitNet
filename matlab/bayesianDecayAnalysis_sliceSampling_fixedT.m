function [testedParameters, likelihoods] = bayesianDecayAnalysis_sliceSampling_fixedT(trueEDC_db, modelOrder, tVals, aCandidates, nCandidates, nIterations, t)
% Following Jasa, T. & Xiang, N. "Efficient estimation of decay parameters in acoustically coupled-spaces using slice sampling." J Acoust Soc Am 126, 1269–1279 (2009).
% With fixed T, so only A and N gets estimated

assert(length(aCandidates)==length(nCandidates), 'There must be an equal number of A and N candidates.');
nPointsPerDim = length(aCandidates);
nParameters = modelOrder+1;

testedParameters = zeros(nIterations, nParameters);
likelihoods = zeros(nIterations, 1);

x0 = randi(nPointsPerDim, nParameters, 1); % randomly draw indices of first evaluated sample
y0 = rand * evaluateLikelihood(trueEDC_db, tVals, aCandidates(x0(1:modelOrder)).', nCandidates(x0(modelOrder+1)), t);

for sampleIdx=1:nIterations
    paramIdx = mod(sampleIdx-1, nParameters) + 1; % vary variables in turn
    
    % 1: Vary parameter until slice is established: the slice is the
    % region, for which the likelihood is higher than the previously found
    % likelihood
    
    % Find left edge of slice
    thisX0Left = x0;
    while(thisX0Left(paramIdx)>1)
        thisX0Left(paramIdx) = thisX0Left(paramIdx) - 1;
        thisY0Left = evaluateLikelihood(trueEDC_db, tVals, aCandidates(thisX0Left(1:modelOrder)).', nCandidates(thisX0Left(modelOrder+1)), t);

        if(thisY0Left < y0)
            break;
        end
    end
    
    % Find right edge of slice
    thisX0Right = x0;
    while(thisX0Right(paramIdx)<nPointsPerDim-1)
        thisX0Right(paramIdx) = thisX0Right(paramIdx) + 1;
        thisY0Right = evaluateLikelihood(trueEDC_db, tVals, aCandidates(thisX0Right(1:modelOrder)).', nCandidates(thisX0Right(modelOrder+1)), t);

        if(thisY0Right < y0)
            break;
        end
    end
    
    % 2: Draw new parameter value from the slice, to find a new and higher
    % likelihood
    while(true)
        x1 = x0;
        x1(paramIdx) = randi(thisX0Right(paramIdx)-thisX0Left(paramIdx)+1) + thisX0Left(paramIdx) - 1;
        y1 = evaluateLikelihood(trueEDC_db, tVals, aCandidates(x1(1:modelOrder)).', nCandidates(x1(modelOrder+1)), t);

        if(y1>y0)
            % higher likelihood found, continue with next iteration step
            break;
        else
            % drawn value is not actually in slice due to quantization
            % error (slice is established as multiples of a step),
            % therefore adapt slice interval
            if vecnorm(x1-thisX0Left) < vecnorm(x1-thisX0Right)
                thisX0Left = x1;
            else
                thisX0Right = x1;
            end
        end
    end
    testedParameters(sampleIdx, :) = x1;
    likelihoods(sampleIdx) = y1;
    
    % prepare for next iteration
    y0 = rand*y1;
    x0 = x1;
end
end