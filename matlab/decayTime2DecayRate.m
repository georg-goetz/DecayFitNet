function decayRate = decayTime2DecayRate(decayTime, fs)
% Convert between exp((-13.8*t)/(fs*decayTime)) and exp(-decayRate*t) with
% t being time in samples
decayRate = log(1e6)./decayTime/fs;
end