function [B, A] = fractionalOctaveFilter(centerFrequency, fs)

fPassBandLow = centerFrequency / sqrt(2);
fPassBandHigh = centerFrequency * sqrt(2);
aPass = db2mag(-3);

end