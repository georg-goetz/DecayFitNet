function envelopeT = schroederT2EnvelopeT(schroederT)
assert(any(size(schroederT)==1), 'schroederT must be a row vector.');

% Envelope is in linear scale, not quadratic, therefore decay rates halve, 
% and T values double
envelopeT = 2*schroederT;
end