function [Z] = rToZ(r)

% [Z] = rToZ(r)
% Calculates Fisher's R to Z transformation

Z = .5*log( (1 + r) ./ (1 - r) );

