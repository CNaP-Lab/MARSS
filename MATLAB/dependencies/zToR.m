function [r] = zToR(Z)

% [r] = zToR(Z)
% Calculates the inverse of Fisher's R to Z transformation

r = (exp(2.*Z) - 1) ./ (exp(2.*Z) + 1);

