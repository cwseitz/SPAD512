function x = dirichletRnd(a)
% Generate samples from a Dirichlet distribution.
% Input:
%   a: k dimensional vector
%   m: k dimensional mean vector
% Outpet:
%   x: generated sample x~Dir(a,m)
% Written by Mo Chen (sth4nth@gmail.com).

x = gamrnd(a,1);
x = x/sum(x);

