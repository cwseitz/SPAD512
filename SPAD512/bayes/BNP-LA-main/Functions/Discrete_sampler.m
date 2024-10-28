function j = Discrete_sampler( p )

% P = cumsum(p);
% j = find( P(end)*rand(1) <= P, 1 );

[p,ind] = sort(p,'descend');
P = cumsum(p);
j = ind( find( P(end)*rand(1) <= P, 1 ) );