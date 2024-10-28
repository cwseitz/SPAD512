function j = Discrete_sampler_row( p , sizee )


% P = cumsum(p,2);
% j=sum(~(P(:,end).*rand(sizee ,1) <= P),2)+1;

j=sum(~(rand(sizee,1) <= cumsum(softmax(p),1)'),2)+1;

