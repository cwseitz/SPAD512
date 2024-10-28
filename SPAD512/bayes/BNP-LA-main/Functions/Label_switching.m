function[Data] = Label_switching(Data,i)

cost_matrix = zeros(Data.Number_species);
      for n=1:Data.Number_species
          for m=1:Data.Number_species

cost_matrix(n,m) = sqrt(( Data.max_lambda(1,m)*(Data.lambda(i,n)+Data.max_lambda(1,m))*(Data.PI(i,n)^2)*...
                       (exp(-2*Data.t_p/Data.lambda(i,n))-exp(-2*Data.T_max/Data.lambda(i,n))) +...
                     4*Data.lambda(i,n)*Data.max_lambda(1,m)*Data.PI(i,n)*Data.max_PI(1,m)*...
                       (exp(-(1/Data.lambda(i,n) +1/Data.max_lambda(1,m))*Data.T_max)-...
                        exp(-(1/Data.lambda(i,n) +1/Data.max_lambda(1,m))*Data.t_p))+...
                     Data.lambda(i,n)*(Data.lambda(i,n)+Data.max_lambda(1,m))*(Data.max_PI(1,m)^2)*...
                       (exp(-2*Data.t_p/Data.max_lambda(1,m))-exp(-2*Data.T_max/Data.max_lambda(1,m))))/...
                  (2*Data.lambda(i,n)*Data.max_lambda(1,m)*(Data.lambda(i,n)+Data.max_lambda(1,m)))) ;
              
          end
      end   


% Applying the Hungarian algorithm
[assig,~] = munkres(cost_matrix);

Data.lambda(i,:) = Data.lambda(i,assig);
Data.PI(i,:)     = Data.PI(i,assig);

