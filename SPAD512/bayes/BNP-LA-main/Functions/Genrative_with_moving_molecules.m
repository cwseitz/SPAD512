function[ Macro_times , Micro_times , S ] = Genrative_model(...
    ...
Popul_species  , lifetime_species , Lxyz            , wxyz    , ...
pulse_width_mu , mu_back          , pulse_frequency , D       , ...
Length_signal )


% The time difference between each pulse
tauu=10^-6./pulse_frequency;

Macro_times=[];
Micro_times=[];

% Check the values
if  max(max(exprnd(repmat(lifetime_species,1,1))))>tauu
    fprintf('Please select a lower pulse frequency')
else
  
k=sqrt(2*D*tauu);

X = zeros ( sum(Popul_species) , 2) ;
Y = zeros ( sum(Popul_species) , 2 ) ;
Z = zeros ( sum(Popul_species) , 2 ) ;



% Randomly sample the location of the molecules from the uniform probability
jk=0;

Time_totals=[];
SS=[];
jj=0;
KK=1;

%background
AB=cumsum(exprnd(mu_back,1,Length_signal));
ABB=AB(AB<=tauu*Length_signal);
[~,~,CCC]=histcounts(ABB,0:tauu:tauu*Length_signal);
back=-ABB+tauu*CCC;

if length(CCC)==0
    CCC=inf;
end


for m=1:length(Popul_species)

    sum_m=sum(Popul_species(1:m-1));
    for jk1=1:Popul_species(m)
        % Uniformly sample the initial loccations of molecules from the defined region
        X(sum_m+jk1,1) = Lxyz(1)*(1-2*rand());
        Y(sum_m+jk1,1) = Lxyz(2)*(1-2*rand());
        Z(sum_m+jk1,1) = Lxyz(3)*(1-2*rand());
        
        if ((1-exp(-pulse_width_mu(m).*exp(-2*( (X(sum_m+jk1,1)./wxyz(1)).^2 + ...
                                                (Y(sum_m+jk1,1)./wxyz(2)).^2 + ...
                                                (Z(sum_m+jk1,1)./wxyz(3)).^2 ))))>rand())==1
                                            
            AA=exprnd(lifetime_species(m));
            if  AA<tauu
                jj=jj+1;
                Time_totals(jj,1)=AA;
                SS(jj,1)=sum_m+jk1;
            end
        end
    end
end



if  CCC(KK)==1
    jk=jk+1;
    Time_totals(jj+1)=back(1);
    SS(jj+1,1)=sum_m+jk1+1;
    Micro_times(1,jk)=min(Time_totals);
    Macro_times(1,jk)=Micro_times(1,jk);
    h= histc(SS(find(Micro_times(1,jk)==Time_totals)),cumsum([1,Popul_species',2]));
    KK=KK+1;
    S(1,jk) = find(max(h)==h);
else
    if  length(Time_totals)>0
        jk=jk+1;
        Micro_times(1,jk)=min(Time_totals);
        Macro_times(1,jk)=Micro_times(1,jk);
        h= histc(SS(find(Micro_times(1,jk)==Time_totals)),cumsum([1,Popul_species',2]));
        S(1,jk) = find(max(h)==h);
    end
    
end



for i=2:Length_signal
    Time_totals=[];
    SS=[];
    jj=0;
    for m=1:length(Popul_species)
        sum_m=sum(Popul_species(1:m-1));
        for j1  = 1:Popul_species(m)
            
            % Sample the locations based on brownian motion
            X(sum_m+j1,2) = X(sum_m+j1,1)+(k(m)*randn());
            Y(sum_m+j1,2) = Y(sum_m+j1,1)+(k(m)*randn());
            Z(sum_m+j1,2) = Z(sum_m+j1,1)+(k(m)*randn());
       
            % Periodoc boundaries
            if  X(sum_m+j1,2) >= Lxyz(1)
                X(sum_m+j1,2)  = X(sum_m+j1,2)-2*Lxyz(1);
            end  
            if  X(sum_m+j1,2) <= -Lxyz(1)
                X(sum_m+j1,2)  = X(sum_m+j1,2)+2*Lxyz(1);
            end 
            if  Y(sum_m+j1,2) >= Lxyz(2)
                Y(sum_m+j1,2)  = Y(sum_m+j1,2)-2*Lxyz(2);
            end  
            if  Y(sum_m+j1,2) <= -Lxyz(2)
                Y(sum_m+j1,2)  = Y(sum_m+j1,2)+2*Lxyz(2);
            end 
            if  Z(sum_m+j1,2) >= Lxyz(3)
                Z(sum_m+j1,2)  = Z(sum_m+j1,2)-2*Lxyz(3);
            end  
            if  Z(sum_m+j1,2) <= -Lxyz(3)
                Z(sum_m+j1,2)  = Z(sum_m+j1,2)+2*Lxyz(3);
            end 
            
            if ((1-exp(-pulse_width_mu(m).*exp(-2*( (X(sum_m+jk1,2)./wxyz(1)).^2 + ...
                                                (Y(sum_m+jk1,2)./wxyz(2)).^2 + ...
                                                (Z(sum_m+jk1,2)./wxyz(3)).^2 ))))>rand())==1
                                             
                AA=exprnd(lifetime_species(m));

                if  AA<tauu
                    jj=jj+1;
                    Time_totals(jj,1)=AA;
                    SS(jj,1)=sum_m+j1;
                end
            end
        end
    end
    X(:,1) = X(:,2);
    Y(:,1) = Y(:,2);
    Z(:,1) = Z(:,2);


    if CCC(KK)==i
   
        jk=jk+1;
        Time_totals(jj+1,1)=back(KK);
        SS(jj+1,1)=sum_m+j1+1;
            
        Micro_times(1,jk) = min(Time_totals);
        Macro_times(1,jk) = i*tauu+Micro_times(1,jk);

        h= histc(SS(find(Micro_times(1,jk)==Time_totals)),cumsum([1,Popul_species']));
        
        KK=KK+1;
        if KK>length(CCC)
           CCC(KK)=inf;
        end
        S(1,jk) = find(max(h)==h);
    else
        if  length(Time_totals)>0
            jk=jk+1;
            Micro_times(1,jk) = min(Time_totals);
            Macro_times(1,jk) = i*tauu+Micro_times(1,jk);

            h= histc(SS(find(Micro_times(1,jk)==Time_totals)),cumsum([1,Popul_species']));
            S(1,jk) = find(max(h)==h);
        end
    end
        
end



  end
end