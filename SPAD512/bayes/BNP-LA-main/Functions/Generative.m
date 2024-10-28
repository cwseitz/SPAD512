function[Data] = Generative(Data)


Data.T_min                   =  0                          ; % The lower time of the cut off (set to zero)  (ns)
Data.T_max                   =  Data.delta_t               ; % The upper time of the cut off (set to interpulse window)  (ns)

Data.Total_experimental_time = Data.Number_pulse*Data.delta_t     ;  % The total time of the experiment

N_back = poissrnd(Data.Total_experimental_time*Data.emission_back) ;  % Sample the total number of photons coming from the background
V      = sort(rand(1,N_back)*Data.Total_experimental_time)         ;  % Sort the background photons


jj_test=1                                   ;
t_det=[];
S_true = [];
for n=1:Data.Number_pulse
    
    v_b = [];
    v_b = V(V<n*Data.delta_t) ;
    v_b = v_b(v_b>=(n-1)*Data.delta_t);
    
    if (1-exp(-2*Data.sigma_p*sum(Data.excitation_species)))>rand()
        sp = Discrete_sampler(Data.excitation_species./sum(Data.excitation_species))    ;
        t_s  = Data.t_p + Data.sigma_p*randn() +exprnd(Data.emission_species(sp))       ;
        t_s = t_s - floor(t_s/Data.T_max)*Data.T_max;
        
        if  ~isempty(v_b)
            t_det(1,jj_test) = min( [ t_s , v_b-(n-1)*Data.delta_t])                    ;
            S_true(1,jj_test) = sp                                                      ;
            jj_test=jj_test+1;
        else
            t_det(1,jj_test) = t_s;
            S_true(1,jj_test) = sp;
            jj_test=jj_test+1;
        end
        
    else
        
        if  ~isempty(v_b)
            t_det(1,jj_test) = min( v_b-(n-1)*Data.delta_t);
            jj_test=jj_test+1;
        end
        
    end
    
end

%t_det=t_det(t_det>=Data.T_min)                      ;
Data.t_det=t_det(t_det<=Data.T_max)                 ;
Data.s_true = S_true                                ;


end