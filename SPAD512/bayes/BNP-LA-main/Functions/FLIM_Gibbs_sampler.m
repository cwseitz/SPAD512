function[Data] = FLIM_Gibbs_sampler( Data , Number_iter  )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%    Pre calculations    %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trace_size         = length(Data.t_det)                               ;  % The length of the data set
OLD_Iteration_size = size(Data.lambda,1)                              ;  % The length of the sampled values
i                  = OLD_Iteration_size                               ;  % Set the indicator to the length of the sampled values
max_iter           = Number_iter+i-1                                  ;  % Maximum number of iterations


% initialization of the posterior values
if i==1
   Data.max_post   = -10^10                                           ;
   Data.max_lambda = Data.lambda                                      ;
   Data.max_S      = Data.S                                           ;
   Data.max_PI     = Data.PI                                          ;
end

% precalculations of values which are used to calculate the likelihood
time_t             = Data.t_p - Data.t_det                            ;
sqrt2sigma         = sqrt(2*(Data.sigma_p^2))                         ;
tmintp             = Data.T_min-Data.t_p                              ;
tmaxtp             = Data.T_max-Data.t_p                              ;
tmax               = Data.T_max                                       ;
erftmin_efttmax    = erf(tmaxtp /sqrt2sigma)-erf(tmintp/sqrt2sigma)   ;
sigma_pp           = Data.sigma_p                                     ; 
if isfield(Data,'Ntmp')
    Ntmp           = Data.Ntmp                                        ;
else
    Ntmp = 0                                                          ;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  Beginning of the sampler loop  %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while i<max_iter
      i=i+1;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%   Sample the labels on each detected photon   %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %constt      = sigma_pp./Data.lambda(i-1,:);
          
      Probs = log(calProbs(time_t ,tmax,sigma_pp,1./Data.lambda(i-1,:),Data.PI(i-1,:),Ntmp)');
      
%       Data.S(1,:) = Discrete_sampler_row( (repmat(log(Data.PI(i-1,:)),trace_size,1)  ...
%               ...
%             -(time_t'./Data.lambda(i-1,:)) +...
%              log((erf((constt +Data.t_p)/sqrt2sigma)-erf((constt -time_t' )/sqrt2sigma))./Data.lambda(i-1,:) +eps) ...
%               ...
%             -log(exp(-constt./(2*Data.lambda(i-1,:))).*erftmin_efttmax + ...
%              exp(-tmintp./Data.lambda(i-1,:)).*(erfc((constt-tmintp)/sqrt2sigma)-erfc((constt+Data.t_p)/sqrt2sigma)+eps)-...
%              exp(-tmaxtp./Data.lambda(i-1,:)).*(erfc((constt-tmaxtp)/sqrt2sigma)-erfc((constt+Data.t_p)/sqrt2sigma)+eps)+eps))'...
%                  ...
%                , trace_size )';

      Data.S(1,:) = Discrete_sampler_row( Probs, trace_size )';
      if max(Data.S(1,:)) > Data.Number_species
         Ind = find(Data.S()>Data.Number_species);
         Data.S(1,Ind) = ceil(Data.Number_species*rand(1,length(Ind)));
      end
           
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%    Sample the hyper-prior from the Drichlet    %%%%%%%%%%%%%
%%%%%%%%%%%%%%          ( The weights on species )            %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Data.PI(i,:) = dirichletRnd(Data.PI_alpha.*Data.PI_beta + histcounts(Data.S(1,:),1:Data.Number_species+1));
              
      
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%   Sample the molecular inverse lifetimes   %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      Data.lambda(i,:)=Data.lambda(i-1,:);
      % To have a better mixing we repeat this sampler multiple times
      for l=1:10
          [Data.lambda(i,:) ,  Data.acceptance_lambda] = Sample_lambda( ...
          ...
          Data.lambda(i,:)        , Data.S(1,:)          , time_t            , ...
          Data.acceptance_lambda  , Data.Number_species  , Data.Prop_lambda  , ...
          Data.alpha_lambda       , Data.beta_lambda     , Data              , ...
          sqrt2sigma              , erftmin_efttmax      , sigma_pp,    tmax ,   Ntmp);
      end
      
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%   Maximum Posterior   %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      [Data] = Max_Posterior(...
      ...
      Data        , Data.lambda(i,:)   , Data.PI(i,:)     , Data.S    , ...
      time_t      , sqrt2sigma         , Data.t_p         , sigma_pp  , ...
      Data.T_max  , Data.T_min         , erftmin_efttmax  , i         );
      
      
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%  Label switching   %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      %Data = Label_switching( Data , i);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%   Save the data   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if find((OLD_Iteration_size+floor((1:1:Data.Save_size).*Number_iter./Data.Save_size))==i)>=1
        save('Param','Data')
    end
    
    
end % End of the iteration loop



end % End of the function
