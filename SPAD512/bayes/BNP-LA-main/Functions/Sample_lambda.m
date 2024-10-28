function[lambda ,  acceptance_lambda] = Sample_lambda(...
          ...
lambda_old         , S                , time_t       , ...
acceptance_lambda  , Num_species      , Prop_lambda  , ...
alpha_lambda       , beta_lambda      , Data         , ...
sqrt2sigma         , erftmin_efttmax  , sigma_pp     , tmax,    Ntmp)
  
  
lambda_new   = (lambda_old/Prop_lambda).*randg(Prop_lambda,1, Num_species ); 
Ind = lambda_new==0;
if sum(Ind) ~= 0
   lambda_new(Ind) = gamrnd(2,1);  
end
  
LikeOld  = calLikelihood(time_t,S,1./lambda_old,tmax,sigma_pp,Ntmp);
LikeProp = calLikelihood(time_t,S,1./lambda_new,tmax,sigma_pp,Ntmp);
Prior_Prop_Ratio = sum(((2*Prop_lambda-alpha_lambda)*log(lambda_old./lambda_new))+...
         ((lambda_old-lambda_new)/beta_lambda)+...
         (Prop_lambda*((lambda_new./lambda_old)-(lambda_old./lambda_new))));
logr = LikeProp - LikeOld + Prior_Prop_Ratio;

  % Accept or reject the proposals
     if  logr>log(rand()) || isnan(logr)
         lambda               = lambda_new            ;
         acceptance_lambda    = acceptance_lambda+1     ;
     else
         lambda               = lambda_old            ;
         acceptance_lambda(1) = acceptance_lambda(1)+1  ;
     end
end
    