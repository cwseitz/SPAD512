function LikeOut = calLikelihood(Data,S,Lambda,T,Sig_IRF,Ntmp)
%calLikelihood finds the likelihood of the proposed lamda
%
%INPUT:
%   Data: Structure array containing photon arrival times (ns)
%   EmpParam: Structure containing parameters of the experiment
%   S:  Sampled S in the previous iteration
%   Lambda: Proposed lambda (1/ns)
%
%OUTPUT:
%   Likelihood: Calculated likelihood of the proposed lambda
%
%Created by:
%   Mohamadreza Fazel (Presse Lab, 2022)
%

if length(Lambda)==1
    LambdaS = Lambda*ones(length(S),1);
else
    LambdaS = Lambda(S)';
end

if size(LambdaS,2) ~= 1
   LambdaS = LambdaS'; 
end
if size(Data,2) ~= 1
   Data = Data'; 
end

LambdaT = repmat(LambdaS,[1,Ntmp+1]);
DataT = repmat(Data,[1,Ntmp+1]);
NT = repmat((0:Ntmp),[length(Data),1]);
  

LikeExp = (LambdaT/2).*exp((LambdaT/2).*(2*(DataT-NT*T) + ...
        LambdaT*Sig_IRF.^2));
LikeErf = erfc((DataT-NT*T+LambdaT*Sig_IRF.^2) ...
        /(sqrt(2)*Sig_IRF)); 

tmpLike = LikeExp.*LikeErf;
tmpLike(isnan(tmpLike)) = 0;
LikeOut = sum(tmpLike,2);
%LikeOut(isnan(LikeOut)) = 0;
if sum(isnan(LikeOut))>0
    a = 0;
elseif sum(isinf(LikeOut))>0
    a = 1;
end
% LikeOut(isnan(LikeOut)) = [];
% LikeOut(isinf(LikeOut)) = [];
LikeOut = sum(log(LikeOut));

end
