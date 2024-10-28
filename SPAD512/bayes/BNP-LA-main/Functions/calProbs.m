function [Probs,LikeOut] = calProbs(Data,Tmax,Sig_IRF,Lambda,Pi_T,Ntmp)
%calProbs finds probabilities used in categorical distribution to sample S
%
%INPUT:
%   Data: Structure array containing photon arrival times (ns)
%   EmpParam: Structure containing parameters of the experiment
%   Lambda: Sampled lambda in the previous iteration (1/ns)
%   Pi_T: Relative probability of photons coming from different species
%
%OUTPUT:
%   Probs: The probabilities of a given photon coming from different species
%
%Created by:
% Mohamadreza Fazel (Presse Lab, 2022)
%

if size(Lambda,2) == 1
    Lambda = Lambda';
end
LambdaS = repmat(Lambda,[length(Data),1,Ntmp+1]);
if size(Data,1) == 1
   Data = Data';
end
DataS = repmat(Data,[1,length(Lambda),Ntmp+1]);
Nt = 0:Ntmp;
Nt = reshape(Nt,[1,1,length(Nt)]);
NS = repmat(Nt,[length(Data),length(Lambda),1]);
LikeExp = (LambdaS/2).*exp((LambdaS/2).*(2*(DataS-NS*Tmax) + ...
        LambdaS*(Sig_IRF^2))); 
LikeErf = erfc((DataS-NS*Tmax+LambdaS*Sig_IRF.^2) ...
        /(sqrt(2)*Sig_IRF));    
LikeOut = sum(LikeExp.*LikeErf,3); 
        
PiT = repmat(Pi_T,[length(Data),1]);
Probs1 = LikeOut.*PiT+eps;
NScale = repmat(sum(Probs1,2),[1,length(Lambda)]); 
Probs = Probs1./NScale;

end