%% Generate Data
% This script generates the simulated data set given in the Data-directory
% Every parameter used in data generation are saved under a structure "Data"

Data.Number_pulse            = 3*10^5                      ; % Number of laser pulses used to illuminate the sample (larger values give more photons)
Data.delta_t                 = 12.80                       ; % Interpulse window (ns) (inverse of laser pulse frequency)

Data.emission_species        =  [0.2 0.6]                  ; % Species Lifetimes (ns)
Data.excitation_species      =  0.008*[1 1]                ; % Species excitation rates (when the sum is normalized to one gives photon ratios)
Data.emission_back           =    10^-16                   ; % Background photon emission rate (set to essentially zero here)

Data.t_p                     =  12.2                       ; % IRF mean (ns)
Data.sigma_p                 =  0.66                       ; % IRF standard deviation (ns)

% Generate the synthetic data
Data = Generative(Data);
Dt = Data.t_det;

figure;
histogram(Dt ,256, 'Normalization','pdf')
xlim([Data.T_min , Data.T_max])
xlabel('Prob. distr. func.')

save('Data_Lifetimes_point2&point6_DefaultParams','Dt')
