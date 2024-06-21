function lifetime = pixel_BNP(raw, PhCount, Iter, RatioThresh, Number_species, PI_alpha, alpha_lambda, beta_lambda, freq, irf_mean, irf_sigma, save_size, step, offset)
%% BNP analysis of a single pixel from a time gated .tiff

    Dt = format_data(raw, step, offset);
    Data = initialize(Dt, PhCount, Number_species, PI_alpha, alpha_lambda, beta_lambda, freq, irf_mean, irf_sigma, save_size);
    
    tic; 
    Data = FLIM_Gibbs_sampler(Data, Iter); 
    elapsedTime = toc;
    fprintf('FLIM_Gibbs_sampler took %f seconds.\n', elapsedTime);

    lifetime = ev_lifetime(Data, RatioThresh);
    fprintf('Lifetime: %f\n', lifetime);
end
    
function [Dt] = format_data(raw, step, offset)
    bin_width = step; % bin spacing (gate step)
    start = offset; % bin start (relevant to gate width, offset)

    data = []; 
    for i = 1:length(raw)
        count = raw(i); 
        for j = 1:count
            data = [data, start + bin_width * (i - 1)];
        end
    end
    
    data = data(randperm(length(data))); % shuffle to unbias non-complete photon sets
    Dt = reshape(data, 1, []);
end

function Data = initialize(Dt, PhCount, Number_species, PI_alpha, alpha_lambda, beta_lambda, freq, irf_mean, irf_sigma, save_size)
    Data.T_max = (1e3/freq);  % interpulse window
    Data.T_min = 0;
    Data.t_p = irf_mean;  % IRF mean
    Data.sigma_p = irf_sigma;  % IRF stdev
    Data.delta_t = (1e3/freq); % interpulse window, probably redundant will check
    Data.Save_size = save_size;  % data save size
    Data.Number_species = Number_species;
    Data.PI_alpha = PI_alpha;
    Data.alpha_lambda = alpha_lambda;
    Data.beta_lambda = beta_lambda;
    Data.Prop_lambda = 1000; % lambda proposal
    Data.Ntmp = 5; % max possible pulse lag of photon detection
    
    Data.DtAll = Dt;
    Data.t_det = Dt(1:min(PhCount, length(Dt))); 

    Data.PI_beta = ones(1, Number_species) / Number_species; 
    Data.PI = dirichletRnd(Data.PI_beta * PI_alpha);
    Data.S = [];
    for k = 1:length(Data.t_det)
        Data.S(1,k) = Discrete_sampler(Data.PI);
    end
    Data.lambda = gamrnd(alpha_lambda, beta_lambda, 1, Number_species);
    Data.acceptance_lambda = [0; 0];
end

function lifetime = ev_lifetime(Data, Thresh)
    Iter = size(Data.lambda, 1);
    burn = floor(3 * Iter / 5);
    tau = Data.lambda(burn:end, :);
    PI = Data.PI(burn:end, :);
    ind = PI > Thresh;
    tmp = tau(ind);
    LifetimeThresh = round(prctile(tmp, 99));
    tmp = tmp(tmp < LifetimeThresh);
    lifetime = mean(tmp);
    
    figure;
    histogram(tmp, 'Normalization', 'pdf');
    xlabel('Lifetime (ns)');
    ylabel('Probability Density Function');
    title('Histogram of Lifetimes');
end
