function run_BNP()
    % Configuration settings
    config.filename = '240604/240604_10ms_adjusted.tif';
    config.gate_num = 10;
    config.gate_step = 1.0;
    config.gate_offset = 0.0;
    config.thresh = 100;
    config.PhCount = 100;
    config.Iter = 1000;
    config.RatioThresh = 0.1;
    config.Number_species = 2;
    config.PI_alpha = 0.5;
    config.alpha_lambda = 2.0;
    config.beta_lambda = 2.0;
    config.freq = 80;
    config.irf_mean = 0.5;
    config.irf_sigma = 0.1;
    config.save_size = 100;
    config.gate_step = 0.01;
    config.gate_offset = 0;

    % Read the image
    image = imread(config.filename);
    [length, x, y] = size(image);
    intensity = zeros(x, y);
    tau = zeros(x, y);
    full_trace = zeros(length, 1);
    track = 0;
    
    % Initialize parallel pool
    parpool;

    parfor i = 1:x
        for j = 1:y
            trace = double(image(1:config.gate_num, i, j));
            if sum(trace) > config.thresh
                full_trace = full_trace + trace;
                intensity(i, j) = sum(trace);
                
                lifetime = pixel_BNP(trace, config.PhCount, config.Iter, config.RatioThresh, ...
                    config.Number_species, config.PI_alpha, config.alpha_lambda, ...
                    config.beta_lambda, config.freq, config.irf_mean, ...
                    config.irf_sigma, config.save_size, config.gate_step, config.gate_offset);

                tau(i, j) = lifetime;
                track = track + 1;
            end
        end
    end

    % Save results
    save_results(config.filename, intensity, tau, full_trace, track);
end

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
    % Check the type of raw and attempt to convert if not numeric
    if ~isa(raw, 'numeric')
        try
            raw = str2num(raw); %#ok<ST2NM>
            if isempty(raw)
                error('raw not convertible empty');
            end
        catch
            error('raw not convertible caught');
        end
    end
    
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
    
    % figure;
    % histogram(tmp, 'Normalization', 'pdf');
    % xlabel('Lifetime (ns)');
    % ylabel('Probability Density Function');
    % title('Histogram of Lifetimes');
end

function save_results(filename, intensity, tau, full_trace, track)
    save([filename '_bnp_results.mat'], 'intensity', 'tau', 'full_trace', 'track');
end
