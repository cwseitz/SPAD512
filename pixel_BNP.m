function lifetime = pixel_BNP(file, x, y)

%% BNP analysis of a single pixel from a time gated .tiff

    PhCount = 5000; % number of photons to use
    Iter = 2500; % gibbs iterations
    RatioThresh = 0.2; % ratio threshold for plotting/EV

    Number_species = 5;  % DP max species
    PI_alpha = 1;  % alpha prior param on PI (species)
    alpha_lambda = 1; % alpha prior param on lambda (lifetimes)
    beta_lambda = 50; % beta prior param lambda

    tiffFile = Tiff(file, 'r');
    info = imfinfo(file);
    numFrames = numel(info); 
    numRows = info(1).Height;
    numCols = info(1).Width;
    image = zeros(numRows, numCols, numFrames, 'uint16'); 

    for k = 1:numFrames
        tiffFile.setDirectory(k);
        image(:, :, k) = tiffFile.read();
    end
    tiffFile.close();

    raw = image(y, x, :);  
    raw = squeeze(raw); 
    Dt = format(raw);
    Data = initialize(Dt, PhCount, Number_species, PI_alpha, alpha_lambda, beta_lambda);
    
    tic; 
    Data = FLIM_Gibbs_sampler(Data, Iter); 
    elapsedTime = toc;
    fprintf('FLIM_Gibbs_sampler took %f seconds.\n', elapsedTime);

    lifetime = ev_lifetime(Data, RatioThresh);
    fprintf('Lifetime: %f\n', lifetime);
end

function [Dt] = format(raw)

%% Turn raw binned photon arrival info into a list of photon arrival times

    bin_width = 0.09; % bin spacing (gate step)
    start = 0.018; % bin start (relevant to gate width, offset)

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

function Data = initialize(Dt, PhCount, Number_species, PI_alpha, alpha_lambda, beta_lambda)

%% Set up Data field for running BNPs

    Data.T_max = 100;  % interpulse window
    Data.T_min = 0;
    Data.t_p = 0;  % IRF mean
    Data.sigma_p = 0;  % IRF stdev
    Data.delta_t = 100; % interpulse window, probably redundant will check
    Data.Save_size = 5;  % data save size
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
            Data.S(1,k) = Discrete_sampler( Data.PI );
        end
    Data.lambda = gamrnd(alpha_lambda, beta_lambda, 1, Number_species);
    Data.acceptance_lambda = [0; 0];
end

function lifetime = ev_lifetime(Data, Thresh)

%% Plot lifetime in a histogram, and report mean value

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
