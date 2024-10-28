function [guiFig,Data] = gui()
%% Open gui figure

    guiFig = figure('NumberTitle','off','Resize','off','Units','Pixels',...
             'MenuBar','none','Name','BNP-LA user interface','Visible','on',...
             'Interruptible','off','Position',[500,1000,700,450]);
         
%% Construct components

    handles.output = guiFig;
    guidata(guiFig,handles);

%% Set Parameters          

    Data.T_max = 12.8;  %Interpulse window;
    Data.T_min = 0;
    Data.t_p = 12.2;  %The mean value of the IRF
    Data.sigma_p = 0.66;  %The standar deviation of IRF
    Data.delta_t = 12.8; %Interpulse window;
    Data.Save_size = 5;  % The save size of the data
    Data.Number_species = 5;  % Number of model species
    Data.PI_alpha = 1;  % Alpha parameter of prior on PI (weight on species)
    Data.alpha_lambda = 1; % Alpha parameter of prior on lambda (lifetimes)
    Data.beta_lambda = 50; % Betal parameter of prior on lambda (lifetimes)
    Data.Prop_lambda = 1000; % Proposal parameter of lambda (lifetimes)
    Data.Ntmp = 5; % Numbrt of empty pulses to be considered
    
%% Experimental parameters  

    ExpParams=uipanel('Parent',guiFig,'Title','Exp Params','Position',[0.025 0.55 0.47 0.4]);
    uicontrol('Parent',ExpParams,'Style','text','String','Interpulse time (ns)',...
        'Position',[5,110,190,25]);
    handles.Interpulse = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 110 100 25],...
        'String','12.8','Callback',@getInterpulse);
    uicontrol('Parent',ExpParams,'Style','text','String','IRF mean (ns)',...
        'Position',[5,70,180,25]);
    handles.IRFmean = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 70 100 25],...
        'String','12.2','Callback',@getIRFmean);
    uicontrol('Parent',ExpParams,'Style','text','String','IRF sigma (ns)',...
        'Position',[5,30,180,25]);
    handles.IRFSig = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 30 100 25],...
        'String','0.66','Callback',@getIRFsigma);
    
%% Algorithm Parameters

    ExpParams = uipanel('Parent',guiFig,'Title','Method Params','Position',[0.51 0.475 0.47 0.475]);
    uicontrol('Parent',ExpParams,'Style','text','String','Max Species',...
        'Position',[5,140,190,25]);
    handles.MaxSpec = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 140 100 25],...
        'String','5','Callback',@getMaxSpec);
    uicontrol('Parent',ExpParams,'Style','text','String','Max Pulses',...
        'Position',[5,100,180,25]);
    handles.MaxN = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 100 100 25],...
        'String','5','Callback',@getMaxPulse);
    uicontrol('Parent',ExpParams,'Style','text','String','Alpha',...
        'Position',[5,60,180,25]);
    handles.Alpha = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 60 100 25],...
        'String','1','Callback',@getAlpha);
    uicontrol('Parent',ExpParams,'Style','text','String','Beta',...
        'Position',[5,20,180,25]);
    handles.Beta = uicontrol('Parent',ExpParams,'Style','edit','Position',[210 20 100 25],...
        'String','50','Callback',@getBeta);
    
%% Results

    Results = uipanel('Parent',guiFig,'Title','Results','Position',[0.51 0.05 0.47 0.4]);
    uicontrol('Parent',Results,'Style','text','String','Photon Ratio Threshold',...
        'Position',[5,110,250,25]);
    handles.RatioThresh = uicontrol('Parent',Results,'Style','edit','Position',[255 110 50 25],...
        'String','0.2','Callback',@getRatioThresh);
    uicontrol('Parent',Results,'Style','text','String','Parameter',...
        'Position',[20,70,100,25]);
    handle.HistParam = uicontrol('Parent',Results,'Style','pop','String',...
        {'Plot Ratio','Plot Lifetime','Hist Lifetime','Hist Ratio'}, ...
        'Position',[150 70 150 25]);
    uicontrol('Parent',Results,'Style','pushbutton','String','PLOT',...
        'Position',[95 15 150 30],'Backgroundcolor',[0 0.45 0.75],'Callback',@makePlot)
    
%% Data

    uicontrol('Parent',guiFig,'Style','text','String','Photon Count', ...
        'Position',[50 200 130 25])
    handles.PhCount = uicontrol('Parent',guiFig,'Style','edit','String','5000',...
        'Position',[230 200 100 25]);
    uicontrol('Parent',guiFig,'Style','text','String','Iterations', ...
        'Position',[50 160 130 25]);
    handles.Iter = uicontrol('Parent',guiFig,'Style','edit','String','2500',...
        'Position',[230 160 100 25]);
    uicontrol('Parent',guiFig,'Style','pushbutton','String','LOAD DATA',...
        'Position',[80 100 200 30],'Callback',@loadData)
    uicontrol('Parent',guiFig,'Style','pushbutton','String','RUN',...
        'Position',[120 30 100 50],'Backgroundcolor','green','Callback',@runCode)
    
%% Functions

    function getInterpulse(~,~)
        Interpulse = get(handles.Interpulse,'String');
        Data.T_max = str2double(Interpulse);
        Data.delta_t = str2double(Interpulse);
    end

    function getIRFmean(~,~)
        IRFmean = get(handles.IRFmean,'String');
        Data.t_p = str2double(IRFmean);
    end

    function getIRFsigma(~,~)
        IRFSig = get(handles.IRFSig,'String');
        Data.sigma_p = str2double(IRFSig);
    end

    function getMaxSpec(~,~)
        MaxSpec = get(handles.MaxSpec,'String');
        Data.Save_size = str2double(MaxSpec);  
        Data.Number_species = str2double(MaxSpec); 
    end

    function getMaxPulse(~,~)
        MaxN = get(handles.MaxN,'String');
        Data.Ntmp = str2double(MaxN);
    end

    function getAlpha(~,~)
        Alpha = get(handles.Alpha,'String'); 
        Data.alpha_lambda = str2double(Alpha);
    end

    function getBeta(~,~)
        Beta = get(handles.Beta,'String'); 
        Data.beta_lambda = str2double(Beta);
    end

    function makePlot(~,~)
        tmp = get(handle.HistParam,'Value');
        StringCase = get(handle.HistParam,'String');
        PlotCase = StringCase{tmp};
        Thresh = str2double(get(handles.RatioThresh,'String'));
        Iter = size(Data.lambda,1);
        Burnin = floor(3*Iter/5);
        Tau = Data.lambda(Burnin:end,:);
        PI = Data.PI(Burnin:end,:);
        switch PlotCase
            case 'Hist Lifetime'
                Ind = PI > Thresh;
                tmp = Tau(Ind);
                LifetimeThresh = round(prctile(tmp,99));
                tmp = tmp(tmp<LifetimeThresh);
                figure;histogram(tmp,'normalization','pdf')
                xlabel('lifetime (ns)');ylabel('pdf')
            case 'Hist Ratio'
                Ind = PI > Thresh;
                tmp = PI(Ind);
                figure;histogram(tmp,'normalization','pdf')
                xlabel('photon ratios');ylabel('pdf')
            case 'Plot Lifetime'
                figure;plot(Data.lambda)
                xlabel('samples');ylabel('lifetime (ns)')
            case 'Plot Ratio'
                figure;plot(Data.PI)
                xlabel('samples');ylabel('photon ratios')
        end
    end

    function loadData(~,~)
        [filename, pathname]=uigetfile(pwd,'\*.mat;*.ics;*.h5');
        load(fullfile(pathname,filename),'Dt')
        if size(Dt,1) == 1
            Data.DtAll = Dt;
        else
            Data.DtAll = Dt';
        end
    end

    function runCode(~,~)
        %Data
        if ~isfield(Data,'DtAll')
           error('No data has been loaded yet.') 
        end
        tmp = get(handles.PhCount,'String');
        K = str2double(tmp);
        Data.t_det = Data.DtAll(1:K);
        % Initailzation the values
        Data.PI_beta = ones(1,Data.Number_species)./...
                      Data.Number_species ;  % Beta parameter of prior on PI (weight on species)
        Data.PI = dirichletRnd( Data.PI_beta*Data.PI_alpha );
        Data.S = [];
        for k = 1:length(Data.t_det)
            Data.S(1,k) = Discrete_sampler( Data.PI );
        end
        Data.lambda = gamrnd(Data.alpha_lambda , Data.beta_lambda ,1,Data.Number_species) ;
        % Initialize the acceptance rate
        Data.acceptance_lambda = [0;0];
        
        tmp = get(handles.Iter,'String');
        Iter = str2double(tmp);
        
        tic;
        [Data] = FLIM_Gibbs_sampler( Data , Iter  );
        T = toc;
        fprintf('It took %fs to finish the run.\n',T)
    end

end
