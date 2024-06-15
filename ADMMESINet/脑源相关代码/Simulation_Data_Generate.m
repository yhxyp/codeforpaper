function [B,Result] = Simulation_Data_Generate (LFM,Cortex,Time,OPTIONS)
% Descriptions: Genarate simulated EEG/MEG data for extended sources
% Inputs: LFM: Lead field Matrix(#sensors X #sources)
%         Cortex.| Vertices
%               .| Faces    Cortex files
%         Time: Time for each sample
%         OPTIONS. DefinedArea: Areas for each patch
%                . seedvox    : seedvoxs of each patch
%                . frequency  : frequency of the gaussian damped time courses
%                . tau        : time delay of the gaussian damped time courses
%                . omega      : variation of the gaussian damped time courses
%                . Amp        : Amplitude of the gaussian damped time courses
%                . uniform    : applying uniform activations (uniform = 1) or not (uniform = 0)
%                . WGN        : adding white gaussian noise (WGN = 1) or not (WGN = 0)

% Version 1: Liu Ke, 2018/8/22
%% ===== DEFINE DEFAULT OPTIONS =====
Def_OPTIONS.WGN         = 1;
Def_OPTIONS.uniform     = 1;
Def_OPTIONS.ar          = 0;
Def_OPTIONS.GridLoc     = [];%GridLoc 每个体素的位置坐标
Def_OPTIONS.MixMatrix = eye(4);%eye(numel(tau));时间延迟
% Copy default options to OPTIONS structure (do not replace defined values)
OPTIONS = struct_copy_fields(OPTIONS, Def_OPTIONS, 0);

GridLoc = OPTIONS.GridLoc;
AreaDef = OPTIONS.DefinedArea;
seedvox = OPTIONS.seedvox;
f = OPTIONS.frequency;
tau = OPTIONS.tau;
omega = OPTIONS.omega;
A = OPTIONS.Amp;%能量
Uniform = OPTIONS.uniform;
WGN = OPTIONS.WGN;
SNR = OPTIONS.SNR;
ar = OPTIONS.ar;%？？？
MixMatrix = OPTIONS.MixMatrix;
nSource = size(LFM,2);%导联矩阵的行数，源数量
%% Active Vox    
ActiveVoxSeed = num2cell(seedvox);
ActiveVox = [];
[~, VertArea] = tess_area(Cortex.Vertices, Cortex.Faces);
Cortex.VertConn = tess_vertconn(Cortex.Vertices, Cortex.Faces);
for k = 1:numel(seedvox)
    ActiveVoxSeed{k} = PatchGenerate(seedvox(k),Cortex.VertConn,VertArea,AreaDef(k));
    ActiveVox = union(ActiveVoxSeed{k},ActiveVox);
end
Area = sum(VertArea(ActiveVox));
%% ------------------ Simulation data ---------------------%
StimTime = find(abs(Time) == min(abs(Time)));
x = zeros(nSource,numel(Time));
Activetime = StimTime+1:numel(Time);
% -----------Gaussian Damped sinusoidal time courses------------------%
if ~ar
    Basis = zeros(numel(tau),numel(Time));
    for k = 1:numel(tau)
        Basis(k,Activetime) = sin(2*pi*f(k)*(Time(Activetime))).*exp(-((Time(Activetime)-tau(k))/omega(k)).^2);
    end
    % Basis = Basis./repmat(max(Basis')',1,size(Basis,2));
    Basis = orth(Basis')';
    %Basis = Basis./repmat(sqrt(sum(Basis.^2,2)),1,size(Basis,2));
%     Basis = Basis./repmat(max(Basis')',1,size(Basis,2));
    AA = MixMatrix*A;
    % % ========================Uniform/NonUniform Sources ==============================%
    if Uniform
        for k = 1:numel(seedvox)
            x(ActiveVoxSeed{k},:) = repmat(AA(k,:)*Basis,numel(ActiveVoxSeed{k}),1);
        end
    else
        for k = 1:numel(seedvox)
            %                x(ActiveVoxSeed{k},:) = 1e6*VertArea(ActiveVoxSeed{k})*AA(k,:)*Basis;
            dis = sqrt(sum((GridLoc(ActiveVoxSeed{k},:)-repmat(GridLoc(seedvox(k),:),numel(ActiveVoxSeed{k}),1)).^2,2));
            Amplitude = exp(-dis.^2./(0.6*max(dis)).^2);
            x(ActiveVoxSeed{k},:) = Amplitude*AA(k,:)*Basis;
        end
    end
    % % ======================================================================%
    
    % %===================== Current Density Model (CDM)======================%
    %     for i =1 :numel(ActiveVox)
    %x(ActiveVox(i),Activetime) = 1e6*VertArea(ActiveVox(i))*1e-10*sin(2*pi*f*Time(Activetime));
    %     end
%% AR time series    
else
    cfg             = [];
    cfg.ntrials     = 1;
    cfg.triallength = max(Time);
    cfg.fsample     = 1/(Time(2) - Time(1));
    cfg.nsignal     = size(OPTIONS.noisecov,1);
    cfg.method      = 'ar';
    
    cfg.params      = OPTIONS.params;
    cfg.noisecov    = OPTIONS.noisecov;
    
    artimeseries              = ft_connectivitysimulation(cfg);
    x_active = zeros(numel(seedvox),numel(Time));
    x_active(:,Activetime) = artimeseries.trial{1}(1:numel(seedvox),:);
%     for i = 1:size(x_active,1)
%    x_active(i,Activetime) = ft_preproc_bandpassfilter(artimeseries(i,:), size(artimeseries(i,:),2), [1 30]);
%     end
    for k = 1:numel(seedvox)
        x(ActiveVoxSeed{k},:) = A*repmat(x_active(k,:),numel(ActiveVoxSeed{k}),1);
    end
end
% % ======================================================================%

% ===================== Realistic EEG background noise ===================%
%load 'HumanEEGNoise.mat'
if  ~WGN
    load 'ArtHumanEEGNoise.mat'
    noise = noise(:,1:TimeLen);
    r = norm(LFM*x,'fro')/norm(noise,'fro')/10^(SNR/20);
    Noise = r*noise;
    B = Gain*x + Noise;
else
    % %==================== White Gaussian Noise ============================= %
    B = awgn(LFM*x,SNR,'measured');
    
end
% =======================================================================%
Result.B = B;
Result.seedvox = seedvox;
Result.ActiveVox = ActiveVox;
Result.ActiveVoxSeed = ActiveVoxSeed;
Result.Area = Area;
 
