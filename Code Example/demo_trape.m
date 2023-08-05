clear all
close all
addpath(genpath('.'));

%% Load Data
% X: 1*N cell; each cell contains an instance
% Y: 1*N vector;
% Musk: 1*N cell; each cell contains an indicator vector for missing
miss.type     = 'R';% C-continous missing, R-random missing
miss.rate     = 0.3; % 0-1
FileName      = 'bc_mushroom_pro_pro';
[X,Y,Fea,Musk]     = DataProcessing_trape(FileName,miss);


%% Parameter Settings
Opt.K       = 25; % Subspace Dimension
Opt.eta     = 0.1; % Learning Rate
Opt.IterNum = 5; % Maximum Iteration
Opt.lambda1 = 0.01;
Opt.lambda2 = 0.001;


%% Model Initilization
% model.U - D*K Subspace Estimator
% model.V - 1*K Representations
% Model.w - 1*K Classifier
FeaModel.Current = Fea{1};
Model.U = rand(length(FeaModel.Current),Opt.K);
Model.V = [];
Model.w = zeros(1,Opt.K);


%% Online Learning
[M,N]     = size(X);
for t = 1:N
    % Online Representation Learning
    [Model, FeaModel] = runORL(X{t}, Fea{t}, Musk{t}, t, Model, Opt, FeaModel);
    % Online Classifier Learning 
    [Model, result.y_hat(t)] = runOCL(Model.V, Y(t), t, Model, Opt);
end

%% Performance Calculating
Metric.ErrNum = sum(result.y_hat~=Y);

