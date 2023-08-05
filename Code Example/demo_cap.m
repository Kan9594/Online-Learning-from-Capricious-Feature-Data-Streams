clear all
close all
addpath(genpath('.'));

%% Load Data
% X: 1*N cell; each cell contains an instance
% Y: 1*N vector;
% M: 1*N cell; each cell contains an indicator vector for missing
miss.type     = 'R';% C-continous missing, R-random missing
miss.rate     = 0.3;
FileName      = 'bc_mushroom_pro_pro';
[X,Y,Fea,Musk]     = DataProcessing_cap(FileName,miss);


%% Parameter Settings
Opt.K       = 25; % Subspace Dimension
Opt.eta     = 0.1; % Learning Rate
Opt.IterNum = 5; % Maximum Iteration
Opt.lambda1 = 0.01;
Opt.lambda2 = 0.001;


%% Model Initilization
% ORL:  ||p(x_t) - P(U*v_t')||_F^2
% OCL:  max(0,1-y_t w_t v_t')
% model.U - D*K Subspace Estimator
% model.v - 1*K Representations
% Model.w - 1*K Classifier
FeaModel.Current = Fea{1};
Model.U = rand(length(FeaModel.Current),Opt.K);
Model.v = [];
Model.w = zeros(1,Opt.K);


%% Online Learning
[M,N]     = size(X);
for t = 1:N
    % Online Representation Learning
    [Model, FeaModel] = runORL(X{t}, Fea{t}, Musk{t}, t, Model, Opt, FeaModel);
    % Online Classifier Learning 
    [Model, result.y_hat(t)] = ...
        runOCL(Model.v, Y(t), t, Model, Opt);
end

%% Performance Calculating
Metric.ErrNum = sum(result.y_hat~=Y);

