function [model,FeaModel] = runORL(x_t, fea, Musk, t, model, opt, FeaModel)


%% Feature Change Detection
FeaModel.Last    = FeaModel.Current;
FeaModel.Current = fea;
if ~isequal(FeaModel.Last,FeaModel.Current)
    FeaModel.Tran_emg = setdiff(FeaModel.Current,FeaModel.Last);
    FeaModel.Tran_etx = setdiff(FeaModel.Last,FeaModel.Current);
    disp(['*****Feature Varied from ', num2str(length(FeaModel.Last)),...
        ' to ', num2str(length(FeaModel.Current)),'.*****'])
    model.U = [model.U;rand(length(FeaModel.Tran_emg),opt.K)];
end

%% Subspace Estimator Adaptation
U_t   = model.U(FeaModel.Current,:);
Omega = diag(Musk);


%% V Updating
Vt = x_t'*Omega*U_t*pinv(U_t'*Omega*U_t + opt.lambda1*eye(opt.K,opt.K));

%% U Updating using Gradient Descent
for iter = 1:opt.IterNum
    g= Omega*(U_t*(Vt')*Vt - x_t*Vt) + opt.lambda2*U_t/t;
    U_t = U_t - opt.eta*g;
    U_t = NormalizeData(U_t,2);
end

%% Return
U(FeaModel.Current,:) = U_t;
model.U = U;
model.v = Vt;
        