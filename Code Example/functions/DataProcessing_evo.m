function [Data, Label, Feature, Musk] = DataProcessing_evo(FileName,miss)

%% load data
load([FileName,'.mat'])
Label   = Y';

X_ori = NormalizeData(X_ori,2);
X_ori = NormalizeData(X_ori,1);
X_syn = NormalizeData(X_syn,2);
X_syn = NormalizeData(X_syn,1);
X_ori_temp = X_ori;
X_ori = X_syn;rand(size(X_syn));
X_syn = X_ori_temp;
T_step = floor(linspace(1,length(Y),4));
T_step(2) = floor(T_step(2)/2);
T_step(3) = T_step(2)+50;
X1 = X_ori(1:T_step(2),:)';
Y1 = Y(1:T_step(2));
X2 = [X_ori(T_step(2)+1:T_step(3),:),X_syn(T_step(2)+1:T_step(3),:)]';
Y2 = Y(T_step(2)+1:T_step(3));
X3 = X_syn(T_step(3)+1:T_step(4),:)';
Y3 = Y(T_step(3)+1:T_step(4));
X1_1 = [X1;nan(size(X2,1)-size(X1,1),size(X1,2))];
X3_1 = [nan(size(X2,1)-size(X3,1),size(X3,2));X3];
X_trape = [X1_1,X2,X3_1];
Y = [Y1;Y2;Y3];

%% Feature Record
[~,N] = size(X_trape);
for t = 1:N
    Feature{t} = find(~isnan(X_trape(:,t)));
end


%% assign miss
Fea_t = [];

if miss.rate ~= 0
    switch miss.type
        case 'R'
            for t = 1:N
                Fea_t = find(~isnan(X_trape(:,t)));
                Musk_t = ones(size(Fea_t));
                Musk_Ind = randperm(length(Fea_t));
                Musk_t(Musk_Ind(1:floor(length(Fea_t)*miss.rate))) = 0;
                Musk{t} = Musk_t;
                Data{t} = X_trape(Fea_t,t).*Musk_t;
            end
        case 'C'
            for t = 1:N
                Fea_old = Fea_t;
                Fea_t  = find(~isnan(X_trape(:,t)));
                if length(Fea_t) ~= length(Fea_old)
                    Flag = 1;
                else
                    Flag = 0;
                end
                if ~rem(t,100) || Flag==1 || t==1
                    Musk_t = ones(size(Fea_t));
                    Musk_Ind = randperm(length( Fea_t));
                    Musk_t(Musk_Ind(1:floor(length( Fea_t)*miss.rate))) = 0;
                    Musk{t} = Musk_t;
                    Data{t} = X_trape(Fea_t,t).*Musk_t;
                else
                    Musk{t} = Musk_t;
                    Data{t} = X_trape(Fea_t,t).*Musk_t;
                end
            end
    end
end