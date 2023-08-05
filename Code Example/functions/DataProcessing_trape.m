function [Data, Label, Feature, Musk] = DataProcessing_trape(FileName,miss)

%% load data
load([FileName,'.mat'])
Label   = Y';

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