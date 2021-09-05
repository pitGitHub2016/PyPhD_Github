cd('D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_FOREX');
% Main Dataset
%load('FxDataAdjRets_0.mat'); %y = 100 * y; Ytrain = y;
%% Delayed Dataset
Ytrain1 = Ytrain;
delays = 250; % 20, 250
for i=1:delays; Ydelayed{i} = lagmatrix(Ytrain1,i); end
Ytrain = Ytrain1; 
for j=1:delays; Ytrain = [Ytrain, Ydelayed{j}]; end
Ytrain = rmmissing(Ytrain); YtrainLift = rmmissing(Ydelayed{delays});
%save(strcat('FxDataAdjRetsDelay_1.mat'));
save(strcat('FxDataAdjRetsMAJORSDelay_1.mat'));
%% MAJORS
Ytrain = Ytrain(:,1:7);
%% FOREX Tester PnL
Ytrain = 100 * Ytrain;
forecastpoints = 1000;
train_last_point = size(Ytrain,1)-forecastpoints;

trainset = Ytrain(1:train_last_point,:);
testset = Ytrain(train_last_point+1:end,:);

RW = fillmissing(lagmatrix(testset,1),'constant', 0) .* testset;
plot(cs(rs(RW)));
sqrt(252) * sharpe(rs(RW),0)