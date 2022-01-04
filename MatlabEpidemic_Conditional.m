%% Matlab Epidemic
dataPath = 'D:\Dropbox\EpidemicModel\WorkingDataset\'; cd('D:\Dropbox\EpidemicModel\WorkingDataset\MatlabEpidemic'); parentFiguresFolder = 'D:\Dropbox\EpidemicModel\WorkingDataset\MatlabEpidemic\MatlabFigures\';
[totale_positivi,txt,raw] = xlsread(strcat(dataPath, 'totale_positivi.xlsx')); datasetID = 'PP'; datasetIDNum = 0; diffFlag = 0; mat = txt(1,2:end); mat = mat';
%totale_positivi = load('D:\Dropbox\EpidemicModel\WorkingDataset\total_positive_siettos.mat'); diffFlag = 0; totale_positivi = totale_positivi.C; datasetID = 'Siettos'; datasetIDNum = 1; mat = dir('D:\Dropbox\EpidemicModel\WorkingDataset\MatlabEpidemic\SiettosCode\*.mat'); 
st = 60; 
%totale_positivi = CleanOutliers(totale_positivi, st);
whichRegions = 1; %[1,2,15]--> Siettos failures : 4 = Calambria in siettos -->13 in mine
for roundCount = 1:length(whichRegions)
    targetRegion = whichRegions(roundCount);
    clc; close all; clf; close all;
    clearvars -except targetRegion mat roundCount dataPath parentFiguresFolder totale_positivi diffFlag whichRegions st datasetID datasetIDNum predStepReport
    
    if datasetIDNum == 0
        targetRegionName = strrep(mat(targetRegion),'totale_positivi_','');
        targetRegionName = targetRegionName{1,1};
    else
        targetRegionName = strrep(mat(targetRegion).name,'.mat','');
    end
    %yMat = totale_positivi;
    yMat = movavg(totale_positivi,'exponential', 5); %exponential
    %yMat = [zeros(1,size(totale_positivi,2));diff(totale_positivi)]; diffFlag = 1; %1
    externalX = yMat;
    
    % Shift inputs
    delaysUsed = [1,2,4,6,8]; xMatFlag = 0; %[1,2],[1,2,4,6,8],[1,2,5,10,15,20,30,50]
    %externalX = load('D:\Dropbox\EpidemicModel\WorkingDataset\total_siettos.mat'); diffFlag = 0; externalX = externalX.D;
    %externalX = xlsread(strcat(dataPath, 'dfAll_toMatlab.xlsx')); 
    %externalX = xlsread(strcat(dataPath, 'nuovi_positivi.xlsx')); externalX = [yMat, externalX];

    xMat = getShifts(externalX, delaysUsed); xMatFlag = 1;
    %xMat = getShifts([zeros(1,size(externalX,2));diff(externalX)], delaysUsed); xMatFlag = 1;

    % Rolling Predict
    modelSelection = 1; % 1 = GPR, 2 = LSTM
    predictAhead = 5;
    predictMode = 3; % 1 = one-step-ahead(not the target), 2 = multi-step-ahead, 3 = conditioned

    for i = st:size(yMat,1)-predictAhead%st+5
        i
        %intv = 1:i;
        intv = i-st+1:i;
        roll_Y_target = yMat(intv,targetRegion);
        
        if xMatFlag == 1
            roll_X_input = xMat(intv,:);
        else
            roll_X_input = getShifts(roll_Y_target, delaysUsed);
        end

        %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        %trainX_mu =  0; trainX_std = 1;
        trainX_mu =  mean(roll_X_input); trainX_std = std(roll_X_input);
        roll_X_input_normed = fillmissing((roll_X_input - trainX_mu) ./ trainX_std, 'constant', trainX_mu);

        %targetY_mu = 0; targetY_std = 1;
        targetY_mu = mean(roll_Y_target); targetY_std = std(roll_Y_target);
        roll_Y_target_normed = fillmissing((roll_Y_target - targetY_mu) ./ targetY_std, 'constant', targetY_mu);
        %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        if modelSelection == 1
            %kernel = optimizableVariable('KernelFunction',{'exponential','squaredexponential',
            %'matern32','matern52', 'rationalquadratic','ardexponential','ardsquaredexponential',
            %'ardmatern32','ardmatern52','ardrationalquadratic'},'Type','categorical');
            %sigma = optimizableVariable('Sigma',[1e-4,10],'Transform','log');
            %bo = bayesopt(@(T)objFcn(T,roll_X_input_normed,roll_Y_target_normed), [sigma, kernel], 'MaxObjectiveEvaluations', 100);           
            %kernel_opt= bo.XAtMinObjective.KernelFunction;
            %gprMdl = fitrgp(roll_X_input_normed, roll_Y_target_normed, 'KernelFunction', char(kernel));
            
            gprMdl = fitrgp(roll_X_input_normed, roll_Y_target_normed, 'KernelFunction','matern32');%,...
               %'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',...
               %struct('AcquisitionFunctionName','expected-improvement-plus'));
               %clf;close all;        
               
            if predictMode == 1
                rawPred = predict(gprMdl,xMat(i+1,:));
                ypred{i+1,targetRegion} = num2cell(targetY_std*rawPred + targetY_mu);
            elseif predictMode == 2
                [out, out_lower, out_upper] = iterPredict_GPR(gprMdl, roll_Y_target_normed, delaysUsed, predictAhead, targetY_mu, targetY_std);
                ypred{i+1,targetRegion} = num2cell(out);
                ypred_lower{i+1,targetRegion} = num2cell(out_lower);
                ypred_upper{i+1,targetRegion} = num2cell(out_upper);
            elseif predictMode == 3
                xConditioned = xMat(i+1:i+predictAhead,:);
                xConditioned_normed = fillmissing((xConditioned - trainX_mu) ./ trainX_std, 'constant', trainX_mu);
                [out, out_std, out_int] = predict(gprMdl,xConditioned_normed); 
                % de-normalise
                for k = 1:length(out); out(k) = targetY_std * out(k)+ targetY_mu; end
                for k = 1:length(out_int); out_lower(k) = targetY_std * out_int(k,1)+ targetY_mu; end
                for k = 1:length(out_int); out_upper(k) = targetY_std * out_int(k,2)+ targetY_mu; end
                ypred{i+1,targetRegion} = num2cell(out);
                ypred_lower{i+1,targetRegion} = num2cell(out_lower);
                ypred_upper{i+1,targetRegion} = num2cell(out_upper);
            end

        elseif modelSelection == 2

            input = num2cell(roll_X_input_normed',1);
            output = num2cell(roll_Y_target_normed',1);

            numResponses = size(output{1},1);
            featureDimension = size(input{1},1);
            numHiddenUnits = 7;

            layers = [ ...
                sequenceInputLayer(featureDimension)
                lstmLayer(numHiddenUnits,'OutputMode','sequence')
                fullyConnectedLayer(numResponses)
                regressionLayer];

            options = trainingOptions('adam', ...
            'MaxEpochs',100, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',0.005, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, 'Verbose',0);
            %, ...
            %'Plots','training-progress');

            net = trainNetwork(input,output,layers,options);

            if predictMode == 1
               newInput = xMat(i+1,:);
               newInput_normed = num2cell(((newInput - trainX_mu) ./ trainX_std)',1);
               [net,YPred] = predictAndUpdateState(net,newInput_normed);
               ypred(i+1,targetRegion) = targetY_std*YPred{1,1} + targetY_mu;
            elseif predictMode == 2
               [net, out] = iterPredict_LSTM(net, roll_Y_target_normed, delaysUsed, predictAhead, targetY_mu, targetY_std);
               ypred{i+1,targetRegion} = num2cell(out);            
            elseif predictMode == 3
               xConditioned = xMat(i+1:i+predictAhead,:);
               xConditioned_normed = fillmissing((xConditioned - trainX_mu) ./ trainX_std, 'constant', trainX_mu);
               inputConditioned = num2cell(xConditioned_normed',1);
               [net,outRaw] = predictAndUpdateState(net,inputConditioned); 
               % de-normalise
               for k = 1:length(outRaw); out(k) = targetY_std * double(outRaw{k})+ targetY_mu; end
               ypred{i+1,targetRegion} = num2cell(out);
            end

        end
    end
    
    magicPlotter_Conditional;
    
end
