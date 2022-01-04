% Matlab Epidemic
% Run Matlab Script
scriptNumber = 1;
addpath('C:\Users\lucia\Desktop\EpidemicModel/WorkingDataset\MatlabEpidemic');
dataPath = 'C:\Users\lucia\Desktop\EpidemicModel/WorkingDataset\'; 
cd(strcat(dataPath, 'MatlabEpidemic')); 
parentFiguresFolder = strcat(dataPath,'MatlabEpidemic\MatlabFigures\');
mapPredictionsFolder = strcat(dataPath,'ItalyInteractiveMap_Active\it-js-map\');
activePredictionsFolder = strcat(dataPath,'ItalyInteractiveMap_Active\it-js-map\Italy_Epidemic_Active_Predictions_90\'); envID = 'Production';
% /////////////////////////////// TOTAL POSITIVE DATASET///////////////////////////////////////////////
[totale_positivi,txt,raw] = xlsread(strcat(dataPath,'totale_positivi.xlsx')); % OFFICIAL
%[totale_positivi,txt,raw] = xlsread(strcat(dataPath, 'totale_casi.xlsx')); totale_positivi = fillmissing([zeros(1,size(totale_positivi,2)); diff(totale_positivi)], 'constant', 0);
% /////////////////////////////////////REST DATASETS//////////////////////////////////////////////////////////
datasetIDNum = 0; mat = txt(1,2:end); mat = mat'; diffFlag = 0; 
dates = txt(2:end,1);
RegionsRange = 1:size(totale_positivi,2);
% ////////////////////////////////////////////////////////////////////////////////////////////////////////////
xMat = movavg(totale_positivi,'linear', 7);
%yMat = totale_positivi; 
yMat = movavg(totale_positivi,'linear', 7); %OFFICIAL ACTIVE
%yMatExternal = movavg(fillmissing([zeros(1,size(totale_positivi,2)); diff(totale_positivi)], 'constant', 0),'exponential', 5);
% ////////////////////////////////////////////////////////////////////////////////////////////////////////////
delaysUsed = [1,2,4,6,8]; %[1,2],[1,2,4,6,8],[1,2,5,10,15,20,30,50]
predictAhead = 5;
st = 90; 
%
for i = st:size(yMat,1)%st+5
    i
    %intv = 1:i;
    intv = i-st+1:i;
    roll_Data_X = xMat(intv,:); 
    roll_Data_Y = yMat(intv,:); 

    % /// X inputs
    trainX_mu =  mean(roll_Data_X); trainX_std = std(roll_Data_X); 
    roll_Data_X_normed = fillmissing((roll_Data_X - trainX_mu) ./ trainX_std, 'constant', trainX_mu);
    % /// Y Targets
    trainY_mu =  mean(roll_Data_Y); trainY_std = std(roll_Data_Y); 
    roll_Data_Y_normed = fillmissing((roll_Data_Y - trainY_mu) ./ trainY_std, 'constant', trainY_mu);
    %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    [out, out_lower, out_upper] = MultiIterPredict_GPR(roll_Data_X_normed, roll_Data_Y_normed, delaysUsed, predictAhead);    
    
    % de-normalise
    for targetRegion=RegionsRange
        outDeNormed = trainY_std(targetRegion) * out(:,targetRegion)+ trainY_mu(targetRegion);
        out_lowerDeNormed = trainY_std(targetRegion) * out_lower(:,targetRegion) + trainY_mu(targetRegion);
        out_upperDeNormed = trainY_std(targetRegion) * out_upper(:,targetRegion) + trainY_mu(targetRegion);
        
        ypred{i+1,targetRegion} = num2cell(outDeNormed);
        ypred_lower{i+1,targetRegion} = num2cell(out_lowerDeNormed);
        ypred_upper{i+1,targetRegion} = num2cell(out_upperDeNormed);    
    end
    
end

if diffFlag == 1
    ypred_Diffed = ypred;
    for k1 = st+1:size(ypred,1)
        for k2 = 1:size(ypred,2)
            ypred{k1,k2} = num2cell(totale_positivi(k1-1)+cumsum(cell2mat(ypred{k1,k2})));
        end
    end
end
% ////////////////////////////////////////////////////////////////////////////////////////////
modelSelection = 1; % 1 = GPR, 2 = LSTM
predictAhead = 5;
predictMode = 2; % 1 = one-step-ahead(not the target), 2 = multi-step-ahead, 3 = conditioned
% ////////////////////////////////////////////////////////////////////////////////////////////
save(strcat(activePredictionsFolder, '\PredictionData_', envID));

% Plots
for targetRegion=RegionsRange
    clear latestPredsMat
    targetRegion
    
    if datasetIDNum == 0
        targetRegionName = strrep(mat(targetRegion),'totale_positivi_','');
        targetRegionName = targetRegionName{1,1};
    else
        targetRegionName = strrep(mat(targetRegion).name,'.mat','');
    end
    
    %RegionRt = [raw(2:end,1), raw(2:end,targetRegion+1), deceduti_raw(2:end,targetRegion+1)];
    %xlswrite(strcat(mapPredictionsFolder, 'RtFolder\',targetRegionName,'.xlsx'), RegionRt);

    magicPlotter;
    
end
