% Matlab Epidemic
% Fetch New Data (Python)
[statusDataFetcher, commandOutDataFetcher] = system('C:\Python39-32\python.exe C:\Users\lucia\Desktop\EpidemicModel/WorkingDataset/RunEpidemicDataFetcher.py');
% Run Matlab Script
scriptNumber = 1;
addpath('C:\Users\lucia\Desktop\EpidemicModel/WorkingDataset\MatlabEpidemic');
dataPath = 'C:\Users\lucia\Desktop\EpidemicModel/WorkingDataset\'; 
cd(strcat(dataPath, 'MatlabEpidemic')); 
parentFiguresFolder = strcat(dataPath,'MatlabEpidemic\MatlabFigures\');
mapPredictionsFolder = strcat(dataPath,'ItalyInteractiveMap_Active\it-js-map\');
activePredictionsFolder = strcat(dataPath,'ItalyInteractiveMap_Active\it-js-map\Italy_Epidemic_Active_Predictions\'); envID = 'Production';
%activePredictionsFolder = strcat(dataPath,'ItalyInteractiveMap_Active\it-js-map\Italy_Epidemic_Active_Predictions_Development\'); envID = 'Development';
% /////////////////////////////// TOTAL POSITIVE DATASET///////////////////////////////////////////////
[totale_positivi,txt,raw] = xlsread(strcat(dataPath,'totale_positivi.xlsx')); % OFFICIAL
%[totale_positivi,txt,raw] = xlsread(strcat(dataPath, 'totale_casi.xlsx')); totale_positivi = fillmissing([zeros(1,size(totale_positivi,2)); diff(totale_positivi)], 'constant', 0);
% /////////////////////////////////////REST DATASETS//////////////////////////////////////////////////////////
[deceduti,deceduti_txt,deceduti_raw] = xlsread(strcat(dataPath, 'deceduti.xlsx'));
[population_Data,population_txt,population_raw] = xlsread(strcat(dataPath, 'PopulationDF.xlsx'));
[allRegionsData,allRegions_txt,allRegions_raw] = xlsread(strcat(dataPath, 'DataRegionsTimeSeries.xlsx'));
allRegions_raw_Report = [allRegions_raw(1,:);allRegions_raw(end,:)]';
[allEpidemicData,allEpidemic_txt,allEpidemic_raw] = xlsread(strcat(dataPath, 'NationalDataAll.xlsx'));
allEpidemicData_Report = [allEpidemic_raw(1,:);allEpidemic_raw(end,:)]';
allEpidemicData_Report(:,1) = strrep(allEpidemicData_Report(:,1),'_', ' ');
allEpidemicData_Report(:,1) = strrep(allEpidemicData_Report(:,1),'''', '');
allEpidemicData_Report(2,2) = strrep(allEpidemicData_Report(2,2),'ITA', 'ITALY');
allEpidemicData_Report = allEpidemicData_Report(find(sum(cellfun(@(cell) any(isnan(cell(:))),allEpidemicData_Report),2)==0), :);
html_table(allEpidemicData_Report, strcat(mapPredictionsFolder, 'allEpidemicData_Report.html'), ...
        'DataFormatStr','%1.0f', ... %'Caption','<b>Current Epidimiological Data</b><br><br>'
        'BackgroundColor','#EFFFFF', 'RowBGColour',{'#000099',[],[],[],[],[],'#FFFFCC',[], '#FFFFCC'}, 'RowFontColour',{'#FFFFB5'}, ...
        'FirstRowIsHeading',1, 'FirstColIsHeading',1);
datasetID = 'PP'; datasetIDNum = 0; mat = txt(1,2:end); mat = mat'; diffFlag = 0; 
dates = txt(2:end,1); deceduti_dates = deceduti_txt(2:end,1);
RegionsRange = 1:size(totale_positivi,2);
% ////////////////////////////////////////////////////////////////////////////////////////////////////////////
xMat = movavg(totale_positivi,'linear', 7);
%yMat = totale_positivi; 
yMat = movavg(totale_positivi,'linear', 7); %OFFICIAL ACTIVE
%yMatExternal = movavg(fillmissing([zeros(1,size(totale_positivi,2)); diff(totale_positivi)], 'constant', 0),'exponential', 5);
% ////////////////////////////////////////////////////////////////////////////////////////////////////////////
delaysUsed = [1,2,4,6,8]; %[1,2],[1,2,4,6,8],[1,2,5,10,15,20,30,50]
predictAhead = 5;
st = 60; 
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
% Run other rolling windows as well
clc;clear all;close all;
MatlabEpidemic_MultiIterative_90;
clc;clear all;close all;
MatlabEpidemic_MultiIterative_120;
% RtHandlingMatlab;
% Rt Handler (Python)
[status, commandOut] = system('C:\Python39-32\python.exe C:\Users\Lucia\Desktop\EpidemicModel/WorkingDataset/RunRtReporter.py');

% When all is done --> upload to site-server
cd('C:\Users\lucia\Desktop\EpidemicModel\WorkingDataset\ItalyInteractiveMap_Active');
[statusUploadSiteServer, commandOutUploadSiteServer] = system('batchSftpUploadRunner.bat');
exit;