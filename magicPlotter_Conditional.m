% Check if Region's Folder exists
activePredictionsFolder = 'D:\Dropbox\EpidemicModel\WorkingDataset\MatlabEpidemic\MatlabFigures\';
targetRegionName = strrep(targetRegionName,'''','');
regionFolder = strcat(activePredictionsFolder, targetRegionName);
save(strcat(activePredictionsFolder, '\regionData'));
if ~exist(regionFolder, 'dir'); mkdir(regionFolder); end

% Plots
figure(1);
realData = totale_positivi(:,targetRegion);
targetPredictionsCell = ypred(:, targetRegion);
targetPredictionsCell_lower = ypred_lower(:, targetRegion);
targetPredictionsCell_upper = ypred_upper(:, targetRegion);
for c = 1:length(targetPredictionsCell)
    % /pred/
    subCellData = cell2mat(targetPredictionsCell{c});
    targetPredictionsMat(c:c+length(subCellData)-1,c) = subCellData'; 
    % /lower/
    subCellData_lower = cell2mat(targetPredictionsCell_lower{c});
    targetPredictionsMat_lower(c:c+length(subCellData_lower)-1,c) = subCellData_lower'; 
    % /upper/
    subCellData_upper = cell2mat(targetPredictionsCell_upper{c});
    targetPredictionsMat_upper(c:c+length(subCellData_upper)-1,c) = subCellData_upper'; 
end
targetPredictionsMat(targetPredictionsMat==0)=nan;
targetPredictionsMat_lower(targetPredictionsMat_lower==0)=nan;
targetPredictionsMat_upper(targetPredictionsMat_upper==0)=nan;
% impose non-negativity
targetPredictionsMat(targetPredictionsMat<0)=0;
targetPredictionsMat_lower(targetPredictionsMat_lower<0)=0;
targetPredictionsMat_upper(targetPredictionsMat_upper<0)=0;

if diffFlag == 1; for j = 1:size(targetPredictionsMat,2); targetPredictionsMat(:,j) = targetPredictionsMat(:,j)+lagmatrix(realData,1);end; end
% ///
plot(realData, 'Color', 'black', 'linewidth', 3); hold on; 
if predictAhead == 1 || predictMode == 1
    plot(targetPredictionsMat, '*', 'Color', 'green'); 
else
    plot(targetPredictionsMat, 'Color', 'green'); hold on; 
    plot(targetPredictionsMat_lower, 'Color', 'blue'); hold on; 
    plot(targetPredictionsMat_upper, 'Color', 'blue'); hold on; 
end
grid;
saveas(figure(1),strcat(regionFolder, '\MultiStepPredictionsFigure','.png'))

% Norm Plots
for predStepReport = 1:predictAhead
    clear firstPreds; close all;
    for col = 1:size(targetPredictionsMat,2)
        subPreds = rmmissing([targetPredictionsMat(:,col), realData]);
        subPreds_lower = rmmissing(targetPredictionsMat_lower(:,col));
        subPreds_upper = rmmissing(targetPredictionsMat_upper(:,col));
        if isempty(subPreds)
            firstPreds(col,:) = [NaN,NaN,NaN,NaN]; 
        else
            firstPreds(col,1:2) = subPreds(predStepReport,:);
            firstPreds(col,3) = subPreds_lower(predStepReport);
            firstPreds(col,4) = subPreds_upper(predStepReport);
        end
    end
    firstPreds = rmmissing(firstPreds);
    firstPreds(firstPreds==0) = NaN;

    subResids = 100*(firstPreds(:,1) - firstPreds(:,2))./firstPreds(:,2);
    %CCC(predStepReport) = f_CCC(subResids,0.05);
    % Statistics on the first predictions set
    rmse(predStepReport) = sqrt(mean((subResids).^2));
    %[lbq_h lbq_p]= lbqtest(resids_firstStep); [jb_h jb_p]= jbtest(resids_firstStep);

    if predStepReport == 1
       allResidsData = [firstPreds(:,2), subResids];
    else
       allResidsData = [allResidsData,subResids];
    end

    figure(2);
    plot(firstPreds(:,1), 'Color', 'blue', 'linewidth', 3); hold on; 
    plot(firstPreds(:,2), 'Color', 'black', 'linewidth', 3); hold on;   
    curve1 = firstPreds(:,3); curve2 = firstPreds(:,4);
    curve1(isnan(curve1)) = 0; curve2(isnan(curve2)) = 0;
    t = 1:size(firstPreds,1);
    shade(t,curve1,t,curve2,'FillType',[1 2;2 1]);
    grid;
    saveas(figure(2),strcat(regionFolder,'\', num2str(predStepReport),'-StepPredictionsFigure','.png'))
end    

figure(3);
resids_firstStep = rmmissing(allResidsData(:,2));
Y = prctile(resids_firstStep',[5 95]);
I=find(resids_firstStep>Y(1) & resids_firstStep<Y(2));
normplot(resids_firstStep(I))
saveas(figure(3),strcat(regionFolder,'\normplot_FirstStepPredictionsOnly','.png'))

figure(4);
allResids = allResidsData(:,2:end);
allResidsNormPlot = rmmissing(allResids(:));
Y = prctile(allResidsNormPlot',[5 95]);
I=find(allResidsNormPlot>Y(1) & allResidsNormPlot<Y(2));
normplot(allResidsNormPlot(I))
saveas(figure(4),strcat(regionFolder,'\normplot_AllPredictions','.png'))        

figure(5);
set(gca,'FontSize',20);
h = bar(1:size(allResids,1),allResids, 'stacked');
set(h, {'DisplayName'}, {'1 Day-Prediction','2 Days-Predictions','3 Days-Predictions', '4 Days-Predictions', '5 Days-Predictions'}')
legend();
saveas(figure(5),strcat(regionFolder,'\StackedResiduals_BarPlot','.png'))        

% Close all figures for next run
close all;