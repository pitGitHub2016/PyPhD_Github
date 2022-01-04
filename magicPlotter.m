targetRegionName = strrep(targetRegionName,'''','');
regionFolder = strcat(activePredictionsFolder, targetRegionName);
if ~exist(regionFolder, 'dir'); mkdir(regionFolder); end

% Region Data HTML
try
region_EpidemicData_Index = find(contains(allRegions_raw_Report(:,1),targetRegionName));
region_EpidemicData_raw_Latest = [allRegions_raw_Report(1,:);allRegions_raw_Report(region_EpidemicData_Index,:)];
region_EpidemicData_raw_Latest(:,1) = strrep(region_EpidemicData_raw_Latest(:,1),targetRegionName, '');
region_EpidemicData_raw_Latest(:,1) = strrep(region_EpidemicData_raw_Latest(:,1),'_', ' ');
region_EpidemicData_raw_Latest(:,1) = strrep(region_EpidemicData_raw_Latest(:,1),'''', '');
html_table(region_EpidemicData_raw_Latest, strcat(mapPredictionsFolder, targetRegionName, '_Epidemic_raw_Latest.html'), ...
        'DataFormatStr','%1.0f', ... %'Caption',strcat('<b>', targetRegionName, ' Epidimiological Data</b><br><br>')
        'BackgroundColor','#EFFFFF', 'RowBGColour',{'#000099',[],[],[],[],'#FFFFCC',[], '#FFFFCC'}, 'RowFontColour',{'#FFFFB5'}, ...
        'FirstRowIsHeading',1, 'FirstColIsHeading',1);
catch
end
% Plots
yMatData = yMat(:,targetRegion); yMatData = [yMatData;NaN(predictAhead, size(yMatData,2))];

%realData = totale_positivi(:,targetRegion); realData = [realData;NaN(predictAhead, size(realData,2))];
realData = yMat(:,targetRegion); realData = [realData;NaN(predictAhead, size(realData,2))]; % use the smoothed data as the °real° ones

%realData = yMatData; % Use the smoothed data as 'real' for residual analysis
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
% Expand Dates (once)
if targetRegion == 1
    for k =1:size(targetPredictionsMat,1)+1
        try
            datesAll{k} = char(dates(k));
        catch
            datesAll{k} = char(dateshift( datetime(datesAll{k-1}),'start','day','next'));
        end        
    end
    datesAll = datesAll';
end

% ///
%figure(1);
%plot(realData, 'Color', 'black', 'linewidth', 3); hold on; 
%plot(targetPredictionsMat, 'Color', 'green'); hold on; 
%plot(targetPredictionsMat_lower, 'Color', 'blue'); hold on; 
%plot(targetPredictionsMat_upper, 'Color', 'blue'); hold on; 
%grid on; set(gca, 'XTickLabel', get(gca, 'XTick')); set(gca, 'YTickLabel', get(gca, 'YTick'));
%saveas(figure(1),strcat(regionFolder, '\MultiStepPredictionsFigure','.png'))

% Norm Plots
for predStepReport = 1:predictAhead
    clear firstPreds; close all;
    for col = 1:size(targetPredictionsMat,2)
        subReadData = realData; subReadData(end-predictAhead+1:end) = 0;
        subPreds = rmmissing([targetPredictionsMat(:,col), subReadData]);
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
    % ///////// FIRST-STEP AHEAD PREDICTION EXCEL ////
    if predStepReport == 1
        latestPredsIntv = 1:size(firstPreds,1);
        latestPredsMat = datesAll(latestPredsIntv);
        latestPredsMat(:,2) = num2cell(realData(latestPredsIntv));
        latestPredsMat(:,3) = num2cell(firstPreds(:,1));
        latestPredsMat(:,4) = num2cell(firstPreds(:,3));
        latestPredsMat(:,5) = num2cell(firstPreds(:,4));
        latestPredsMat(:,6) = num2cell(firstPreds(:,1)-realData(latestPredsIntv));
        latestPredsMat(:,7) = num2cell(100*(firstPreds(:,1)-realData(latestPredsIntv))./realData(latestPredsIntv));
        latestPredsExcel = [{'data', 'Real', 'Expected', 'Lower', 'Upper', 'Error (#Cases)', 'Relative Error(%)'};latestPredsMat];%(end-(predictAhead):end,:);
        xlswrite(strcat(regionFolder,'\',targetRegionName, ' Predictions.xlsx'), latestPredsExcel)
    end
    % /////////////////////
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
    set(gca,'FontSize',14);
    plot(firstPreds(:,1), 'Color', 'blue', 'linewidth', 3); hold on; 
    plot(firstPreds(:,2), 'Color', 'black', 'linewidth', 3); hold on;   
    curve1 = firstPreds(:,3); curve2 = firstPreds(:,4);
    curve1(isnan(curve1)) = 0; curve2(isnan(curve2)) = 0;
    t = 1:size(firstPreds,1);
    shade(t,curve1,t,curve2,'FillType',[1 2;2 1]);
    set(gca,'xtick',[1:30:length(dates)],'xticklabel',dates(1:30:length(dates)));
    xtickangle(45);
    grid on; set(gca, 'YTickLabel', get(gca, 'YTick'));
    legend('Predictions', 'Real', 'Lower', 'Upper', 'CI Band');
    ylabel('#Infected','FontWeight','bold');
    saveas(figure(2),strcat(regionFolder,'\', num2str(predStepReport),'-StepPredictionsFigure','.png'))
end    

for predStepNormPlot = 1:predictAhead
    figure(3);
    resids_firstStep = rmmissing(allResidsData(:,predStepNormPlot+1));
    Y = prctile(resids_firstStep',[5 95]);
    I=find(resids_firstStep>Y(1) & resids_firstStep<Y(2));
    normplot(resids_firstStep(I))
    grid on;
    saveas(figure(3),strcat(regionFolder,'\normplot_PredictionsStep_', num2str(predStepNormPlot),'.png'))
    close(3);
end

%figure(4);
%allResids = allResidsData(:,2:end);
%allResidsNormPlot = rmmissing(allResids(:));
%Y = prctile(allResidsNormPlot',[5 95]);
%I=find(allResidsNormPlot>Y(1) & allResidsNormPlot<Y(2));
%normplot(allResidsNormPlot(I))
%grid on;
%saveas(figure(4),strcat(regionFolder,'\normplot_AllPredictions','.png'))        

%figure(5);
%set(gca,'FontSize',20);
%h = bar(1:size(allResids,1),allResids, 'stacked');
%set(h, {'DisplayName'}, {'1 Day-Prediction','2 Days-Predictions','3 Days-Predictions', '4 Days-Predictions', '5 Days-Predictions'}')
%legend();
%grid on; set(gca, 'XTickLabel', get(gca, 'XTick')); set(gca, 'YTickLabel', get(gca, 'YTick'));
%saveas(figure(5),strcat(regionFolder,'\StackedResiduals_BarPlot','.png'))        

figure(6);
daysBack = size(dates,1) - find(contains(dates, '2021-11-21')) + predictAhead;
latestIntv = size(targetPredictionsMat,1)-daysBack:size(targetPredictionsMat,1);
realDataLatest = realData(latestIntv);
yMatDataLatest = yMatData(latestIntv);
targetPredictionsMatLatest = targetPredictionsMat(latestIntv,:);
targetPredictionsMat_lowerLatest = targetPredictionsMat_lower(latestIntv,:);
targetPredictionsMat_upperLatest = targetPredictionsMat_upper(latestIntv,:);
for k =1:length(latestIntv)
    try
        datesLatest{k} = char(dates(latestIntv(k)));
    catch
        datesLatest{k} = char(dateshift( datetime(datesLatest{k-1}),'start','day','next'));
    end        
end

latest_preds = targetPredictionsMatLatest(:,size(targetPredictionsMat,2));
latest_preds(1:size(realDataLatest,1)-predictAhead) = realDataLatest(1:size(realDataLatest,1)-predictAhead);
plot(latest_preds, 'Color', 'blue', 'linewidth', 3); hold on;
latest_lower = targetPredictionsMat_lowerLatest(:,size(targetPredictionsMat_lowerLatest,2)); 
latest_lower(1:size(realDataLatest,1)-predictAhead) = realDataLatest(1:size(realDataLatest,1)-predictAhead);
latest_upper = targetPredictionsMat_upperLatest(:,size(targetPredictionsMat_upperLatest,2));
latest_upper(1:size(realDataLatest,1)-predictAhead) = realDataLatest(1:size(realDataLatest,1)-predictAhead);
%plot(yMatDataLatest, 'Color', 'green', 'linewidth', 3); hold on;
%text(round(size(yMatDataLatest,1)/2), yMatDataLatest(round(size(yMatDataLatest,1)/2)), 'MA','Rotation',30);

t = 1:size(latest_lower,1);
shade(t,latest_lower,t,latest_upper,'FillType',[1 2;2 1]);
plot(realDataLatest, 'Color', 'black', 'linewidth', 3); hold on; 
set(gca,'xtick',[1:length(datesLatest)],'xticklabel',datesLatest(1:length(datesLatest)));
xtickangle(45);
y = ylim; xLinePos = length(datesLatest)-predictAhead; plot([xLinePos xLinePos],[y(1) y(2)],'--','linewidth', 3);
grid on; set(gca, 'YTickLabel', get(gca, 'YTick'));
ax = gca; labels = string(ax.XAxis.TickLabels); % extract
labels(1:2:end-1) = nan; % remove every other one
ax.XAxis.TickLabels = labels; % set

posShift = [0,5];
for p=1:length(posShift)
textPos = size(latest_preds,1)-posShift(p);

text(textPos,latest_preds(textPos)+3,num2str(round(latest_preds(textPos))));
text(textPos,latest_preds(textPos)+3,num2str(round(latest_preds(textPos))));
text(textPos,latest_lower(textPos)+3,num2str(round(latest_lower(textPos))));
text(textPos,latest_upper(textPos)+3,num2str(round(latest_upper(textPos))));

end
title(strcat(num2str(predictAhead),'-Day Predictions'));
saveas(figure(6),strcat(regionFolder, '\LatestPredictionsFigure','.png'))

% Close all figures for next run
close all;
zippedfiles = zip(strcat(regionFolder,'\',targetRegionName,'.zip'),regionFolder);