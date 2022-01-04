%% Rt Handling
[status, commandOut] = system('C:\Python39\python.exe D:/Dropbox/EpidemicModel/WorkingDataset/RtFilesHandler.py');
% Rt values per region
for targetRegion=1:size(roll_Data,2)
    targetRegion
    
    if datasetIDNum == 0
        targetRegionName = strrep(mat(targetRegion),'totale_positivi_','');
        targetRegionName = targetRegionName{1,1};
    else
        targetRegionName = strrep(mat(targetRegion).name,'.mat','');
    end
    
    RegionPopulation = cell2mat(population_raw(find(contains(population_raw(:,1), targetRegionName)),2));
    RegionRtFileName = strcat(mapPredictionsFolder, 'RtFolder\',targetRegionName,'.txt');
    find_Rt_App(RegionPopulation, RegionRtFileName, targetRegionName);
    close all;
end
[status, commandOut] = system('C:\Python39\python.exe D:/Dropbox/EpidemicModel/WorkingDataset/RtPhotosResize.py');
