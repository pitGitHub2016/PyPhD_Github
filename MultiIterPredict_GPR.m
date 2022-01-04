function [out, out_lower, out_upper] = MultiIterPredict_GPR(DataInX, DataInY, delaysUsed, predictAhead)
    %DataInX = roll_Data_X_normed; DataInY = roll_Data_Y_normed;
        
    for predStep = 1:predictAhead
        for targetRegion = 1:size(DataInY,2)            
            X_input_normed = getShifts(DataInX, delaysUsed);
            Y_target_normed = DataInY(:,targetRegion);

            gprModels{targetRegion} = fitrgp(X_input_normed, Y_target_normed, 'KernelFunction','matern32');  

            X_input_LastStep = getShifts(DataInX, delaysUsed-1);
            singlePointX = X_input_LastStep(size(X_input_LastStep,1),:);
            
            [raw_iterPreds,raw_iterPreds_std, raw_iterPreds_int] = predict(gprModels{targetRegion}, singlePointX);
            predMat(targetRegion) = raw_iterPreds;
            out(predStep, targetRegion) = raw_iterPreds;
            out_lower(predStep, targetRegion) = raw_iterPreds_int(1);
            out_upper(predStep, targetRegion) = raw_iterPreds_int(2);
        end
        DataInX = [DataInX;predMat];
        DataInY = [DataInY;predMat];
    end
    
end