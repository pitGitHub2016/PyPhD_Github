function [model, out] = iterPredict_LSTM(model, newDataIn, delaysUsed, predictAhead, targetY_mu, targetY_std)
    %model = net; newDataIn = roll_Y_target_normed; 
        
    for j = 1:predictAhead
        newShiftedX = getShifts(newDataIn, delaysUsed-1);
        singlePointX_raw = newShiftedX(size(newShiftedX,1),:);
        singlePointX = num2cell(singlePointX_raw',1); 
        if j == 1
            [model,raw_iterPreds] = predictAndUpdateState(model,singlePointX);
        else
            raw_iterPreds = predict(model,singlePointX);
        end
        iterPreds(j) = cell2mat(raw_iterPreds);
        newDataIn = [newDataIn; iterPreds(j)];
    end

    %de-normalize predictions
    for k = 1:length(iterPreds)
        out(k) = targetY_std * iterPreds(k)+ targetY_mu;
    end
        
end