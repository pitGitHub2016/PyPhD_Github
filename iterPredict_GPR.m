function [out, out_lower, out_upper] = iterPredict_GPR(model, newDataIn, delaysUsed, predictAhead, targetY_mu, targetY_std)
    %model = gprMdl; newDataIn = roll_Y_target_normed; 
        
    for j = 1:predictAhead
        newShiftedX = getShifts(newDataIn, delaysUsed-1);
        singlePointX = newShiftedX(size(newShiftedX,1),:);
        [raw_iterPreds,raw_iterPreds_std, raw_iterPreds_int] = predict(model, singlePointX);
        iterPreds(j) = raw_iterPreds;
        iterPreds_lower(j) = raw_iterPreds_int(1);
        iterPreds_upper(j) = raw_iterPreds_int(2);
        newDataIn = [newDataIn;iterPreds(j)];
    end

    %de-normalize predictions
    for k = 1:length(iterPreds); out(k) = targetY_std*iterPreds(k)+ targetY_mu; end
    for k = 1:length(iterPreds_lower); out_lower(k) = targetY_std * iterPreds_lower(k)+ targetY_mu; end
    for k = 1:length(iterPreds_upper); out_upper(k) = targetY_std * iterPreds_upper(k)+ targetY_mu; end
        
end