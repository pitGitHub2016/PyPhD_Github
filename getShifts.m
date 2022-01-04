function out = getShifts(data, lags)
%lags = [1,2,4,6,8]; data = yMat;
sub_out = lagmatrix(data,lags(1));
for i = 2:length(lags)
    sub_out = [sub_out, lagmatrix(data,lags(i))];
end

out = fillmissing(sub_out, 'constant', 0);
