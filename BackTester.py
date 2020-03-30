import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import sqlite3
from itertools import combinations
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import History
import warnings, os, tensorflow as tf
from Slider import Slider as sl

conn = sqlite3.connect('RollingPCA/FXeodData.db')

HistLag = 0
TrainWindow = 100
epochsIn = 100
batchSIzeIn = 99
medBatchTrain = 99
HistoryPlot = 0
PredictionsPlot = [1, 'cs']
#PredictionsPlot = [1, 'NoCs']
LearningMode = 'static'
modelNum = '1'

def RunRnn(manifoldIn):
    df = pd.read_sql('SELECT * FROM '+manifoldIn+'_RsExPostProjections', conn).set_index('Dates', drop=True)
    #df = sl.cs(df)

    out = sl.AI.gRNN(df, [HistLag, TrainWindow, epochsIn, batchSIzeIn, medBatchTrain, HistoryPlot, PredictionsPlot, LearningMode])
    out[0].to_sql(
        'df_real_price_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(epochsIn) + '_' + str(
            batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(
        epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
        PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')
    out[2].to_sql(
        'scoreList_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(epochsIn) + '_' + str(
            batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')

def BackTestRnn(manifoldIn):

    df_real_price = pd.read_sql(
        'SELECT * FROM df_real_price_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn).set_index('Dates', drop=True)

    df_predicted_price = pd.read_sql(
        'SELECT * FROM df_predicted_price_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn).set_index('Dates', drop=True)

    df_predicted_price_errors = pd.read_sql(
        'SELECT * FROM scoreList_RNN_'+manifoldIn+'_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn)
    print(df_predicted_price_errors)

    df_predicted_price.columns = df_real_price.columns

    df_real_price.plot(); df_predicted_price.plot(title='RNN : Real vs Predicted Dynamics'); plt.show()

    sig = sl.S(sl.sign(df_predicted_price), nperiods=-1)
    #sig = sl.S(sl.sign(df_predicted_price))

    #pnl = sig * df_real_price.iloc[0:round(0.5*6110),:]
    #pnl = sl.ExPostOpt(sig * df_real_price.iloc[0:round(0.5*6110),:])[0]
    pnl = sl.ExPostOpt(sig * df_real_price)[0]
    #pnl = sl.ExPostOpt(sig * allProjectionsDF)[0] # From ARIMA ...

    pnl.to_sql(
        'pnl_RNN_' + manifoldIn + '_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')
    #pnl = sl.ExPostOpt(sig * sl.d(df_real_price))[0]
    rsPnL = sl.rs(pnl)
    print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
    print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

    fig = plt.figure()
    sl.cs(pnl).plot(ax=fig.add_subplot(121), title='RNN Trading on Projections')
    sl.cs(rsPnL).plot(ax=fig.add_subplot(122), title='EWP of RNN Projections')

    #df_predicted_price_errors.plot(ax=fig.add_subplot(121), title='RNN Trading on Projections Error')
    plt.show()

def MergeRNNpnls(modelSel):
    PCA_pnl = pd.read_sql(
        'SELECT * FROM pnl_RNN_PCA_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelSel, conn).set_index('Dates', drop=True)

    LLE_pnl = pd.read_sql(
        'SELECT * FROM pnl_RNN_LLE_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelSel, conn).set_index('Dates', drop=True)

    pnlAll = pd.concat([PCA_pnl, LLE_pnl], axis=1, ignore_index=True)

    rsPnL = sl.rs(pnlAll)
    print((np.sqrt(252) * sl.sharpe(pnlAll)).round(4))
    print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

    fig = plt.figure()
    sl.cs(pnlAll).plot(ax=fig.add_subplot(121), title='RNN Trading on PCA and LLE Projections combined')
    sl.cs(rsPnL).plot(ax=fig.add_subplot(122), title='EWP of RNN PnLs')

    # df_predicted_price_errors.plot(ax=fig.add_subplot(121), title='RNN Trading on Projections Error')
    plt.show()

def BackTestArima(manifoldIn, setIn):
    allProjections = []
    for pr in range(5):
        exPostProjections = pd.read_sql('SELECT * FROM ' + 'PCA' + '_exPostProjections_' + str(pr), conn).set_index(
            'Dates', drop=True)
        allProjections.append(sl.rs(exPostProjections))
    allProjectionsDF = pd.concat(allProjections, axis=1, ignore_index=True)
    allProjectionsDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    df_predicted_price = pd.read_sql('SELECT * FROM '+manifoldIn+'_ARIMA_Predictions_'+setIn, conn).set_index('Dates', drop=True)
    df_predicted_price.columns = allProjectionsDF.columns

    #sig = sl.S(sl.sign(df_predicted_price), nperiods=-1) # Currently On Paper
    sig = sl.S(sl.sign(df_predicted_price))

    #pnl = sig * allProjectionsDF.iloc[0:round(0.5*6110),:]
    pnl = sl.ExPostOpt(sig * allProjectionsDF.iloc[0:round(0.5*6110),:])[0]
    rsPnL = sl.rs(pnl)
    print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
    print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

    fig = plt.figure()
    sl.cs(pnl).plot(ax=fig.add_subplot(121), title='ARIMA Trading on Projections')
    sl.cs(rsPnL).plot(ax=fig.add_subplot(122), title='EWP of ARIMA Projections')
    plt.show()

    df_confDF = pd.read_sql('SELECT * FROM '+manifoldIn+'_ARIMA_confDF_'+setIn, conn).set_index('Dates', drop=True)
    df_errDF = pd.read_sql('SELECT * FROM '+manifoldIn+'_ARIMA_errDF_'+setIn, conn).set_index('Dates', drop=True)
    df_stderrDFDF = pd.read_sql('SELECT * FROM '+manifoldIn+'_ARIMA_stderrDF_'+setIn, conn).set_index('Dates', drop=True)

    fig = plt.figure()
    df_confDF.iloc[500:].plot(ax=fig.add_subplot(131), title='ARIMA on '+manifoldIn+' Projections - Confidence Intervals')
    df_errDF.iloc[500:].plot(ax=fig.add_subplot(132), title='ARIMA on '+manifoldIn+' Projections - Residuals')
    df_stderrDFDF.iloc[500:].plot(ax=fig.add_subplot(133), title='ARIMA on '+manifoldIn+' Projections - Standard Deviations of Residuals')
    plt.show()


#testDf = pd.DataFrame([0,1,2,3]); print(sl.S(testDf)); print(sl.S(testDf, nperiods=-1))

#RunRnn('PCA')
#RunRnn('LLE')

#BackTestRnn('PCA')
#BackTestRnn('LLE')

MergeRNNpnls('1')

#BackTestArima('PCA', '110')
#BackTestArima('LLE', '710')

"""
# Test Shifting
real_price = pd.DataFrame([10,10.5,11.8,11.2,9.6,9.4,8.6,10.2,10.6])
pred_price = pd.DataFrame([10.1,11,12,13,12.5,12,11.8,10.5,11.4])
sig = sl.sign(pred_price-real_price)
print(pd.concat([real_price, pred_price, sl.d(real_price), sig, sl.S(sig), sl.S(sig, nperiods=-1)], axis=1))
"""




