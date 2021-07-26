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

def RunRnn():
    df = pd.read_sql('SELECT * FROM PCA_RsExPostProjections', conn).set_index('Dates', drop=True)[['P0']]
    df = sl.cs(df)

    HistLag = 0;TrainWindow = 50;epochsIn = 50;batchSIzeIn = 49;medBatchTrain = 1;HistoryPlot = 0;PredictionsPlot = [1, 'cs'];LearningMode = 'static'

    out = sl.AI.gRNN(df, [HistLag, TrainWindow, epochsIn, batchSIzeIn, medBatchTrain, HistoryPlot, PredictionsPlot, LearningMode])
    out[0].to_sql('df_real_price_RNN_PCA_Projections_'+str(HistLag)+'_'+str(TrainWindow)+'_'+str(epochsIn)+'_'+str(batchSIzeIn)+'_'+str(medBatchTrain)+'_'+str(HistoryPlot)+'_'+str(PredictionsPlot[0])+'_'+LearningMode, conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_PCA_Projections_'+str(HistLag)+'_'+str(TrainWindow)+'_'+str(epochsIn)+'_'+str(batchSIzeIn)+'_'+str(medBatchTrain)+'_'+str(HistoryPlot)+'_'+str(PredictionsPlot[0])+'_' + LearningMode, conn, if_exists='replace')
    out[2].to_sql('scoreList_RNN_PCA_Projections_'+str(HistLag)+'_'+str(TrainWindow)+'_'+str(epochsIn)+'_'+str(batchSIzeIn)+'_'+str(medBatchTrain)+'_'+str(HistoryPlot)+'_'+str(PredictionsPlot[0])+'_'+ LearningMode, conn, if_exists='replace')

def BackTestRnn():
    df_real_price = pd.read_sql('SELECT * FROM df_real_price_RNN_PCA_Projections_0_50_1_49_2_0_1_static', conn).set_index('Dates', drop=True)
    df_predicted_price = pd.read_sql('SELECT * FROM df_predicted_price_RNN_PCA_Projections_0_50_1_49_2_0_1_static', conn).set_index('Dates', drop=True)

    dfAll = pd.concat([df_real_price, df_predicted_price], axis=1)

    print(dfAll.tail())

BackTestRnn()
