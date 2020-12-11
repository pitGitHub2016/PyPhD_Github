import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import sqlite3, tqdm
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
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20

conn = sqlite3.connect('RollingPCA/FXeodData.db')

def getProjections(mode):
    rng = [0, 1, 2, 3, 4]
    allProjectionsPCA = []
    #PCAls = pd.read_sql('SELECT * FROM PCA_lambdasDf', conn).set_index('Dates', drop=True)
    for pr in rng:
        # PCA
        PCArs = pd.DataFrame(
            sl.rs(pd.read_sql('SELECT * FROM PCA_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)))
        PCArs.columns = ['PCA' + str(pr)]
    #    medls = pd.DataFrame(PCAls.iloc[:, pr])
    #    medls.columns = ['PCA' + str(pr)]
        allProjectionsPCA.append(PCArs)
    PCAdf = pd.concat(allProjectionsPCA, axis=1)

    allProjectionsLLE = []
    for pr in rng:
        try:
            # LLE
            LLErs = pd.DataFrame(sl.rs(
                pd.read_sql('SELECT * FROM LLE_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)))
            # LLErs = pd.DataFrame(pd.read_sql('SELECT * FROM LLE_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True))
            LLErs.columns = ['LLE' + str(pr)]
            allProjectionsLLE.append(LLErs)
        except Exception as e:
            print(e)
    LLEdf = pd.concat(allProjectionsLLE, axis=1)

    allProjectionsDF = pd.concat([PCAdf, LLEdf], axis=1)

    if mode == 'RV':
        allProjectionsDF = sl.RV(allProjectionsDF)
    elif mode == 'RVPriceRatio':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="priceRatio")
    elif mode == 'RVExpCorr':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="ExpCorr")
    elif mode == 'RVRollHedgeRatio':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="RollHedgeRatio")
    elif mode == 'Baskets':
        allProjectionsDF = sl.Baskets(allProjectionsDF)

    return allProjectionsDF

def runRnn(mode):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 0:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 2,
                "epochsIn": 50,
                "batchSIzeIn": 1,
                "LearningMode": 'online',
                "medSpecs": [
                             {"LayerType": "SimpleRNN", "units": 10, "RsF": False, "Dropout": 0}
                             ],
                "modelNum": magicNum,
                "TrainEndPct": 0.3,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }
        elif magicNum == 1:

            #'xShape1'
            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 2,
                "epochsIn": 50,
                "batchSIzeIn": 1,
                "LearningMode": 'online',
                "medSpecs": [
                             {"LayerType": "LSTM", "units": 10, "RsF": False, "Dropout": 0}
                             ],
                "modelNum": magicNum,
                "TrainEndPct": 0.3,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }
        elif magicNum == 2:

            #'xShape1'
            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 5,
                "epochsIn": 20,
                "batchSIzeIn": 5,
                "LearningMode": 'online',
                "medSpecs": [
                             {"LayerType": "SimpleRNN", "units": 10, "RsF": True, "Dropout": 0.1},
                             {"LayerType": "SimpleRNN", "units": 5, "RsF": False, "Dropout": 0}
                             ],
                "modelNum": magicNum,
                "TrainEndPct": 0.3,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }
        elif magicNum == 3:

            #'xShape1'
            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 5,
                "epochsIn": 20,
                "batchSIzeIn": 5,
                "LearningMode": 'online',
                "medSpecs": [
                             {"LayerType": "LSTM", "units": 10, "RsF": True, "Dropout": 0.1},
                             {"LayerType": "LSTM", "units": 5, "RsF": False, "Dropout": 0}
                             ],
                "modelNum": magicNum,
                "TrainEndPct": 0.3,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }

        elif magicNum == 4:

            #'xShape1'
            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 25,
                "epochsIn": 20,
                "batchSIzeIn": 25,
                "LearningMode": 'online',
                "medSpecs": [
                             {"LayerType": "LSTM", "units": 20, "RsF": True, "Dropout": 0.1},
                             {"LayerType": "LSTM", "units": 2, "RsF": False, "Dropout": 0}
                             ],
                "modelNum": magicNum,
                "TrainEndPct": 0.3,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }
        return paramsSetup

    allProjectionsDF = getProjections('')
    # allProjectionsDF = getProjections('RV')
    # allProjectionsDF = getProjections('RVExpCorr')
    #allProjectionsDF = getProjections('RVRollHedgeRatio')
    # allProjectionsDF = getProjections('Baskets')

    allProjectionsDFPCA = allProjectionsDF[[x for x in allProjectionsDF.columns if 'PCA' in x]]
    allProjectionsDFLLE = allProjectionsDF[[x for x in allProjectionsDF.columns if 'LLE' in x]]
    allProjectionsDF = pd.concat([sl.rs(allProjectionsDFPCA), sl.rs(allProjectionsDFLLE)], axis=1)
    allProjectionsDF.columns = ["PCA", "LLE"]

    #targetSystems = range(4)
    targetSystems = [4]

    if mode == "run":

        for magicNum in targetSystems:
            params = Architecture(magicNum)
            for selection in allProjectionsDF.columns:
                print(selection)
                out = sl.AI.gRNN(allProjectionsDF[selection], params)
                out[0].to_sql('df_real_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
                out[1].to_sql('df_predicted_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
                out[2].to_sql('scoreList_RNN_' + selection + str(magicNum), conn,if_exists='replace')
                df_real_price = out[0]
                df_predicted_price = out[1]

                #df_real_price = pd.read_sql('SELECT * FROM df_real_price_RNN_'+ whatToRun + str(magicNum), conn).set_index('Dates', drop=True)
                #df_predicted_price = pd.read_sql('SELECT * FROM df_predicted_price_RNN_'+ whatToRun + str(magicNum), conn).set_index('Dates', drop=True)

                df_predicted_price.columns = df_real_price.columns

                # Returns Prediction
                sig = sl.sign(df_predicted_price)
                pnl = sig * df_real_price

                pnl.to_sql('pnl_RNN_' + selection + str(magicNum), conn, if_exists='replace')
                rsPnL = sl.rs(pnl)
                print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
                print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

    elif mode == "report":
        shList = []
        for magicNum in targetSystems:
            for selection in allProjectionsDF.columns:
                pnl = pd.read_sql(
                'SELECT * FROM pnl_RNN_' + selection + str(magicNum), conn).set_index('Dates', drop=True)
                medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                shList.append([selection + str(magicNum), medSh])
        shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
        print(shDF.abs().sort_index(ascending=False))

runRnn("run")
runRnn("report")

