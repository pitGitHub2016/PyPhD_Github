import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import sqlite3, tqdm, time
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

conn = sqlite3.connect('/home/gekko/Desktop/PyPhD/RollingManifoldLearning/FXeodData.db')

def runRnn(scanMode, mode):
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

    allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

    targetSystems = range(5)

    if scanMode == 'Main':

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
            notProcessed = []
            for magicNum in targetSystems:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_RNN_' + selection + str(magicNum), conn).set_index('Dates', drop=True)
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                        shList.append([selection + str(magicNum), medSh])
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_RNN_' + selection + str(magicNum))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
            shDF.to_sql("RNN_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql('notProcessedDF_RNN', conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        notProcessedDF = pd.read_sql('SELECT * FROM notProcessedDF_RNN', conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            Info = row['NotProcessedProjection'].replace("pnl_RNN_", "")
            selection = Info[:-1]
            magicNum = Info[-1]
            print("Rerunning NotProcessed : ", selection, ", ", magicNum)

            params = Architecture(magicNum)
            out = sl.AI.gRNN(allProjectionsDF[selection], params)
            out[0].to_sql('df_real_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
            out[1].to_sql('df_predicted_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
            out[2].to_sql('scoreList_RNN_' + selection + str(magicNum), conn,if_exists='replace')
            df_real_price = out[0]
            df_predicted_price = out[1]

            df_predicted_price.columns = df_real_price.columns

            # Returns Prediction
            sig = sl.sign(df_predicted_price)
            pnl = sig * df_real_price

            pnl.to_sql('pnl_RNN_' + selection + str(magicNum), conn, if_exists='replace')
            rsPnL = sl.rs(pnl)
            print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
            print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

def plotRnnSharpes(manifoldIn):
    df = pd.read_sql('SELECT * FROM RNN_sharpe', conn)
    df['model'] = df['selection'].str[-1]
    df['selection'] = df['selection'].str[:-1]
    print(df[df['sharpe'] > 0.8].set_index('selection', drop=True))
    df.set_index(['selection', 'model'], inplace=True)
    df.sort_index(inplace=True)
    dfUnstack = df.unstack(level=0)
    print(dfUnstack)
    PCAlist = [x for x in dfUnstack.columns if manifoldIn in x[1]]
    dfToplot = dfUnstack.loc[:,PCAlist].abs()
    dfToplot.columns = [x[1] for x in dfToplot.columns]

    fig, ax = plt.subplots()
    dfToplot.plot(ax=ax, kind='bar')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Sharpe Ratio")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 12}, borderaxespad=0.)
    plt.show()

runRnn('Main', "run")
runRnn('Main', "report")
#runRnn('ScanNotProcessed', "")

#plotRnnSharpes("PCA")
#plotRnnSharpes("LLE")
