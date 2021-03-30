import pandas as pd, numpy as np, matplotlib.pyplot as plt
import sqlite3, time
from tqdm import tqdm
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
import multiprocessing as mp

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20

conn = sqlite3.connect('/home/gekko/Desktop/PyPhD/RollingManifoldLearning/FXeodData.db')
twList = [25, 100, 150, 250, 'ExpWindow25']

def RNNprocess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    out = sl.AI.gRNN(df, params)
    out[0].to_sql('df_real_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_' + selection + str(magicNum), conn, if_exists='replace')
    out[2].to_sql('scoreList_RNN_' + selection + str(magicNum), conn, if_exists='replace')
    df_real_price = out[0]
    df_predicted_price = out[1]

    df_predicted_price.columns = df_real_price.columns

    # Returns Prediction
    sig = sl.sign(df_predicted_price)
    pnl = sig * df_real_price

    pnl.to_sql('pnl_RNN_' + selection + str(magicNum), conn, if_exists='replace')

def runRnn(Portfolios, scanMode, mode):
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

    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'ClassicPortfolios':
        allPortfoliosList = []
        for tw in twList:
            subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_' + str(tw), conn).set_index('Dates', drop=True)
            subDF.columns = ["RP_" + str(tw)]
            allPortfoliosList.append(subDF)
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        allPortfoliosList.append(LOportfolio)
        allProjectionsDF = pd.concat(allPortfoliosList, axis=1)

    targetSystems = range(5)

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], params, magicNum])

            p = mp.Pool(mp.cpu_count())
            result = p.map(RNNprocess, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for magicNum in targetSystems:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_RNN_' + selection + str(magicNum), conn).set_index('Dates', drop=True)
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).abs().values[0]
                        shList.append([selection + str(magicNum), medSh])
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_RNN_' + selection + str(magicNum))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
            shDF.to_sql(Portfolios+"_RNN_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_RNN', conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_RNN', conn).set_index('index', drop=True)
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

def plotRnnSharpes(Portfolios, manifoldIn):

    df = pd.read_sql('SELECT * FROM '+Portfolios+'_RNN_sharpe', conn)

    if Portfolios == 'Projections':

        df["model"] = None
        df['manifold'] = None
        df['prType'] = None
        for idx, row in df.iterrows():
            df.loc[idx, ['model']] = row['selection'][-1]
            infoSplit = row['selection'][:-1].split('_')
            if len(infoSplit) == 3:
                df.loc[idx, ['selection']] = "$y_{" + str(int(infoSplit[2]) + 1) + ",s" + str(infoSplit[0]) + ",t}^{" + str(infoSplit[1]) + "}$"
                df.loc[idx, ['prType']] = "coordinate"
            elif len(infoSplit) == 2:
                df.loc[idx, ['selection']] = "$Y_{s" + str(infoSplit[0]) + "," + str(infoSplit[1]) + ",t}$"
                df.loc[idx, ['prType']] = "global"
            df.loc[idx, ['manifold']] = str(infoSplit[0])
        df["model"] = df["model"].replace("0", "A").replace("1", "B").replace("2", "C").replace("3", "D").replace("4", "E")

        dfCoordinates = df[df["prType"] == "coordinate"][["selection", "sharpe", "model"]]
        top_dfCoordinates = dfCoordinates.copy()
        top_dfCoordinates['sharpe'] = top_dfCoordinates['sharpe'].abs().round(4)
        top_dfCoordinates = top_dfCoordinates[top_dfCoordinates['selection'].str.contains(manifoldIn)].set_index(
            "selection",
            drop=True)
        top_dfCoordinates = sl.Paperize(top_dfCoordinates.sort_values(by="sharpe", ascending=False).iloc[:5])
        print("top_dfCoordinates")
        print(top_dfCoordinates["PaperText"])
        dfGlobal = df[df["prType"] == "global"][["selection", "sharpe", "model"]]
        top_Global = dfGlobal.copy()
        top_Global['sharpe'] = top_Global['sharpe'].abs().round(4)
        top_Global = top_Global[top_Global['selection'].str.contains(manifoldIn)].set_index("selection", drop=True)
        top_Global = sl.Paperize(top_Global.sort_values(by="sharpe", ascending=False).iloc[:5])
        print("top_Global")
        print(top_Global["PaperText"])

        ##################################################
        dfCoordinates.set_index(['selection', 'model'], inplace=True)
        dfCoordinates.sort_index(inplace=True)
        dfUnstackCoordinates = dfCoordinates.unstack(level=0)
        manifoldlist = [x for x in dfUnstackCoordinates.columns if manifoldIn in x[1]]
        dfToplotCoordinates = dfUnstackCoordinates.loc[:, manifoldlist]
        dfToplotCoordinates.columns = [x[1] for x in dfToplotCoordinates.columns]

        fig, ax = plt.subplots()
        dfToplotCoordinates.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.tick_params(axis='x', labelrotation=0)
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, ncol=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
        plt.show()
        ##################################################
        dfGlobal.set_index(['selection', 'model'], inplace=True)
        dfGlobal.sort_index(inplace=True)
        dfUnstackGlobal = dfGlobal.unstack(level=0)
        manifoldlist = [x for x in dfUnstackGlobal.columns if manifoldIn in x[1]]
        dfToplotGlobal = dfUnstackGlobal.loc[:, manifoldlist]
        dfToplotGlobal.columns = [x[1] for x in dfToplotGlobal.columns]

        fig, ax = plt.subplots()
        dfToplotGlobal.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.tick_params(axis='x', labelrotation=0)
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 18}, borderaxespad=0.)
        plt.show()

    elif Portfolios == "ClassicPortfolios":

        df['model'] = df['selection'].str[-1]
        df['selection'] = df['selection'].str[:-1]

        dfLO = df[df['selection'].str.contains("LO")][["model", "sharpe"]]
        print("LO")
        print(dfLO)
        dfRP = df[df['selection'].str.contains("RP")]
        dfRP.set_index(['selection', 'model'], inplace=True)
        dfRP.sort_index(inplace=True)
        dfRPUnstack = dfRP.unstack(level=0)
        print("RP")
        print(dfRPUnstack)

def GPCprocess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]

    out = sl.AI.gGPC(df, params)
    out[0].to_sql('df_real_price_GPC_' + selection + "_" + params["sys"], conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_GPC_' + selection + "_" + params["sys"], conn, if_exists='replace')
    out[2].to_sql('df_proba_price_GPC_' + selection + "_" + params["sys"], conn, if_exists='replace')
    df_real_price = out[0]
    df_predicted_price = out[1]

    df_predicted_price.columns = df_real_price.columns

    # Returns Prediction
    sig = sl.sign(df_predicted_price)
    pnl = sig * df_real_price

    pnl.to_sql('pnl_GPC_' + selection + "_" + params["sys"], conn, if_exists='replace')

def runGpc(Portfolios, scanMode, mode):

    def GPCparams(sys):
        if sys == 1:
            params = {
                "sys": sys,
                "TrainWindow": 5,
                "LearningMode": 'online',
                "subhistory": 50,
                "Kernel": "Matern",
                "TrainEndPct": 0.3,
                "writeLearnStructure": 0
            }
        elif sys == 2:
            params = {
                "sys": sys,
                "TrainWindow": 25,
                "LearningMode": 'online',
                "subhistory": 10,
                "Kernel": "Matern",
                "TrainEndPct": 0.3,
                "writeLearnStructure": 0
            }
        return params

    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'ClassicPortfolios':
        allPortfoliosList = []
        for tw in twList:
            subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_' + str(tw), conn).set_index('Dates', drop=True)
            subDF.columns = ["RP_" + str(tw)]
            allPortfoliosList.append(subDF)
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        allPortfoliosList.append(LOportfolio)
        allProjectionsDF = pd.concat(allPortfoliosList, axis=1)

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for sys in [1,2]:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], GPCparams(sys)])

            p = mp.Pool(mp.cpu_count())
            result = p.map(GPCprocess, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for sys in [1,2]:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_GPC_' + selection + "_" + str(sys), conn).set_index('Dates', drop=True)
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).abs().values[0]
                        shList.append([selection + "_" + str(sys), medSh])
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_GPC_' + selection + "_" + str(sys))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
            shDF.to_sql(Portfolios+"_GPC_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_GPC', conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_GPC', conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            Info = row['NotProcessedProjection'].replace("pnl_GPC_", "")
            selection = Info[:-1]
            sys = Info[-1]
            print("Rerunning NotProcessed : ", selection, ", ", sys)

            params = GPCparams(sys)
            out = sl.AI.gRNN(allProjectionsDF[selection], params)
            out[0].to_sql('df_real_price_RNN_' + selection + "_" + str(sys), conn, if_exists='replace')
            out[1].to_sql('df_predicted_price_RNN_' + selection + "_" + str(sys), conn, if_exists='replace')
            out[2].to_sql('scoreList_RNN_' + selection + "_" + str(sys), conn,if_exists='replace')
            df_real_price = out[0]
            df_predicted_price = out[1]

            df_predicted_price.columns = df_real_price.columns

            # Returns Prediction
            sig = sl.sign(df_predicted_price)
            pnl = sig * df_real_price

            pnl.to_sql('pnl_GPC_' + selection + "_" + str(sys), conn, if_exists='replace')
            rsPnL = sl.rs(pnl)
            print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
            print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

#runRnn("ClassicPortfolios", 'Main', "run")
#runRnn("ClassicPortfolios", 'Main', "report")
#runRnn("Projections", 'Main', "run")
#runRnn("Projections", 'Main', "report")
#runRnn("Projections", 'Main', "run")
#runGpc("Projections", 'Main', "run")
#runGpc("Projections", 'Main', "report")
#runRnn('ScanNotProcessed', "")

#plotRnnSharpes("ClassicPortfolios", "")
#plotRnnSharpes("Projections", "PCA")
#plotRnnSharpes("Projections", "LLE")
