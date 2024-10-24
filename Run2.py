import pandas as pd, numpy as np, matplotlib.pyplot as plt
import sqlite3, time, pickle
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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20

try:
    conn = sqlite3.connect('/home/gekko/Desktop/PyPhD/RollingManifoldLearning/FXeodData.db')
except:
    conn = sqlite3.connect('Temp.db')
twList = [25, 100, 150, 250, 'ExpWindow25']

calcMode = 'run'
#calcMode = 'read'
pnlCalculator = 1

def RNNprocess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    if calcMode == 'run':
        out = sl.AI.gRNN(df, params)
        out[0].to_sql('df_real_price_RNN_' + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[1].to_sql('df_predicted_price_RNN_' + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[2].to_sql('scoreList_RNN_' + selection + "_" + str(magicNum), conn, if_exists='replace')

    elif calcMode == 'read':
        print(selection)
        out = [
            pd.read_sql('SELECT * FROM df_real_price_RNN_'+ selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_predicted_price_RNN_' + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM scoreList_RNN_' + selection + "_" + str(magicNum), conn)]

    df_real_price = out[0]
    df_predicted_price = out[1]
    df_predicted_price.columns = df_real_price.columns

    if pnlCalculator == 0:
        sig = sl.sign(df_predicted_price)
        pnl = sig * df_real_price
    elif pnlCalculator == 1:
        #pnl = (df_predicted_price-0.5) * df_real_price
        pnl = np.sign(df_predicted_price-0.5) * df_real_price

    reportSh = np.sqrt(252) * sl.sharpe(pnl)
    print(reportSh)
    fig, ax = plt.subplots(nrows=3, ncols=1)
    sl.cs(pnl).plot(ax=ax[0], title='csPnL')
    sl.cs(df_real_price).plot(ax=ax[1], title='Real Price')
    df_predicted_price.plot(ax=ax[2],title = 'Predicted Price')
    plt.show()

    pnl.to_sql('pnl_RNN_' + selection + "_" + str(magicNum), conn, if_exists='replace')

def runRnn(Portfolios, scanMode, mode):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 0:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 1,
                "epochsIn": 100,
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
                "TrainWindow": 1,
                "epochsIn": 100,
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
        return paramsSetup

    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'ClassicPortfolios':
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        RPportfolio = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["RP"]
        allProjectionsDF = pd.concat([LOportfolio, RPportfolio], axis=1)
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            globalProjectionsList.append(
                pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif Portfolios == 'FinalistsProjections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
            ['PCA_ExpWindow25_0', 'PCA_ExpWindow25_19', 'LLE_ExpWindow25_0', "LLE_ExpWindow25_18"]]
    elif Portfolios == 'FinalistsGlobalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            globalProjectionsList.append(
                pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)[
            ["PCA_ExpWindow25_5_Head", "PCA_ExpWindow25_5_Tail", "LLE_ExpWindow25_5_Head", "LLE_ExpWindow25_5_Tail",
             "PCA_ExpWindow25_3_Head", "PCA_ExpWindow25_3_Tail", "LLE_ExpWindow25_3_Head", "LLE_ExpWindow25_3_Tail"]]

    targetSystems = [0, 1]

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

def Test(mode):
    if mode == 'run':
        magicNum = 1000
        #selection = 'PCA_250_3_Head'
        #selection = 'LLE_250_3_Head'
        selection = 'LLE_250_0'
        #selection = 'PCA_250_19'
        df_Main = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
        #df_Main = pd.read_csv("E:/PyPhD\PCA_LLE_Data/allProjectionsDF.csv").set_index('Dates', drop=True)[selection]
        #df_Main = pd.read_sql('SELECT * FROM globalProjectionsDF_PCA', conn).set_index('Dates', drop=True)[selection]
        #df_Main = pd.read_sql('SELECT * FROM globalProjectionsDF_LLE', conn).set_index('Dates', drop=True)[selection]

        #sub_trainingSetIvlIn, sub_testSetInvIn = 750, 250
        #dfList = sl.AI.overlappingPeriodSplitter(df_Main, sub_trainingSetIvl=sub_trainingSetIvlIn, sub_testSetInv=sub_testSetInvIn)

        dfList = [df_Main]
        df_real_price_List = []
        df_predicted_price_List = []
        for df in dfList:

            print("len(df) = ", len(df))

            params = {
                "model" : "GPC",
                "HistLag": 0,
                "InputSequenceLength": 240, #240
                "SubHistoryLength": 760, #760
                "SubHistoryTrainingLength": 510, #510
                "Scaler": None, #Standard
                "LearningMode": 'static', #'static', 'online'
                "modelNum": magicNum
            }
            #df = sl.cs(df)
            #RNNprocess([selection, df, params, magicNum])
            out = sl.AI.gClassification(df, params)

            out[0].to_sql('df_predicted_price_train_DF_Test', conn, if_exists='replace')
            out[1].to_sql('df_real_price_class_train_DF_Test', conn, if_exists='replace')
            out[2].to_sql('df_real_price_train_DF_Test', conn, if_exists='replace')
            out[3].to_sql('df_predicted_price_test_DF_Test', conn, if_exists='replace')
            out[4].to_sql('df_real_price_class_test_DF_Test', conn, if_exists='replace')
            out[5].to_sql('df_real_price_test_DF_Test', conn, if_exists='replace')

    elif mode == 'read':

        df_predicted_price_test_DF_Test = pd.read_sql('SELECT * FROM df_predicted_price_test_DF_Test', conn).set_index('Dates', drop=True)
        df_predicted_price_test_DF_Test[df_predicted_price_test_DF_Test>1.5] = -1
        df_predicted_price_test_DF_Test[(df_predicted_price_test_DF_Test<=1.5)&(df_predicted_price_test_DF_Test>=0.5)] = 1
        df_predicted_price_test_DF_Test[df_predicted_price_test_DF_Test<0.5] = 0
        df_real_price_test_DF_Test = pd.read_sql('SELECT * FROM df_real_price_test_DF_Test', conn).set_index('Dates', drop=True)
        dfPnl = pd.concat([df_real_price_test_DF_Test, df_predicted_price_test_DF_Test], axis=1)
        dfPnl.columns = ["real_price", "predicted_price"]

        pnl = dfPnl["real_price"] * dfPnl["predicted_price"]
        sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
        print(sh_pnl)

        #print(len(df_predicted_price_test_DF_Test), len(df_real_price_test_DF_Test))
        #print(df_predicted_price_test_DF_Test.tail(10))
        #print(df_real_price_test_DF_Test.tail(10))
        #df_predicted_price_test_DF_Test.plot()
        #df_real_price_test_DF_Test.plot()
        sl.cs(pnl).plot()
        plt.show()

#runRnn("ClassicPortfolios", 'Main', "run")
#runRnn("ClassicPortfolios", 'Main', "report")
#runRnn("Projections", 'Main', "run")
#runRnn("Projections", 'Main', "report")
#runRnn("FinalistsProjections", 'Main', "run")
#runRnn("FinalistsProjections", 'Main', "report")
#runRnn('ScanNotProcessed', "")

#runGpc("Projections", 'Main', "run")
#runGpc("Projections", 'Main', "report")

Test("run")
#Test("read")

#train_size = 50
#rng = np.random.RandomState(0)
#X = rng.uniform(0, 5, 100)[:, np.newaxis]
#y = np.array(X[:, 0] > 2.5, dtype=int)
#print(X)
#print(y)
#print(X.shape, y.shape)