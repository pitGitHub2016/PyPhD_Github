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
pnlCalculator = 0

def ClassificationProcess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    if calcMode == 'run':

        out = sl.AI.gClassification(df, params)

        out[0].to_sql('df_predicted_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[1].to_sql('df_real_price_class_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[2].to_sql('df_real_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[3].to_sql('df_predicted_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[4].to_sql('df_real_price_class_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[5].to_sql('df_real_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')

    elif calcMode == 'read':
        print(selection)
        out = [
            pd.read_sql('SELECT * FROM df_predicted_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_class_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_predicted_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_class_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
        ]

    if pnlCalculator == 0:
        sig = out[3]

        sig[sig < 0.5] = 0
        sig[(sig <= 1.5) & (sig >= 0.5)] = 1
        sig[sig > 1.5] = -1

    df_real_price_test_DF = out[5]

    dfPnl = pd.concat([df_real_price_test_DF, sig], axis=1)
    dfPnl.columns = ["Real_Price", "Sig"]

    pnl = dfPnl["Real_Price"] * dfPnl["Sig"]
    sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
    print(sh_pnl)

    pnl.to_sql('pnl_'+params['model']+'_' + selection + "_" + str(magicNum), conn, if_exists='replace')

def runClassification(Portfolios, scanMode, mode):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 0:

            paramsSetup = {
                "model": "RNN",
                "HistLag": 0,
                "InputSequenceLength": 240,  # 240
                "SubHistoryLength": 760,  # 760
                "SubHistoryTrainingLength": 510,  # 510
                "Scaler": "Standard",  # Standard
                "epochsIn": 100,  # 100
                "batchSIzeIn": 16,  # 16
                "EarlyStopping_patience_Epochs": 10,  # 10
                "LearningMode": 'static',  # 'static', 'online'
                "medSpecs": [
                    {"LayerType": "LSTM", "units": 50, "RsF": True, "Dropout": 0.25},
                    {"LayerType": "LSTM", "units": 50, "RsF": True, "Dropout": 0.25},
                    {"LayerType": "LSTM", "units": 50, "RsF": False, "Dropout": 0.25}
                ],
                "modelNum": magicNum,
                "CompilerSettings": ['adam', 'mean_squared_error'],
            }

        elif magicNum == 1:

            paramsSetup = {
                "model": "GPC",
                "HistLag": 0,
                "InputSequenceLength": 240,  # 240
                "SubHistoryLength": 760,  # 760
                "SubHistoryTrainingLength": 510,  # 510
                "Scaler": None,  # Standard
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }

        return paramsSetup

    if Portfolios == 'Projections':
        #allProjectionsDF = pd.read_csv("E:/PyPhD/PCA_LLE_Data/allProjectionsDF.csv").set_index('Dates', drop=True)
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
             globalProjectionsList.append(pd.read_sql('SELECT * FROM globalProjectionsDF_'+manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif Portfolios == 'ClassicPortfolios':
        allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["RP"]
        allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    elif Portfolios == 'Finalists':
        allProjectionsDF = pd.read_csv("E:/PyPhD/PCA_LLE_Data/allProjectionsDF.csv").set_index('Dates', drop=True)[['PCA_250_0', 'LLE_250_0', 'PCA_250_19', 'LLE_250_18']]
        #allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_250_0', 'LLE_250_0', 'PCA_250_19', 'LLE_250_18']]

    targetSystems = [0]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], params, magicNum])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ClassificationProcess, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for magicNum in targetSystems:
                if magicNum in [0]:
                    Classifier = "RNN"
                elif magicNum in [1]:
                    Classifier = "GPC"
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_'+Classifier+'_' + selection + str(magicNum), conn).set_index('Dates', drop=True)
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).abs().values[0]
                        shList.append([selection + str(magicNum), medSh])
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_'+Classifier+'_' + selection + str(magicNum))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
            shDF.to_sql(Portfolios+"_"+Classifier+"_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_'+Classifier, conn, if_exists='replace')
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

def Test(mode):
    magicNum = 1000
    # selection = 'PCA_250_3_Head'
    # selection = 'LLE_250_3_Head'
    selection = 'PCA_250_0'
    # selection = 'PCA_250_19'
    df_Main = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
    #df_Main = pd.read_csv("E:/PyPhD\PCA_LLE_Data/allProjectionsDF.csv").set_index('Dates', drop=True)[selection]
    # df_Main = pd.read_sql('SELECT * FROM globalProjectionsDF_PCA', conn).set_index('Dates', drop=True)[selection]
    # df_Main = pd.read_sql('SELECT * FROM globalProjectionsDF_LLE', conn).set_index('Dates', drop=True)[selection]

    df_Main = df_Main.iloc[-500:]

    if mode == 'run':

        dfList = [df_Main]
        df_real_price_List = []
        df_predicted_price_List = []
        for df in dfList:

            print("len(df) = ", len(df))

            """
            params = {
                "model" : "RNN",
                "HistLag": 0,
                "InputSequenceLength": 240, #240
                "SubHistoryLength": 760, #760
                "SubHistoryTrainingLength": 510, #510
                "Scaler": "Standard", #Standard
                "epochsIn": 2, #100
                "batchSIzeIn": 1, #16
                "EarlyStopping_patience_Epochs": 1, #10
                "LearningMode": 'static', #'static', 'online'
                "medSpecs": [
                    #{"LayerType": "LSTM", "units": 50, "RsF": True, "Dropout": 0.25},
                    #{"LayerType": "LSTM", "units": 50, "RsF": True, "Dropout": 0.25},
                    {"LayerType": "LSTM", "units": 5, "RsF": False, "Dropout": 0.25}
                ],
                "modelNum": magicNum,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                "writeLearnStructure": 0
            }
            """
            params = {
               "model": "GPC",
                "HistLag": 0,
                "InputSequenceLength": 25,  # 240
                "SubHistoryLength": 50,  # 760
                "SubHistoryTrainingLength": 30,  # 510
                "Scaler": None,  # Standard
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }
            out = sl.AI.gClassification(df, params)

            out[0].to_sql('df_predicted_price_train_DF_Test_'+selection, conn, if_exists='replace')
            out[1].to_sql('df_real_price_class_train_DF_Test_'+selection, conn, if_exists='replace')
            out[2].to_sql('df_real_price_train_DF_Test_'+selection, conn, if_exists='replace')
            out[3].to_sql('df_predicted_price_test_DF_Test_'+selection, conn, if_exists='replace')
            out[4].to_sql('df_real_price_class_test_DF_Test_'+selection, conn, if_exists='replace')
            out[5].to_sql('df_real_price_test_DF_Test_'+selection, conn, if_exists='replace')

    elif mode == 'read':

        sig = pd.read_sql('SELECT * FROM df_predicted_price_test_DF_Test_'+selection, conn).set_index('Dates', drop=True)
        sig[sig < 0.5] = 0
        sig[(sig <= 1.5) & (sig >= 0.5)] = 1
        sig[sig > 1.5] = -1
        df_real_price_test_DF_Test = pd.read_sql('SELECT * FROM df_real_price_test_DF_Test_'+selection, conn).set_index('Dates', drop=True)
        dfPnl = pd.concat([df_real_price_test_DF_Test, sig], axis=1)
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

if __name__ == '__main__':

    #runClassification("ClassicPortfolios", 'Main', "run")
    #runClassification("ClassicPortfolios", 'Main', "report")
    #runClassification("Projections", 'Main', "run")
    #runClassification("Projections", 'Main', "report")
    runClassification("Finalists", 'Main', "run")
    #runClassification("FinalistsProjections", 'Main', "report")
    #runClassification('ScanNotProcessed', "")

    #Test("run")
    #Test("read")
