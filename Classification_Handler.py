import pandas as pd, numpy as np, matplotlib.pyplot as plt
import sqlite3, time, pickle
try:
    from tqdm import tqdm
    from hurst import compute_Hc
except:
    pass
from itertools import combinations
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import History
import warnings, os, tensorflow as tf
from scipy import stats as st
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

#calcMode = 'runSerial'
#calcMode = 'runParallel'
calcMode = 'read'
pnlCalculator = 3
targetSystems = [1]#[0,1]

def ClassificationProcess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    if calcMode in ['runSerial', 'runParallel']:
        print("Running gClassification")
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

    sig = out[3] # Predicted Price
    df_real_price_class_DF = out[4]
    df_real_price_test_DF = out[5]

    sigDF = sig.copy()
    if pnlCalculator == 0:
        sigDF = sigDF["Predicted_Test_" + selection]
        sigDF = (sigDF-sigDF.mean()) / sigDF.std()
    elif pnlCalculator == 1:
        sigDF = sigDF["Predicted_Test_" + selection]
        sigDF[sigDF < 2/3] = 0
        sigDF[(sigDF >= 2/3) & (sigDF <= 1+1/3)] = 1
        sigDF[sigDF > 1+1/3] = -1
    elif pnlCalculator == 2:
        sigDF = sigDF["Predicted_Test_" + selection]
        sigDF[sigDF < 0.1] = 0
        sigDF[(sigDF >= 0.95) & (sigDF <= 1.05)] = 1
        sigDF[sigDF > 1.9] = -1
    elif pnlCalculator == 3:
        probDF = sigDF[["Predicted_Proba_Test_0.0", "Predicted_Proba_Test_1.0", "Predicted_Proba_Test_2.0"]]
        sigDF = sigDF["Predicted_Test_" + selection]

        sigDF[(sigDF < 2 / 3) & (probDF["Predicted_Proba_Test_0.0"] >= 0.7)] = 0
        sigDF[(sigDF >= 2 / 3) & (sigDF <= 1 + 1 / 3) & (probDF["Predicted_Proba_Test_1.0"] >= 0.7)] = 1
        sigDF[(sigDF > 1 + 1 / 3) & (probDF["Predicted_Proba_Test_2.0"] >= 0.7)] = -1

    sigDF.columns = ["ScaledSignal"]

    if selection == "LO1":
        fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
        df_real_price_class_DF.plot(ax=ax[0])
        sig.plot(ax=ax[1])
        sigDF.plot(ax=ax[2])
        plt.show()

    dfPnl = pd.concat([df_real_price_test_DF, sigDF], axis=1)
    dfPnl.columns = ["Real_Price", "Sig"]
    #dfPnl["Sig"].plot()
    #plt.show()
    #time.sleep(3000)

    pnl = dfPnl["Real_Price"] * dfPnl["Sig"]
    #pnl = dfPnl["Real_Price"] * sl.sign(dfPnl["Sig"])
    sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
    print("selection = ", selection, ", Target System = ", magicNum, ", ", sh_pnl)

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
                "Scaler": "Standard",  # Standard
                'Kernel': '6',
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }

        elif magicNum == 2:

            paramsSetup = {
                "model": "GPC",
                "HistLag": 0,
                "InputSequenceLength": 5,  # 240 || 5
                "SubHistoryLength": 300,  # 760 || 300
                "SubHistoryTrainingLength": 295,  # 510 || 295
                "Scaler": "Standard",  # Standard
                'Kernel': '0',
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }

        return paramsSetup

    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_csv("allProjectionsDF.csv").set_index('Dates', drop=True)
        #allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            #medDF = pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True)
            medDF = pd.read_csv('globalProjectionsDF_' + manifoldIn +'.csv').set_index('Dates', drop=True)
            globalProjectionsList.append(medDF)
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
        print("len(allProjectionsDF.columns) = ", len(allProjectionsDF.columns))
    elif Portfolios == 'ClassicPortfolios':
        allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["RP"]
        allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    elif Portfolios == 'Finalists':
        #allProjectionsDF = pd.read_csv("E:/PyPhD/PCA_LLE_Data/allProjectionsDF.csv").set_index('Dates', drop=True)[['PCA_250_0', 'LLE_250_0', 'PCA_250_19', 'LLE_250_18']]
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_250_0', 'LLE_250_0',
                                                                                                              'PCA_250_19', 'LLE_250_18',
                                                                                                              'PCA_ExpWindow25_0', 'LLE_ExpWindow25_0',
                                                                                                              'PCA_ExpWindow25_19', 'LLE_ExpWindow25_18']]

    #allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'ExpWindow25' not in x]]

    if scanMode == 'Main':

        if mode == "runSerial":
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    ClassificationProcess([selection, allProjectionsDF[selection], params, magicNum])

        elif mode == "runParallel":
            processList = []
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], params, magicNum])

            if calcMode == 'read':
                p = mp.Pool(2)
            else:
                p = mp.Pool(mp.cpu_count())
                #p = mp.Pool(len(processList))
            #result = p.map(ClassificationProcess, tqdm(processList))
            result = p.map(ClassificationProcess, processList)
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for magicNum in targetSystems:
                if magicNum in [0]:
                    Classifier = "RNN"
                elif magicNum in [1,2]:
                    Classifier = "GPC"
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_'+Classifier+'_' + selection + '_' + str(magicNum), conn).set_index('Dates', drop=True)

                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252 * 100).set_index("index",drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                        stats = pd.concat([statsMat.iloc[0, :], statsMat.iloc[1, :]], axis=0)
                        stats.index = ["Classifier_sh", "Classifier_Mean", "Classifier_tConf", "Classifier_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["Classifier_tConf", "RW_tConf"]] = stats[["Classifier_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["ttestPair_statistic"] = np.round(ttestPair.statistic, 2)
                        stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue, 2)
                        stats["Classifier"] = Classifier

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_'+Classifier+'_' + selection + '_' + str(magicNum))

            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(2)
            shDF.to_sql(Portfolios+"_"+Classifier+"_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_'+Classifier, conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        systemClass = 'GPC'
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_'+systemClass, conn).set_index('index', drop=True)
        print("len(notProcessedDF) = ", len(notProcessedDF))
        notProcessedList = []
        for idx, row in notProcessedDF.iterrows():
            Info = row['NotProcessedProjection'].replace("pnl_"+systemClass+"_", "")
            selection = Info[:-2]
            magicNum = Info[-1]
            params = Architecture(magicNum)
            print("Rerunning NotProcessed : ", selection, ", ", magicNum)
            notProcessedList.append([selection, allProjectionsDF[selection], params, magicNum])

        p = mp.Pool(mp.cpu_count())
        result = p.map(ClassificationProcess, tqdm(notProcessedList))
        p.close()
        p.join()

def Test(mode):
    magicNum = 1000
    selection = 'PCA_ExpWindow25_3_Head'
    #selection = 'LLE_ExpWindow25_3_Head'
    #selection = 'PCA_250_3_Head'
    #selection = 'LLE_250_3_Head'
    #selection = 'PCA_250_0'
    #selection = 'PCA_250_19'
    #selection = 'PCA_ExpWindow25_0' #
    #selection = 'PCA_ExpWindow25_19' #
    #selection = 'LLE_ExpWindow25_0' #
    #selection = 'RP'
    #df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
    #df = pd.read_csv("allProjectionsDF.csv").set_index('Dates', drop=True)[selection]
    df = pd.read_sql('SELECT * FROM globalProjectionsDF_PCA', conn).set_index('Dates', drop=True)[selection]
    # df = pd.read_sql('SELECT * FROM globalProjectionsDF_LLE', conn).set_index('Dates', drop=True)[selection]
    #allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
    #allProjectionsDF.columns = ["RP"]
    #allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    #df = allProjectionsDF[selection]

    #df_Main = df_Main.iloc[-500:]

    if mode == 'run':

        print("len(df) = ", len(df))

        params = {
            "model": "GPC",
            "HistLag": 0,
            "InputSequenceLength": 250,  # 240 (main) || 5 (MR) ||
            "SubHistoryLength": 500,  # 760 (main) || 500 (MR) ||
            "SubHistoryTrainingLength": 500-25,  # 510 (main) || 500-25 (MR) ||
            "Scaler": "Standard",  # Standard
            'Kernel': '0',
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

        sigDF = pd.read_sql('SELECT * FROM df_predicted_price_test_DF_Test_'+selection, conn).set_index('Dates', drop=True)

        probDF = sigDF[["Predicted_Proba_Test_0.0", "Predicted_Proba_Test_1.0", "Predicted_Proba_Test_2.0"]]
        sigDF = sigDF["Predicted_Test_" + selection]

        probThr = 0.7
        sigDF[(sigDF < 2 / 3) & (probDF["Predicted_Proba_Test_0.0"] >= probThr)] = 0
        sigDF[(sigDF >= 2 / 3) & (sigDF <= 1 + 1 / 3) & (probDF["Predicted_Proba_Test_1.0"] >= probThr)] = 1
        sigDF[(sigDF > 1 + 1 / 3) & (probDF["Predicted_Proba_Test_2.0"] >= probThr)] = -1

        #sigDF[(sigDF < 2 / 3)] = 0
        #sigDF[(sigDF >= 2 / 3) & (sigDF <= 1 + 1 / 3)] = 1
        #sigDF[(sigDF > 1 + 1 / 3)] = -1

        #sigDF[sigDF < 0.1] = 0
        #sigDF[(sigDF >= 0.95) & (sigDF <= 1.05)] = 1
        #sigDF[sigDF > 1.9] = -1

        df_real_price_test_DF_Test = pd.read_sql('SELECT * FROM df_real_price_test_DF_Test_'+selection, conn).set_index('Dates', drop=True)
        dfPnl = pd.concat([df_real_price_test_DF_Test, sigDF], axis=1)
        dfPnl.columns = ["real_price", "predicted_price"]

        pnl = dfPnl["real_price"] * dfPnl["predicted_price"]
        sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
        print(sh_pnl)

        sl.cs(pnl).plot()
        plt.show()

if __name__ == '__main__':

    #runClassification("ClassicPortfolios", 'Main', "runParallel")
    #runClassification("ClassicPortfolios", 'Main', "report")
    #runClassification("Projections", 'Main', "runParallel")
    #runClassification("Projections", 'Main', "report")
    #runClassification('Projections', 'ScanNotProcessed', "")
    #runClassification("globalProjections", 'Main', "runParallel")
    #runClassification("globalProjections", 'Main', "report")
    #runClassification('globalProjections', 'ScanNotProcessed', "")
    #runClassification("Finalists", 'Main', "runParallel")
    #runClassification("Finalists", 'Main', "report")

    Test("run")
    Test("read")