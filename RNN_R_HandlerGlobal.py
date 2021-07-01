import pandas as pd, numpy as np, matplotlib.pyplot as plt
import sqlite3, time, pickle, os
try:
    from tqdm import tqdm
except:
    pass
from scipy import stats as st
from Slider import Slider as sl
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from random import randint

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20

try:
    conn = sqlite3.connect('/home/gekko/Desktop/PyPhD/RollingManifoldLearning/FXeodDataGlobal.db')
except:
    conn = sqlite3.connect('TempGlobal.db')
twList = [25, 100, 150, 250, 'ExpWindow25']

#calcMode = 'runSerial'
calcMode = 'runParallel'
#calcMode = 'read'
pnlCalculator = 0
targetSystems = [0, 1, 2, 3, 4]

def RegressionProcess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    if calcMode in ['runSerial', 'runParallel']:
        print("Running gRNN_Regression")
        out = sl.AI.gRNN_Regression(df, params)

        writeFlag = False
        while writeFlag == False:
            try:
                out[0].to_sql('df_predicted_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                out[1].to_sql('df_real_price_class_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                out[2].to_sql('df_real_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                out[3].to_sql('df_predicted_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                out[4].to_sql('df_real_price_class_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                out[5].to_sql('df_real_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
                writeFlag = True
            except Exception as e:
                print(e)
                conn.close()
                print("Sleeping for some seconds and retrying ... ")
                time.sleep(randint(0, 5))

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
        sigDF = sl.sign(sigDF)

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
    pnl = pnl.dropna()
    sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
    print("selection = ", selection, ", Target System = ", magicNum, ", ", sh_pnl)

    pnl.to_sql('pnl_'+params['model']+'_' + selection + "_" + str(magicNum), conn, if_exists='replace')

def runRegression(Portfolios, scanMode, mode):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 0:
            InputSequenceLength = 1
        elif magicNum == 1:
            InputSequenceLength = 3
        elif magicNum == 2:
            InputSequenceLength = 5
        elif magicNum == 3:
            InputSequenceLength = 10
        elif magicNum == 4:
            InputSequenceLength = 25

        paramsSetup = {
            "model": "RNNr",
            "HistLag": 0,
            "InputSequenceLength": InputSequenceLength,  # 240
            "SubHistoryLength": 250,  # 760
            "SubHistoryTrainingLength": 250 - 1,  # 510
            "Scaler": "Standard",  # Standard
            "epochsIn": 100,  # 100
            "batchSIzeIn": 10,  # 16
            "EarlyStopping_patience_Epochs": 10,  # 10
            "LearningMode": 'static',  # 'static', 'online'
            "medSpecs": [
                {"LayerType": "LSTM", "units": 25, "RsF": False, "Dropout": 0.25}
            ],
            "modelNum": magicNum,
            "CompilerSettings": ['adam', 'mean_squared_error'],
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
                    RegressionProcess([selection, allProjectionsDF[selection], params, magicNum])

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
            #result = p.map(RegressionProcess, tqdm(processList))
            result = p.map(RegressionProcess, processList)
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for magicNum in targetSystems:
                Classifier = "RNNr"
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_'+Classifier+'_' + selection + '_' + str(magicNum), conn).set_index('Dates', drop=True).dropna()

                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = np.sqrt(252) * sl.sharpe(pnl)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252 * 100).set_index("index",drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                        pnl_ttest_0 = st.ttest_1samp(pnl[selection].values, 0)
                        rw_pnl_ttest_0 = st.ttest_1samp(pnl['RW'].values, 0)
                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                        stats = pd.concat([statsMat.iloc[0, :], statsMat.iloc[1, :]], axis=0)
                        stats.index = ["Classifier_sh", "Classifier_Mean", "Classifier_tConf", "Classifier_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["Classifier_tConf", "RW_tConf"]] = stats[["Classifier_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue,2)
                        stats["pnl_ttest_0_pvalue"] = np.round(pnl_ttest_0.pvalue, 2)
                        stats["rw_pnl_ttest_0_value"] = np.round(rw_pnl_ttest_0.pvalue, 2)
                        stats["Classifier"] = Classifier+str(magicNum)

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_'+Classifier+'_' + selection + '_' + str(magicNum))

            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True)
            shDF = shDF[["Classifier_sh", "Classifier_Mean", "Classifier_tConf", "Classifier_Std", "pnl_ttest_0_pvalue",
                         "RW_sh", "RW_Mean", "RW_tConf", "RW_Std", "rw_pnl_ttest_0_value", "ttestPair_pvalue", "Classifier"]]
            shDF.to_sql(Portfolios+"_"+Classifier+"_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_'+Classifier, conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        systemClass = 'RNNr'
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_'+systemClass, conn).set_index('index', drop=True)
        print("len(notProcessedDF) = ", len(notProcessedDF))
        notProcessedList = []
        for idx, row in notProcessedDF.iterrows():
            Info = row['NotProcessedProjection'].replace("pnl_"+systemClass+"_", "")
            selection = Info[:-2]
            magicNum = Info[-1]
            params = Architecture(magicNum)
            print("Rerunning NotProcessed : ", selection, ", ", magicNum)
            #RegressionProcess([selection, allProjectionsDF[selection], params, magicNum])
            notProcessedList.append([selection, allProjectionsDF[selection], params, magicNum])

        p = mp.Pool(mp.cpu_count())
        result = p.map(RegressionProcess, tqdm(notProcessedList))
        p.close()
        p.join()

def Test(mode):
    magicNum = "test"
    #selection = 'PCA_ExpWindow25_3_Head'
    #selection = 'LLE_ExpWindow25_3_Head'
    #selection = 'PCA_250_3_Head'
    #selection = 'LLE_250_3_Head'
    #selection = 'PCA_250_0'
    #selection = 'PCA_250_19'
    #selection = 'PCA_ExpWindow25_0' #
    selection = 'PCA_ExpWindow25_19' #
    #selection = 'LLE_ExpWindow25_0' #
    #selection = 'RP'
    #df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
    df = pd.read_csv("allProjectionsDF.csv").set_index('Dates', drop=True)[selection]
    #df = pd.read_sql('SELECT * FROM globalProjectionsDF_PCA', conn).set_index('Dates', drop=True)[selection]
    # df = pd.read_sql('SELECT * FROM globalProjectionsDF_LLE', conn).set_index('Dates', drop=True)[selection]
    #df = pd.read_csv("globalProjectionsDF_PCA.csv").set_index('Dates', drop=True)[selection]
    #df = pd.read_csv("globalProjectionsDF_LLE.csv").set_index('Dates', drop=True)[selection]
    #allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
    #allProjectionsDF.columns = ["RP"]
    #allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    #df = allProjectionsDF[selection]

    if mode == 'run':

        print("len(df) = ", len(df))

        params = {
            "model": "RNNr",
            "HistLag": 0,
            "InputSequenceLength": 5,  # 240
            "SubHistoryLength": 250,  # 760
            "SubHistoryTrainingLength": 250 - 1,  # 510
            "Scaler": "Standard",  # Standard
            "epochsIn": 100,  # 100
            "batchSIzeIn": 10,  # 16
            "EarlyStopping_patience_Epochs": 10,  # 10
            "LearningMode": 'static',  # 'static', 'online'
            "medSpecs": [
                {"LayerType": "LSTM", "units": 25, "RsF": False, "Dropout": 0.25}
            ],
            "modelNum": magicNum,
            "CompilerSettings": ['adam', 'mean_squared_error'],
        }

        out = sl.AI.gRNN_Regression(df, params)

        out[0].to_sql('df_predicted_price_train_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[1].to_sql('df_real_price_class_train_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[2].to_sql('df_real_price_train_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[3].to_sql('df_predicted_price_test_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[4].to_sql('df_real_price_class_test_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[5].to_sql('df_real_price_test_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')
        out[6].to_sql('df_score_test_DF_Test_'+selection+"_"+str(magicNum), conn, if_exists='replace')

    elif mode == 'read':

        df_real_price_test_DF_Test = pd.read_sql('SELECT * FROM df_real_price_test_DF_Test_'+selection+"_"+str(magicNum), conn).set_index('Dates', drop=True)
        sigDF = pd.read_sql('SELECT * FROM df_predicted_price_test_DF_Test_'+selection+"_"+str(magicNum), conn).set_index('Dates', drop=True)
        scoreDF = pd.read_sql('SELECT * FROM df_score_test_DF_Test_'+selection+"_"+str(magicNum), conn).set_index('index', drop=True)

        #pd.concat([sl.cs(df_real_price_test_DF_Test), sl.cs(sigDF)], axis=1).plot()
        #sl.cs(df_real_price_test_DF_Test).plot()
        #sl.cs(sigDF).plot()
        #scoreDF.plot()
        #plt.show()
        #time.sleep(3000)

        sigDF = sigDF["Predicted_Test_" + selection]
        sigDF = sl.sign(sigDF)

        dfPnl = pd.concat([df_real_price_test_DF_Test, sigDF], axis=1)
        dfPnl.columns = ["real_price", "predicted_price"]

        pnl = dfPnl["real_price"] * dfPnl["predicted_price"]
        pnl = pnl.dropna()
        sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
        print(sh_pnl)

        sl.cs(pnl).plot()
        plt.show()

if __name__ == '__main__':

    #runRegression("ClassicPortfolios", 'Main', "runParallel")
    #runRegression("ClassicPortfolios", 'Main', "report")
    #runRegression("Projections", 'Main', "runParallel")
    #runRegression("Projections", 'Main', "report")
    #runRegression('Projections', 'ScanNotProcessed', "")
    #runRegression("globalProjections", 'Main', "runSerial")
    #runRegression("globalProjections", 'Main', "runParallel")
    #runRegression("globalProjections", 'Main', "report")
    runRegression('globalProjections', 'ScanNotProcessed', "")
    #runRegression("Finalists", 'Main', "runParallel")
    #runRegression("Finalists", 'Main', "report")

    #Test("run")
    #Test("read")