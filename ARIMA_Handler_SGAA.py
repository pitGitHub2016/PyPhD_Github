from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
from scipy import stats as st
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('SmartGlobalAssetAllocation.db')

rw = 250
pnlCalculator = 0

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    orderIn = argList[3]
    rw = argList[4]
    scenarioNumber = argList[5]
    print(selection, ",", trainLength, ",", orderIn, ", ", rw, ", ", scenarioNumber)

    try:
        Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn, rw)

        Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) + "_" + str(scenarioNumber), conn,
                                if_exists='replace')
        Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw)+ "_" + str(scenarioNumber), conn,
                                if_exists='replace')

        pickle.dump(Arima_Results[2], open(selection + '_ARIMA_arparamList_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw)
                                           + "_" + str(scenarioNumber) + ".p", "wb"))

        if pnlCalculator == 0:
            sig = sl.sign(Arima_Results[1])
            pnl = sig * Arima_Results[0]
        elif pnlCalculator == 1:
            sig = sl.S(sl.sign(Arima_Results[1]))
            pnl = sig * Arima_Results[0]
        elif pnlCalculator == 2:
            sig = sl.sign(Arima_Results[1])
            pnl = sig * sl.S(Arima_Results[0], nperiods=-1)

        pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) + "_" + str(scenarioNumber), conn, if_exists='replace')

    except Exception as e:
        print(e)

def ARIMAonPortfolios(scenarioNumber, scanMode, mode):
    allProjectionsDF = pd.read_sql('SELECT * FROM pnlDF_Scenario_'+str(scenarioNumber), conn).set_index('Dates', drop=True)
    orderList = [[1, 3], [0, 1]]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for OrderP in orderList[0]:
                for OrderQ in orderList[1]:
                    orderIn = (OrderP, 0, OrderQ)
                    for selection in allProjectionsDF.columns:
                        processList.append([selection, allProjectionsDF[selection], 0.3, orderIn, rw, scenarioNumber])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            shList = []
            for OrderP in orderList[0]:
                for OrderQ in orderList[1]:
                    orderIn = (OrderP, 0, OrderQ)
                    for selection in allProjectionsDF.columns:
                        try:
                            pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw) + "_" + str(scenarioNumber),
                                              conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                            pnl.columns = [selection]
                            pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                            sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                            MEANs = (252 * pnl.mean() * 100).round(2)
                            tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252*100).set_index("index", drop=True).round(2)
                            STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                            ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                            statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                            stats = pd.concat([statsMat.iloc[0,:], statsMat.iloc[1,:]], axis=0)
                            stats.index = ["ARIMA_sh", "ARIMA_Mean", "ARIMA_tConf", "ARIMA_Std", "RW_sh", "RW_Mean", "RW_tConf", "RW_Std"]
                            stats[["ARIMA_tConf", "RW_tConf"]] = stats[["ARIMA_tConf", "RW_tConf"]].astype(str)
                            stats["selection"] = selection
                            stats["ttestPair_statistic"] = np.round(ttestPair.statistic,2)
                            stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue,2)
                            stats["order"] = str(orderIn[0])+str(orderIn[1])+str(orderIn[2])

                            shList.append(stats)
                        except Exception as e:
                            print(e)
                            notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw)+ "_" + str(scenarioNumber))
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(2)
            shDF.to_sql(str(scenarioNumber)+'_sh_ARIMA_pnl_' + str(rw), conn, if_exists='replace')

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(str(scenarioNumber)+'_notProcessedDF'+ '_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        notProcessedDF = pd.read_sql('SELECT * FROM '+str(scenarioNumber)+'_notProcessedDF'+ '_' + str(rw), conn).set_index('index', drop=True)

        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
            selection = splitInfo[0]
            orderStr = str(splitInfo[1])
            orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
            processList.append([selection, allProjectionsDF[selection], 0.3, orderIn, rw])

        print("#ARIMA Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(ARIMAlocal, tqdm(processList))
        p.close()
        p.join()

for i in range(30):
    print("Running ARIMA Scenario : ", i)
    try:
        ARIMAonPortfolios(i, 'Main', "run")
        ARIMAonPortfolios(i, 'Main', "report")
    except Exception as e:
        print(e)
#ARIMAonPortfolios(0, "ScanNotProcessed", "")