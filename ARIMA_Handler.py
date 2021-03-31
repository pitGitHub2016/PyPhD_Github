from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
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

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']
lagList = [2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250]

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    orderIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", orderIn, ", ", rw)

    Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn, rw)

    Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')
    Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')

    pickle.dump(Arima_Results[2], open(selection + '_ARIMA_arparamList_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) +".p", "wb"))

    sig = sl.sign(Arima_Results[1])

    pnl = sig * Arima_Results[0]
    pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn, if_exists='replace')

def ARIMAonPortfolios(Portfolios, scanMode, mode):
    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        orderList = [1,3,5]
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
             globalProjectionsList.append(pd.read_sql('SELECT * FROM globalProjectionsDF_'+manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
        orderList = [1,3,5]
    elif Portfolios == 'ClassicPortfolios':
        allPortfoliosList = []
        for tw in twList:
            subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True)
            subDF.columns = ["RP_"+str(tw)]
            allPortfoliosList.append(subDF)
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        allPortfoliosList.append(LOportfolio)
        allProjectionsDF = pd.concat(allPortfoliosList, axis=1)
        orderList = [1, 3, 5]
    elif Portfolios == 'Finalists':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_2', 'LLE_ExpWindow25']]
        orderList = [1]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for OrderP in orderList:
                orderIn = (OrderP, 0, 0)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, orderIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for OrderP in orderList:
                orderIn = (OrderP, 0, 0)
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(4)
                        MEANs = pnl.mean() * 100
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0)).set_index("index", drop=True)
                        STDs = pnl.std() * 100

                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)
                        stats = pd.concat([statsMat.iloc[0,:], statsMat.iloc[1,:]], axis=0)
                        stats.index = ["ARIMA_sh", "ARIMA_Mean", "ARIMA_tConf", "ARIMA_Std", "RW_sh", "RW_Mean", "RW_tConf", "RW_Std"]
                        stats[["ARIMA_tConf", "RW_tConf"]] = stats[["ARIMA_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["order"] = str(orderIn[0])+str(orderIn[1])+str(orderIn[2])

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw))
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(4)
            shDF.to_sql(Portfolios+'_sh_ARIMA_pnl_' + str(rw), conn, if_exists='replace')
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF'+ '_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        rw = 250
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF'+ '_' + str(rw), conn).set_index('index', drop=True)

        for idx, row in notProcessedDF.iterrows():
            print(row)
            time.sleep(300)
            splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
            selection = splitInfo[0]
            if 1==1:
            #if float(selection.split("_")[2]) <= 5:
                orderStr = str(splitInfo[1])
                orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
                processList.append([selection, allProjectionsDF[selection], 0.3, orderIn, rw])

        print("#ARIMA Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(ARIMAlocal, tqdm(processList))
        p.close()
        p.join()

    elif scanMode == 'ReportSpecificStatistics':
        rw = 250
        stats = pd.read_sql('SELECT * FROM '+Portfolios+'_sh_ARIMA_pnl_'+str(rw), conn)
        stats = stats[(stats['selection'].str.split("_").str[2].astype(float)<5)].set_index("selection", drop=True).round(4)
        stats.to_sql('ARIMA_SpecificStatistics_' + Portfolios, conn, if_exists='replace')

def Test():
    selection = 'PCA_ExpWindow25_2'
    trainLength = 0.3
    tw = 250
    df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
    rwDF = pd.read_sql('SELECT * FROM PCA_randomWalkPnlRSprojections_tw_ExpWindow25', conn).set_index('Dates',
                                                                                                      drop=True).iloc[
           round(0.3 * len(df)):, 2]
    medSh = (np.sqrt(252) * sl.sharpe(rwDF)).round(4)
    print("Random Walk Sharpe : ", medSh)
    # GaussianProcess_Results = sl.GPC_Walk(df, trainLength, tw)
    magicNum = 1
    params = {
        "TrainWindow": 5,
        "LearningMode": 'static',
        "Kernel": "DotProduct",
        "modelNum": magicNum,
        "TrainEndPct": 0.3,
        "writeLearnStructure": 0
    }
    out = sl.AI.gGPC(df, params)

    out[0].to_sql('df_real_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    out[1].to_sql('df_predicted_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    out[2].to_sql('df_predicted_proba_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    df_real_price = out[0]
    df_predicted_price = out[1]
    df_predicted_price.columns = df_real_price.columns
    # Returns Prediction
    sig = sl.sign(df_predicted_price)
    pnl = sig * df_real_price
    pnl.to_sql('pnl_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn, if_exists='replace')
    print("pnl_GPC_TEST_sharpe = ", np.sqrt(252) * sl.sharpe(pnl))
    sl.cs(pnl).plot()
    print(out[2].tail(10))
    out[2].plot()
    plt.show()

#####################################################

#ARIMAonPortfolios("ClassicPortfolios", 'Main', "run")
#ARIMAonPortfolios("ClassicPortfolios", 'Main', "report")
#ARIMAonPortfolios("Projections", 'Main', "run")
#ARIMAonPortfolios("Projections", 'Main', "report")
#ARIMAonPortfolios("Projections", "ScanNotProcessed", "")
ARIMAonPortfolios("globalProjections", 'Main', "run")
#ARIMAonPortfolios("globalProjections", 'Main', "report")
#ARIMAonPortfolios("globalProjections", "ScanNotProcessed", "")
#ARIMAonPortfolios("Projections", "ReportSpecificStatistics", "")
#ARIMAonPortfolios("Finalists", 'Main', "run")
