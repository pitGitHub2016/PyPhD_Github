from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
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
kernelList = ["RBF"]

calcMode = 'run'
pnlCalculator = 0

def GPRlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    kernelIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", kernelIn, ", ", rw)
    try:
        if calcMode == 'run':
            GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)

            GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
            GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

            pickle.dump(GPR_Results[2], open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) +".p", "wb"))

        elif calcMode == 'read':

            GPR_Results = [pd.read_sql('SELECT * FROM ' + selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn).set_index('Dates', drop=True),
                           pd.read_sql('SELECT * FROM ' + selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn).set_index('Dates', drop=True)]

        if pnlCalculator == 0:
            sig = sl.sign(GPR_Results[1])
            pnl = sig * GPR_Results[0]
        elif pnlCalculator == 1:
            sig = sl.S(sl.sign(GPR_Results[1]))
            pnl = sig * GPR_Results[0]
        elif pnlCalculator == 2:
            sig = sl.sign(GPR_Results[1])
            pnl = sig * sl.S(GPR_Results[0], nperiods=1)

        reportSh = np.sqrt(252) * sl.sharpe(pnl)
        print(reportSh)
        #time.sleep(30000)

        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

    except Exception as e:
        print(e)

def GPRonPortfolios(Portfolios, scanMode, mode):
    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
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
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
             globalProjectionsList.append(pd.read_sql('SELECT * FROM globalProjectionsDF_'+manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif Portfolios == 'Specific':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
            set(["PCA_250_0","PCA_250_1","PCA_250_2","PCA_250_19",
             "PCA_ExpWindow25_1","PCA_ExpWindow25_2",
             "PCA_150_19","PCA_100_19","PCA_ExpWindow25_18",
             "PCA_25_6","PCA_ExpWindow25_13","PCA_250_8", "PCA_150_18",
             "PCA_25_16","PCA_150_8", "PCA_ExpWindow25_4", "PCA_250_13", "PCA_ExpWindow25_11",
             "PCA_100_7", "PCA_150_5", "PCA_150_10",
             "LLE_250_0","LLE_250_1","LLE_250_2",
             "LLE_ExpWindow25_1", "LLE_ExpWindow25_2",
             "LLE_ExpWindow25_17", "LLE_25_11", "LLE_250_18","LLE_ExpWindow25_14",
             "LLE_150_1", "LLE_250_9", "LLE_25_15", "LLE_ExpWindow25_4",
             "LLE_ExpWindow25_11", "LLE_100_11", "LLE_150_3", "LLE_150_2", "LLE_100_4",
             "LLE_250_5", "LLE_250_11", "LLE_150_6", "LLE_25_7"])] #"PCA_ExpWindow25_0","PCA_ExpWindow25_19","LLE_ExpWindow25_0","LLE_ExpWindow25_18",
    elif Portfolios == 'FinalistsProjections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_0', 'PCA_ExpWindow25_19', 'LLE_ExpWindow25_0', "LLE_ExpWindow25_18"]]
    elif Portfolios == 'FinalistsGlobalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            globalProjectionsList.append(
                pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)[["PCA_ExpWindow25_5_Head", "PCA_ExpWindow25_5_Tail", "LLE_ExpWindow25_5_Head", "LLE_ExpWindow25_5_Tail",
                                                                     "PCA_ExpWindow25_3_Head","PCA_ExpWindow25_3_Tail", "LLE_ExpWindow25_3_Head", "LLE_ExpWindow25_3_Tail"]]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for kernelIn in kernelList:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, kernelIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(GPRlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for kernelIn in kernelList:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        #pnl /= pnl.std() * 100

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252 * 100).set_index("index",drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)
                        stats = pd.concat([statsMat.iloc[0, :], statsMat.iloc[1, :]], axis=0)
                        stats.index = ["GPR_sh", "GPR_Mean", "GPR_tConf", "GPR_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["GPR_tConf", "RW_tConf"]] = stats[["GPR_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["kernel"] = kernelIn

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw))
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(4)
            shDF.to_sql(Portfolios + '_sh_GPR_pnl_' + str(rw), conn, if_exists='replace')
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_GPR_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        rw = 250
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_GPR_' + str(rw), conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_GPR_pnl_")
            selection = splitInfo[0]
            try:
                targetSel = float(selection.split("_")[2])
            except:
                targetSel = 1000000000000000
            kernelIn = str(splitInfo[1]).split("_")[0] + "_" + str(splitInfo[1]).split("_")[1]
            if (targetSel <= 5):
                processList.append([selection, allProjectionsDF[selection], 0.3, kernelIn, rw])

        print("#GPR Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(GPRlocal, tqdm(processList))
        p.close()
        p.join()

    elif scanMode == 'ReportStatistics':
        shGPR = []
        rw = 250
        for kernelIn in ["RBF_DotProduct", "RBF_Matern", "RBF_RationalQuadratic", "RBF_WhiteKernel"]:
            for selection in tqdm(allProjectionsDF.columns):
                try:
                    pnl = pd.read_sql('SELECT * FROM ' + selection + '_GPR_pnl_' + kernelIn + '_' + str(rw),
                                      conn).set_index('Dates', drop=True).iloc[round(0.3 * len(allProjectionsDF)):]
                    pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl).round(4)).reset_index()
                    pnlSharpes['kernelIn'] = kernelIn

                    tConfDf_gpr = sl.tConfDF(pnl.fillna(0)).set_index("index", drop=True)

                    pnlSharpes = pnlSharpes.set_index("index", drop=True)
                    pnlSharpes = pd.concat(
                        [pnlSharpes, pnl.mean() * 100, tConfDf_gpr.astype(str), pnl.std() * 100], axis=1)
                    pnlSharpes.columns = ["pnlSharpes", "kernelIn", "pnl_mean", "tConfDf_sema", "pnl_std"]
                    pnlSharpes['selection'] = selection
                    pnlSharpes = pnlSharpes.set_index("selection", drop=True)
                    shGPR.append(pnlSharpes)
                except:
                    pass

        shGprDF = pd.concat(shGPR).round(4)
        shGprDF.to_sql('GPR_pnlSharpes_' + Portfolios, conn, if_exists='replace')

    elif scanMode == 'ReportSpecificStatistics':
        stats = pd.read_sql('SELECT * FROM GPR_pnlSharpes_'+Portfolios, conn)
        stats = stats[(stats['selection'].str.split("_").str[2].astype(float)<5)&(stats['kernelIn']=="RBF_DotProduct")].set_index("selection", drop=True)
        stats.to_sql('GPR_SpecificStatistics_' + Portfolios, conn, if_exists='replace')

def Test(mode):
    if mode == 'GPC':
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

    elif mode == 'GPR':
        selection = 'PCA_ExpWindow25_2'
        trainLength = 0.9
        kernelIn = "RBF_Matern"
        rw = 10
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
        GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)
        GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
        GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn,
                              if_exists='replace')

        pickle.dump(GPR_Results[2],
                    open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) + ".p", "wb"))

        sig = sl.sign(GPR_Results[1])

        pnl = sig * GPR_Results[0]
        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

#####################################################
#GPRonPortfolios("ClassicPortfolios", 'Main', "run")
#GPRonPortfolios("ClassicPortfolios", 'Main', "report")
#GPRonPortfolios("Projections", 'Main', "run")
#GPRonPortfolios("Projections", 'Main', "report")
#GPRonPortfolios("Projections", "ScanNotProcessed", "")
#GPRonPortfolios("globalProjections", 'Main', "run")
#GPRonPortfolios("globalProjections", 'Main', "report")
#GPRonPortfolios("globalProjections", "ScanNotProcessed", "")
#GPRonPortfolios("Projections", "ReportStatistics", "")
#GPRonPortfolios("Projections", "ReportSpecificStatistics", "")
#GPRonPortfolios("FinalistsProjections", 'Main', "run")
#GPRonPortfolios("FinalistsProjections", 'Main', "report")
#GPRonPortfolios("FinalistsProjections", 'ScanNotProcessed', "")
#GPRonPortfolios("FinalistsGlobalProjections", 'Main', "run")
#GPRonPortfolios("FinalistsGlobalProjections", 'Main', "report")
#GPRonPortfolios("FinalistsGlobalProjections", 'ScanNotProcessed', "")
GPRonPortfolios("Specific", 'Main', "run")
#GPRonPortfolios("Specific", 'Main', "report")
#GPRonPortfolios("Specific", "ScanNotProcessed", "")
