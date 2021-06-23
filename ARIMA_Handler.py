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

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']

pnlCalculator = 0

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    orderIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", orderIn, ", ", rw)

    try:
        Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn, rw)

        Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                                if_exists='replace')
        Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                                if_exists='replace')

        pickle.dump(Arima_Results[2], open(selection + '_ARIMA_arparamList_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) +".p", "wb"))

        if pnlCalculator == 0:
            sig = sl.sign(Arima_Results[1])
            pnl = sig * Arima_Results[0]
        elif pnlCalculator == 1:
            sig = sl.S(sl.sign(Arima_Results[1]))
            pnl = sig * Arima_Results[0]
        elif pnlCalculator == 2:
            sig = sl.sign(Arima_Results[1])
            pnl = sig * sl.S(Arima_Results[0], nperiods=-1)

        pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn, if_exists='replace')

    except Exception as e:
        print(e)
        print("General break-error...")

def ARIMAonPortfolios(Portfolios, scanMode, mode):

    if Portfolios == 'Projections':
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
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_250_0', 'LLE_250_0', 'PCA_250_19', 'LLE_250_18']]

    #allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'ExpWindow25' not in x]]

    orderList = [(1, 0, 0), (2, 0, 0), (3, 0, 0)]
    #orderList = [(2, 0, 1)]
    startPct = 0.1
    rw = 250
    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for orderIn in orderList:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], startPct, orderIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            shList = []
            for orderIn in orderList:
                for selection in allProjectionsDF.columns:
                    print(selection)
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(startPct*len(allProjectionsDF)):]
                        AR_paramList = pickle.load( open( selection+"_ARIMA_arparamList_"+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_'+str(rw)+".p", "rb" ) )
                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252*100).set_index("index", drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                        pnl_ttest_0 = st.ttest_1samp(pnl[selection].values, 0)
                        rw_pnl_ttest_0 = st.ttest_1samp(pnl['RW'].values, 0)
                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                        stats = pd.concat([statsMat.iloc[0,:], statsMat.iloc[1,:]], axis=0)
                        stats.index = ["ARIMA_sh", "ARIMA_Mean", "ARIMA_tConf", "ARIMA_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["ARIMA_tConf", "RW_tConf"]] = stats[["ARIMA_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["order"] = str(orderIn[0])+str(orderIn[1])+str(orderIn[2])
                        stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue,2)
                        stats["pnl_ttest_0_pvalue"] = np.round(pnl_ttest_0.pvalue, 2)
                        stats["rw_pnl_ttest_0_value"] = np.round(rw_pnl_ttest_0.pvalue, 2)

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw))
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(2)
            shDF.to_sql(Portfolios+'_sh_ARIMA_pnl_' + str(rw), conn, if_exists='replace')

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            print("Len notProcessedDF = ", len(notProcessedDF))
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF'+ '_' + str(rw), conn).set_index('index', drop=True)

        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
            selection = splitInfo[0]
            orderStr = str(splitInfo[1])
            orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
            processList.append([selection, allProjectionsDF[selection], startPct, orderIn, rw])

        print("#ARIMA Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(ARIMAlocal, tqdm(processList))
        p.close()
        p.join()

    elif scanMode == 'ReportSpecificStatistics':
        stats = pd.read_sql('SELECT * FROM '+Portfolios+'_sh_ARIMA_pnl_'+str(rw), conn)
        stats = stats[(stats['selection'].str.split("_").str[2].astype(float)<5)].set_index("selection", drop=True).round(4)
        stats.to_sql('ARIMA_SpecificStatistics_' + Portfolios, conn, if_exists='replace')

def Test(mode):
    if mode == 'test0':

        from quantstats import stats as qs
        from scipy.stats import norm
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
            ['PCA_ExpWindow25_2']]
        rw_pnl = sl.S(sl.sign(allProjectionsDF)) * allProjectionsDF
        semaPnL = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=3))) * allProjectionsDF
        pnl = pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_ARIMA_pnl_100_250',  conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
        pnlAll = pd.concat([rw_pnl, semaPnL, pnl], axis=1).dropna()
        pnlAll.columns = ['PCA_ExpWindow25_2_RandomWalk', 'PCA_ExpWindow25_2_Sema', 'PCA_ExpWindow25_2_ARIMA_pnl_100_250']

        ttestRV = sl.ttestRV(pnlAll)
        print("ttestRV on Returns : ")
        print(ttestRV)
        rollSharpes = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.sharpe(pd.Series(x))).fillna(0)
        ttestRV_Sharpes = sl.ttestRV(rollSharpes)
        print("ttestRV on Rolling Sharpes : ")
        print(ttestRV_Sharpes)

        rollSharpes.plot()
        plt.show()

        #out = sl.cs(pnlAll)
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.sortino(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.adjusted_sortino(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.avg_loss(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: pd.Series(x).std())
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.ulcer_performance_index(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.risk_of_ruin(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.value_at_risk(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.conditional_value_at_risk(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.tail_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.payoff_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.profit_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.profit_factor(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.cpc_index(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.common_sense_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.outlier_win_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.outlier_win_ratio(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.recovery_factor(pd.Series(x))) # GOOD (*)
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.max_drawdown(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.to_drawdown_series(pd.Series(x)))
        #out = pnlAll.rolling(window=250, center=False).apply(lambda x: qs.kelly_criterion(pd.Series(x)))

def TCA():
    TCspecs = pd.read_excel('TCA.xlsx').set_index('Asset', drop=True)
    selection = 'PCA_250_19'
    #selection = 'PCA_ExpWindow25_19'
    localConn = sqlite3.connect('FXeodDataARIMA.db')

    allProjectionsDF = pd.read_sql('SELECT * FROM '+selection + '_ARIMA_testDF_100_250', localConn).set_index('Dates', drop=True)
    allProjectionsDF.columns = [selection]
    arimaSigCore = pd.read_sql('SELECT * FROM '+selection + '_ARIMA_PredictionsDF_100_250', localConn).set_index('Dates', drop=True)
    #.iloc[round(0.1*len(allProjectionsDF)):]
    sig = sl.sign(arimaSigCore)
    sig.columns = [selection]
    strat_pnl = (sig * allProjectionsDF).iloc[round(0.1*len(allProjectionsDF)):]
    rawSharpe = np.sqrt(252) * sl.sharpe(strat_pnl)
    print(rawSharpe)

    prinCompsDF = pd.read_sql(
        'SELECT * FROM ' + selection.split('_')[0] + '_principalCompsDf_tw_' + selection.split('_')[1] + '_' +
        selection.split('_')[2], sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True)

    # print(prinCompsDF)
    trW = prinCompsDF.mul(sig[selection], axis=0)
    # print(sl.d(trW).tail())
    delta_pos = sl.d(trW).fillna(0)
    for scenario in ['Scenario0', 'Scenario1', 'Scenario2', 'Scenario3']:
        my_tcs = delta_pos.copy()
        for c in my_tcs.columns:
            my_tcs[c] = my_tcs[c].abs() * TCspecs.loc[TCspecs.index == c, scenario].values[0]
        strat_pnl_afterCosts = (strat_pnl - pd.DataFrame(sl.rs(my_tcs), columns=strat_pnl.columns)).dropna()
        after_TCA_Sharpe = np.sqrt(252) * sl.sharpe(strat_pnl_afterCosts)
        print(after_TCA_Sharpe)

#####################################################

#ARIMAonPortfolios("ClassicPortfolios", 'Main', "run")
#ARIMAonPortfolios("ClassicPortfolios", 'Main', "report")
#ARIMAonPortfolios("ClassicPortfolios", 'ScanNotProcessed', "")
#ARIMAonPortfolios("Projections", 'Main', "run")
#ARIMAonPortfolios("Projections", 'Main', "report")
#ARIMAonPortfolios("Projections", "ScanNotProcessed", "")
#ARIMAonPortfolios("globalProjections", 'Main', "run")
#ARIMAonPortfolios("globalProjections", 'Main', "report")
#ARIMAonPortfolios("globalProjections", "ScanNotProcessed", "")
#ARIMAonPortfolios("Projections", "ReportSpecificStatistics", "")
#ARIMAonPortfolios("Finalists", 'Main', "run")
#ARIMAonPortfolios("Finalists", 'Main', "report")

TCA()