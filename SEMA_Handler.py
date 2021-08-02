from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy import stats as st

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData_sema.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']
lagList = [2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250]

# MegaScale = pd.read_sql('SELECT * FROM Edf_classic', conn).set_index('Dates', drop=True).mean().values[0]

def semaOnProjections(space, mode):
    if space == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif space == 'LLE_Temporal':
        allProjectionsDF = pd.read_sql('SELECT * FROM LLE_Temporal_allProjectionsDF',
                                       sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True)
    elif space == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            globalProjectionsList.append(
                pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif space == 'ClassicPortfolios':
        allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["RP"]
        allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    elif space == 'Finalists':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
            ['PCA_ExpWindow25_0', 'PCA_ExpWindow25_2']]

    # allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'ExpWindow25' not in x]]

    if mode == 'Direct':

        print("Sema on Projections")
        shSema = []
        for Lag in lagList:
            print(Lag)

            rw_pnl = (sl.S(sl.sign(allProjectionsDF)) * allProjectionsDF).fillna(0)
            rw_pnlSharpes = np.sqrt(252) * sl.sharpe(rw_pnl).round(4)

            sema_sig = sl.sign(sl.ema(allProjectionsDF, nperiods=Lag)) * (-1)
            sema_sig.to_sql('sema_sig_' + str(Lag), conn, if_exists='replace')

            pnl = (sl.S(sema_sig) * allProjectionsDF).fillna(0)
            pnl.to_sql('semapnl' + str(Lag), conn, if_exists='replace')

            pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl)).round(2).reset_index()
            pnlSharpes['Lag'] = Lag

            tConfDf_rw = sl.tConfDF(rw_pnl, scalingFactor=252 * 100).set_index("index", drop=True)
            tConfDf_sema = sl.tConfDF(pnl, scalingFactor=252 * 100).set_index("index", drop=True)

            tPairsList = []
            for c in pnl.columns:
                ttestPair = st.ttest_ind(pnl[c].values, rw_pnl[c].values, equal_var=False)

                pnl_ttest_0 = st.ttest_1samp(pnl[c].values, 0)
                rw_pnl_ttest_0 = st.ttest_1samp(rw_pnl[c].values, 0)
                tPairsList.append([c, np.round(ttestPair.pvalue, 2), pnl_ttest_0.pvalue, rw_pnl_ttest_0.pvalue])
            tPairsDF = pd.DataFrame(tPairsList, columns=['index', 'ttestPair_pvalue', 'pnl_ttest_0_pvalue',
                                                         'rw_pnl_ttest_0_pvalue']).set_index("index", drop=True)

            pnlSharpes = pnlSharpes.set_index("index", drop=True)
            pnlSharpes = pd.concat(
                [pnlSharpes, pnl.mean() * 100 * 252, tConfDf_sema.astype(str), pnl.std() * 100 * np.sqrt(252),
                 rw_pnlSharpes, rw_pnl.mean() * 100 * 252, tConfDf_rw.astype(str), rw_pnl.std() * 100 * np.sqrt(252),
                 tPairsDF], axis=1)
            pnlSharpes.columns = ["pnlSharpes", "Lag", "pnl_mean", "tConfDf_sema", "pnl_std",
                                  "rw_pnl_sharpe", "rw_pnl_mean", "tConfDf_rw", "rw_pnl_std", "ttestPair_pvalue",
                                  "pnl_ttest_0_pvalue", "rw_pnl_ttest_0_pvalue"]
            shSema.append(pnlSharpes)

        shSemaDF = pd.concat(shSema).round(2)
        shSemaDF.to_sql('semapnlSharpes_' + space, conn, if_exists='replace')

def TCA():
    # selection = 'LO'; Lag = 2; co = 'LO'; rev = 1
    # selection = 'RP'; Lag = 2; co = 'RP'; rev = 1

    selection = 'PCA_250_19'; Lag = 5; co = 'single'; rev = -1
    # selection = 'PCA_150_19'; Lag = 2; co = 'single'; rev = -1
    # selection = 'PCA_100_0'; Lag = 2; co = 'single'; rev = 1
    # selection = 'PCA_100_4_Tail'; Lag = 2; co = 'global_PCA'; rev = -1

    #selection = 'PCA_ExpWindow25_19'; Lag = 2; co = 'single'; rev = -1
    # selection = 'PCA_ExpWindow25_2'; Lag = 2; co = 'single'; rev = -1
    #selection = 'PCA_ExpWindow25_4_Head'; Lag = 2; co = 'global_PCA'; rev = 1

    #selection = 'PCA_250_0'; Lag = "EMA_100_LLE_Temporal_250_0"; co = 'single'; rev = 1
    #selection = "PCA_100_8"; Lag = "EMA_2_LLE_Temporal_250_4"; co = 'single'; rev = 1
    #selection = "PCA_ExpWindow25_0"; Lag = "EMA_50_LLE_Temporal_ExpWindow25_0"; co = 'single'; rev = -1
    #selection = "PCA_100_3_Head"; Lag = "EMA_150_LLE_Temporal_25_4"; co = 'global_PCA'; rev = -1
    #selection = "LO"; Lag = "EMA_50_LLE_Temporal_ExpWindow25_0"; co = 'LO'; rev = 1
    #selection = "RP"; Lag = "EMA_150_LLE_Temporal_250_2"; co = 'RP'; rev = 1

    if co == 'single':
        allProjectionsDF = pd.read_csv('allProjectionsDF.csv').set_index('Dates', drop=True)
        prinCompsDF = pd.read_sql(
            'SELECT * FROM ' + selection.split('_')[0] + '_principalCompsDf_tw_' + selection.split('_')[1] + '_' +
            selection.split('_')[2], sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True)
    elif co.split("_")[0] == 'global':
        allProjectionsDF = pd.read_csv('globalProjectionsDF_' + co.split("_")[1] + '.csv').set_index('Dates', drop=True)
        prinCompsList = []
        for pr in range(int(selection.split("_")[2])):
            prinCompsList.append(pd.read_sql(
                'SELECT * FROM ' + selection.split('_')[0] + '_principalCompsDf_tw_' + selection.split('_')[1] + '_' +
                str(pr), sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True))
        prinCompsDF = prinCompsList[0]
        for l in range(1, len(prinCompsList)):
            prinCompsDF += prinCompsList[l]
    elif co == 'LO':
        allProjectionsDF = pd.read_sql('SELECT * FROM LongOnlyEWPEDf',
                                       sqlite3.connect('FXeodData_FxData.db')).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["LO"]
        prinCompsDF = sl.sign(
            pd.read_sql('SELECT * FROM riskParityVol_tw_250', sqlite3.connect('FXeodData_FxData.db')).set_index('Dates',
                                                                                                                drop=True).abs())
    elif co == 'RP':
        allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250',
                                       sqlite3.connect('FXeodData_FxData.db')).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["RP"]
        prinCompsDF = 1 / pd.read_sql('SELECT * FROM riskParityVol_tw_250',
                                      sqlite3.connect('FXeodData_FxData.db')).set_index('Dates', drop=True)

    TCspecs = pd.read_csv("TCA.csv").set_index('Asset', drop=True)

    if type(Lag) == str and "LLE_Temporal" in Lag:
        sig = pd.DataFrame(pd.read_sql('SELECT * FROM storedSigDF_EMA', sqlite3.connect('FXeodData_LLE_Temporal.db')).set_index('Dates', drop=True)[Lag]) * rev
        sig.columns = [selection]
        sema_pnl = allProjectionsDF.mul(sig, axis=0).fillna(0)
    else:
        sig = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=Lag))) * rev
        sema_pnl = (sig * allProjectionsDF).fillna(0)

    sema_pnl = sema_pnl
    strat_pnl = sema_pnl[selection]
    rawSharpe = (np.sqrt(252) * sl.sharpe(strat_pnl)).round(2)
    print(rawSharpe)

    trW = prinCompsDF.mul(sig[selection], axis=0)
    delta_pos = sl.d(trW).fillna(0)
    netPnL_List = []
    net_SharpeList = []
    for scenario in ['Scenario1', 'Scenario2', 'Scenario3', 'Scenario4', 'Scenario5', 'Scenario6']:
        my_tcs = delta_pos.copy()
        for c in my_tcs.columns:
            my_tcs[c] = my_tcs[c].abs() * TCspecs.loc[TCspecs.index == c, scenario].values[0]
        strat_pnl_afterCosts = strat_pnl - sl.rs(my_tcs)
        strat_pnl_afterCosts.name = scenario
        netPnL_List.append(strat_pnl_afterCosts)
        net_Sharpe = (np.sqrt(252) * sl.sharpe(strat_pnl_afterCosts)).round(2)
        net_SharpeList.append(net_Sharpe)
    strat_pnl_afterCosts_DF = pd.concat(netPnL_List, axis=1)
    print(strat_pnl_afterCosts_DF)
    pickle.dump(strat_pnl_afterCosts_DF,open("Repo/FinalPortfolio/EMA_" + selection + "_" + str(Lag) + "_" + co + ".p", "wb"))
    print("net_SharpeList")
    print(' & '.join([str(x) for x in net_SharpeList]))

#####################################################
# semaOnProjections("ClassicPortfolios", "Direct")
# semaOnProjections("Projections", "Direct")
# semaOnProjections("LLE_Temporal", "Direct")
# semaOnProjections("globalProjections", "Direct")
TCA()
