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
#MegaScale = pd.read_sql('SELECT * FROM Edf_classic', conn).set_index('Dates', drop=True).mean().values[0]

def semaOnProjections(space, mode):

    if space == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif space == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
             globalProjectionsList.append(pd.read_sql('SELECT * FROM globalProjectionsDF_'+manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif space == 'ClassicPortfolios':
        allProjectionsDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        allProjectionsDF.columns = ["RP"]
        allProjectionsDF["LO"] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    elif space == 'Finalists':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_0', 'PCA_ExpWindow25_2']]

    #allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'ExpWindow25' not in x]]

    if mode == 'Direct':

        print("Sema on Projections")
        shSema = []
        for Lag in lagList:
            print(Lag)

            rw_pnl = (sl.S(sl.sign(allProjectionsDF)) * allProjectionsDF).fillna(0)
            rw_pnlSharpes = np.sqrt(252) * sl.sharpe(rw_pnl).round(4)
    
            pnl = (sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=Lag))) * allProjectionsDF).fillna(0) *(-1)
            pnl.to_sql('semapnl'+str(Lag), conn, if_exists='replace')
    
            pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl)).round(2).reset_index()
            pnlSharpes['Lag'] = Lag

            tConfDf_rw = sl.tConfDF(rw_pnl, scalingFactor=252*100).set_index("index", drop=True)
            tConfDf_sema = sl.tConfDF(pnl, scalingFactor=252*100).set_index("index", drop=True)

            tPairsList = []
            for c in pnl.columns:
                ttestPair = st.ttest_ind(pnl[c].values, rw_pnl[c].values, equal_var=False)

                pnl_ttest_0 = st.ttest_1samp(pnl[c].values, 0)
                rw_pnl_ttest_0 = st.ttest_1samp(rw_pnl[c].values, 0)
                tPairsList.append([c, np.round(ttestPair.pvalue,2), pnl_ttest_0.pvalue, rw_pnl_ttest_0.pvalue])
            tPairsDF = pd.DataFrame(tPairsList, columns=['index', 'ttestPair_pvalue', 'pnl_ttest_0_pvalue', 'rw_pnl_ttest_0_pvalue']).set_index("index", drop=True)

            pnlSharpes = pnlSharpes.set_index("index", drop=True)
            pnlSharpes = pd.concat([pnlSharpes, pnl.mean()*100*252, tConfDf_sema.astype(str), pnl.std()*100*np.sqrt(252),
                                    rw_pnlSharpes, rw_pnl.mean()*100*252, tConfDf_rw.astype(str), rw_pnl.std()*100*np.sqrt(252), tPairsDF], axis=1)
            pnlSharpes.columns = ["pnlSharpes", "Lag", "pnl_mean", "tConfDf_sema", "pnl_std",
                                    "rw_pnl_sharpe", "rw_pnl_mean", "tConfDf_rw", "rw_pnl_std", "ttestPair_pvalue","pnl_ttest_0_pvalue", "rw_pnl_ttest_0_pvalue"]
            shSema.append(pnlSharpes)
    
        shSemaDF = pd.concat(shSema).round(2)
        shSemaDF.to_sql('semapnlSharpes_'+space, conn, if_exists='replace')

def TCA():
    allProjectionsDF = pd.read_csv('allProjectionsDF.csv').set_index('Dates', drop=True)
    TCspecs = pd.read_excel('TCA.xlsx').set_index('Asset', drop=True)
    selection = 'PCA_250_19'
    #selection = 'PCA_ExpWindow25_19'

    sig = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=5)))
    sema_pnl = (sig * allProjectionsDF).fillna(0) *(-1)
    strat_pnl = sema_pnl[selection]
    rawSharpe = np.sqrt(252) * sl.sharpe(strat_pnl)
    print(rawSharpe)

    prinCompsDF = pd.read_sql('SELECT * FROM '+selection.split('_')[0]+'_principalCompsDf_tw_'+selection.split('_')[1]+'_'+selection.split('_')[2], sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True)

    #print(prinCompsDF)
    trW = prinCompsDF.mul(sig[selection], axis=0)
    #print(sl.d(trW).tail())
    delta_pos = sl.d(trW).fillna(0)
    for scenario in ['Scenario0','Scenario1','Scenario2','Scenario3','Scenario4']:
        my_tcs = delta_pos.copy()
        #print(my_tcs.tail())
        for c in my_tcs.columns:
            my_tcs[c] = my_tcs[c].abs() * TCspecs.loc[TCspecs.index==c, scenario].values[0]
        #print(my_tcs.tail())
        #print(sl.rs(my_tcs).tail())
        #time.sleep(3000)
        strat_pnl_afterCosts = strat_pnl - sl.rs(my_tcs)
        after_TCA_Sharpe = np.sqrt(252) * sl.sharpe(strat_pnl_afterCosts)
        print(after_TCA_Sharpe)

#####################################################
#semaOnProjections("ClassicPortfolios", "Direct")
#semaOnProjections("Projections", "Direct")
#semaOnProjections("globalProjections", "Direct")
TCA()
