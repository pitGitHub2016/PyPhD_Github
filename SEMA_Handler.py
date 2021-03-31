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

def semaOnProjections(space, mode):

    if space == "":
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        allProjectionsDF = allProjectionsDF.iloc[round(0.3*len(allProjectionsDF)):]
        ExPostOpt_allProjectionsDF = sl.ExPostOpt(pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True))[0]
    elif space == 'global':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            sub_globalProjectionsDF = pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index(
                'Dates', drop=True)
            globalProjectionsList.append(sub_globalProjectionsDF.iloc[round(0.3 * len(sub_globalProjectionsDF)):])
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)

    if mode == 'Direct':

        print("Sema on Projections")
        shSema = []
        for Lag in lagList:
            print(Lag)
    
            allProjectionsDF_pnlSharpes = np.sqrt(252) * sl.sharpe(allProjectionsDF).round(4)
    
            rw_pnl = sl.S(sl.sign(allProjectionsDF)) * allProjectionsDF
            rw_pnlSharpes = np.sqrt(252) * sl.sharpe(rw_pnl).round(4)
    
            pnl = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=Lag))) * allProjectionsDF
            pnl.to_sql('semapnl'+str(Lag), conn, if_exists='replace')
    
            pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl).round(4)).reset_index()
            pnlSharpes['Lag'] = Lag
    
            tConfDf = sl.tConfDF(allProjectionsDF).set_index("index", drop=True)
            tConfDf_rw = sl.tConfDF(rw_pnl.fillna(0)).set_index("index", drop=True)
            tConfDf_sema = sl.tConfDF(pnl.fillna(0)).set_index("index", drop=True)
    
            pnlSharpes = pnlSharpes.set_index("index", drop=True)
            pnlSharpes = pd.concat([allProjectionsDF_pnlSharpes, allProjectionsDF.mean()*100, tConfDf.astype(str), allProjectionsDF.std()*100,
                                    rw_pnlSharpes, rw_pnl.mean()*100, tConfDf_rw.astype(str),rw_pnl.std()*100,
                                    pnlSharpes, pnl.mean()*100, tConfDf_sema.astype(str), pnl.std()*100], axis=1)
            pnlSharpes.columns = ["allProjectionsDF_pnlSharpes", "allProjectionsDF_mean", "tConfDf", "allProjectionsDF_std",
                                    "rw_pnlSharpes", "rw_pnl_mean", "tConfDf_rw", "rw_pnl_std",
                                    "pnlSharpes", "Lag", "pnl_mean", "tConfDf_sema", "pnl_std"]
            shSema.append(pnlSharpes)
    
        shSemaDF = pd.concat(shSema).round(4)
        shSemaDF.to_sql('semapnlSharpes_'+space, conn, if_exists='replace')

    elif mode == 'BasketsCombos':
        from itertools import combinations
        shList = []
        for combos in [2,3]:
            print(combos)
            cc = list(combinations(allProjectionsDF.columns, combos))
            for c in cc:
                print(c)
                BasketDF = allProjectionsDF[c[0]] + allProjectionsDF[c[1]]
                shBasket = np.sqrt(252) * sl.sharpe(BasketDF).round(4)
                exPostBasketDF = ExPostOpt_allProjectionsDF[c[0]] + ExPostOpt_allProjectionsDF[c[1]]
                exPostshBasket = np.sqrt(252) * sl.sharpe(exPostBasketDF).round(4)
                for Lag in lagList:
                    print(Lag)
                    semaBasketDF = sl.S(sl.sign(sl.ema(BasketDF, nperiods=Lag))) * BasketDF
                    shSemaBasket = np.sqrt(252) * sl.sharpe(semaBasketDF).round(4)
                    semaBasketDF_ExPost = sl.S(sl.sign(sl.ema(exPostBasketDF, nperiods=Lag))) * exPostBasketDF
                    shSemaBasket_ExPost = np.sqrt(252) * sl.sharpe(semaBasketDF_ExPost).round(4)
                    shList.append([c[0]+"_"+c[1], shBasket, exPostshBasket, Lag, shSemaBasket, shSemaBasket_ExPost])

        sh_Baskets = pd.DataFrame(shList, columns=['Basket', 'ShBasket', 'exPostshBasket', 'Lag', 'ShSemaBasket', 'shSemaBasket_ExPost'])
        sh_Baskets.to_sql('sh_Baskets_' + str(combos), conn, if_exists='replace')

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
        trainLength = 0.3
        tw = 250
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
        out = sl.GPR_Walk(df, 0.3, "RBF_DotProduct", 250)

#####################################################
#semaOnProjections("", "Direct")
semaOnProjections("global", "Direct")
#semaOnProjections("", "BasketsCombos")
