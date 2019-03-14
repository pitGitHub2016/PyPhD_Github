from TorusEngine import Torus as tr
import numpy as np
import itertools
#from fuzzywuzzy import fuzz
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_finance import plot_day_summary_oclh as pdso
from flask import Flask, render_template, jsonify
import time
import os, time
from mpl_finance import candlestick2_ohlc as candle
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

' FMA Torus Testing Script'

df0 = tr.active_teams(tr.getDBdf('sqlite', 'Football_Data_co_uk.db', 'SELECT * FROM FDCUk_TeamsHome_PremierLeague').set_index('index'))[0]
#df0 = tr.getDBdf('sqlite', 'Football_Data_co_uk.db', 'SELECT * FROM BetmechsTopPremierLeague').set_index('index')
window = 30

"""

Draw_cols = [col for col in df0.columns if ('-D') in col]
FTU25_cols = [col for col in df0.columns if ('-FTU25') in col]
FTO25_cols = [col for col in df0.columns if ('-FTO25') in col]
H_cols = [col for col in df0.columns if ('-H') in col]
A_cols = [col for col in df0.columns if ('-A') in col]

df0 = df0[Draw_cols+A_cols]

df = tr.rs(df0)
#tr.cs(df).plot(); plt.show()

meanX = tr.roller(df, np.mean, window)
print(meanX[np.abs(meanX)<0.009])
stdX = tr.roller(df, np.std, window)
print(stdX[np.abs(meanX)<0.009])
print(meanX[np.abs(meanX)<0.009]/stdX[np.abs(meanX)<0.009])

a = stdX / meanX
a = a.replace(np.inf, 1)

#meanX.dropna().plot(); plt.show()
stdX.dropna().plot(); plt.show()
#tr.cs(a).dropna().plot(); plt.show()
"""

dfodds = tr.getDBdf('sqlite', 'Football_Data_co_uk.db', 'SELECT * FROM FDCUk_TeamsHomeOdds_PremierLeague').set_index('index')
#df0[dfodds < 2] = 0
#df0[dfodds > 5] = 0
TOP = df0[df0 != 0].count(); posTOP = TOP[TOP > 230].index.tolist()
df0 = df0[posTOP]

#Draw_cols = [col for col in df0.columns if '-D' in col]
#FTU25_cols = [col for col in df0.columns if ('-FTU25') in col]
#W_cols = [col for col in df0.columns if ('-W') in col]
#L_cols = [col for col in df0.columns if ('-L') in col]
#Away_cols = [col for col in df0.columns if '-A' in col]

#df0 = df0[Draw_cols]
dfIndex = tr.rs(df0)

#signal = tr.S(tr.sign(tr.bema(dfIndex, nperiods=5)))
#RP = tr.S(tr.sign(tr.bema(df0, nperiods=5))) * df0
#tr.BetPnL.tcf(a, bComm=0.05, BLs=0.1)

st = 25; pcaN = 5; eigsPC = [0] #pcaN-1
pcaDA = []; pcaComps = []
for i in range(st, len(df0)):

    df = df0.iloc[i-st:i, :]
    #df = df0.iloc[0:i, :]

    Dates = df.index
    features = df.columns.values
    x = df.loc[:, features].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=pcaN)
    principalComponents = pca.fit_transform(x)
    #if i == st:
    #    print(pca.explained_variance_ratio_)
    wContribs = pd.DataFrame(pca.components_[eigsPC])
    pcaComps.append(wContribs.sum().tolist())

df1 = df0.iloc[st:, :]
principalCompsDf = pd.DataFrame(pcaComps, columns=df0.columns, index=df1.index)#.abs()

################## FILTERS ON WEIGHTS #######################
#principalCompsDf[principalCompsDf < 0.0] = 0

################## PROJECTIONS AND PLOTS ####################

principalCompsDf.iloc[:,0].plot(); plt.show()

#tr.cs(principalCompsDf).plot(); plt.show()
RP = tr.S(principalCompsDf) * df1 * (-1)
#RP = tr.sign(tr.S(principalCompsDf)) * df1

RP = tr.BetPnL.tcf_Lev(RP, bComm=0.05, BLs=0.0, Lev=1)
#RP = pd.DataFrame(tr.rs(RP).dropna())
#meanX = tr.roller(RP, np.mean, window); stdX = tr.roller(RP, np.std, window); a = stdX / meanX; a = a.replace(np.inf, 1); tr.cs(a).dropna().plot(); plt.show()

#RP = tr.rs(tr.BetPnL.ExPostOpt(RP)[0])
#RP = tr.rs(tr.BetPnL.ExPostOpt(tr.S(tr.sign(tr.bema(RP, nperiods=3))) * RP)[0])
#RP = tr.S(tr.sign(tr.bema(RP, nperiods=50))).dropna() * RP

RPcs = tr.cs(RP); RPcs.plot()

plt.title('Strategy Sharpe : ' + str(np.sqrt(len(RP)/14) * tr.sharpe(RP))); plt.show()

