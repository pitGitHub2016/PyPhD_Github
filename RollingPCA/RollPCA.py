from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')

def DataBuilder():
    P = pd.read_csv('P.csv', header=None); P.columns = pd.read_csv('Basket.csv', delimiter=' ')['Names'].tolist()
    P['Dates'] = pd.read_csv('Dates.csv', header=None)
    P['Dates'] = pd.to_datetime(P['Dates'], infer_datetime_format=True)
    P = P.set_index('Dates', drop=True)
    P.to_sql('FxData', conn, if_exists='replace')
    pd.read_csv('BasketTS.csv', delimiter=' ').to_sql('BasketTS', conn, if_exists='replace')

def RunRollPCAOnFXPairs():
    df = pd.read_sql('SELECT * FROM FxData', conn).set_index('Dates', drop=True)
    specNames = pd.read_sql('SELECT * FROM BasketTS', conn)['Names'].tolist(); df = df[specNames]
    df.to_sql('InitialDf', conn, if_exists='replace')
    df = df.replace([np.inf, -np.inf, 0], np.nan).ffill(); df = sl.dlog(df).fillna(0)
    df.to_sql('Dlogs', conn, if_exists='replace')

    out = sl.AI.gRollingPca(df, 50, 5, [0,1,2,3,4])
    out[0].to_sql('df', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql('principalCompsDf_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql('exPostProjections_'+str(k), conn, if_exists='replace')

def semaOnPCAProjections():
    exPostProjections = pd.read_sql('SELECT * FROM exPostProjections_0', conn).set_index('Dates', drop=True)

    pcaEMApnl = []; pcaEMApnlSh = []
    for L in [3, 25, 500]:
        pnl = sl.rs(sl.S(sl.sema(exPostProjections, nperiods=L)) * exPostProjections) * (-1)
        pcaEMApnl.append(pnl)
        pcaEMApnlSh.append(np.sqrt(252) * sl.sharpe(pnl))

    print(np.sqrt(252) * sl.sharpe(sl.rs(exPostProjections)))
    pcaEMApnlShDF = pd.DataFrame(pcaEMApnlSh, columns=['Sharpe Ratio']).round(4)
    print(pcaEMApnlShDF)
    pcaEMApnlDF = pd.concat(pcaEMApnl, axis=1, ignore_index=True); pcaEMApnlDF.columns = ['Ema L 3', 'Ema L 25', 'Ema L 500']

    #fig, ax = plt.subplots(figsize=(19.2, 10.8))
    fig, ax = plt.subplots()
    #sl.cs(exPostProjections).plot(ax=ax)
    #sl.cs(sl.rs(exPostProjections)).plot(ax=ax)
    sl.cs(pcaEMApnlDF).plot(ax=ax)
    plt.legend(); plt.show()

def ARIMAonPCAProjections():
    exPostProjections = pd.read_sql('SELECT * FROM exPostProjections_0', conn).set_index('Dates', drop=True).fillna(0)
    exPostProjections.index = (exPostProjections.index).resample('D')

    Backtest = sl.BacktestPnL.ModelPnL(sl.Models(exPostProjections.copy()).ARIMA_signal(start=100, mode='roll', opt='BIC', multi=1, indextype=1), retmode=1)
    print(Backtest)
    fig, ax = plt.subplots(figsize=(19.2, 10.8))
    sl.cs(Backtest).plot(ax=ax, c='green')
    exPostProjections.plot(ax=ax, c='blue')
    plt.show()

#DataBuilder()
#RunRollPCAOnFXPairs()
#semaOnPCAProjections()
#ARIMAonPCAProjections()

from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
X = pd.read_sql('SELECT * FROM Dlogs', conn).iloc[0:250,:].set_index('Dates', drop=True).T.values

# Locally linear embedding of the digits dataset
print("Computing LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=5, method='standard')
X_lle = clf.fit_transform(X)
print(X_lle[:, 0])
print(X_lle[:, 0].shape)
print(X_lle.shape)
print(clf.get_params())
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)

"""
# ----------------------------------------------------------------------
# Modified Locally linear embedding of the digits dataset
print("Computing modified LLE embedding")
clf = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=2,
                                      method='modified')

X_mlle = clf.fit_transform(X)
print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
#plot_embedding(X_mlle, "Modified Locally Linear Embedding of the digits ")
"""


