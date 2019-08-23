from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')

def DataBuilder():
    names = pd.read_csv('Basket.csv', delimiter=' ')['Names'].tolist()
    P = pd.read_csv('fxEODdata.csv', header=None)
    P.columns = names
    P['Dates'] = pd.to_datetime(P['Dates']-719529, unit='D')
    P = P.set_index('Dates', drop=True)
    P.to_sql('FxData', conn, if_exists='replace')
    pd.read_csv('BasketTS.csv', delimiter=' ').to_sql('BasketTS', conn, if_exists='replace')

def RunRollPcaOnFXPairs():
    df = pd.read_sql('SELECT * FROM FxData', conn).set_index('Dates', drop=True)
    specNames = pd.read_sql('SELECT * FROM BasketTS', conn)['Names'].tolist(); df = df[specNames]
    df = sl.dlog(df).fillna(0); df = df.replace([np.inf, -np.inf], 0)

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

#DataBuilder()
#RunRollPcaOnFXPairs()
semaOnPCAProjections()
