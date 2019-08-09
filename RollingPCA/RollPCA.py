from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
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
    specNames = pd.read_sql('SELECT * FROM BasketTS', conn)['Names'].tolist()
    df = df[specNames]
    #df = df.iloc[:100]

    out = sl.AI.gRollingPca(df, 50, 5, [0,1,2,3,4])
    out[0].to_sql('df1', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    k = 0
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql('principalCompsDf_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql('exPostProjections_'+str(k), conn, if_exists='replace')

def semaOnPCAProjections():
    exPostProjections_0 = pd.read_sql('SELECT * FROM exPostProjections_0', conn).set_index('Dates', drop=True)
    print(exPostProjections_0)
    pnl = sl.S(sl.sema(exPostProjections_0, nperiods=3)) * sl.d(exPostProjections_0)
    print(sl.sharpe(pnl))

    #print(pnl)

#DataBuilder()
#RunRollPcaOnFXPairs()
semaOnPCAProjections()