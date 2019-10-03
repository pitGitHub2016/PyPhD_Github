from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
import pydiffmap

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')

pd.read_csv('BasketTS.csv', delimiter=' ').to_sql('BasketTS', conn, if_exists='replace')
pd.read_csv('BasketGekko.csv', delimiter=' ').to_sql('BasketGekko', conn, if_exists='replace')

def DataSelect(set, mode):
    df = pd.read_sql('SELECT * FROM FxData', conn).set_index('Dates', drop=True)
    specNames = pd.read_sql('SELECT * FROM ' + set, conn)['Names'].tolist()
    df = df[specNames]
    df = df.replace([np.inf, -np.inf, 0], np.nan).ffill()
    if mode == 'rets':
        df = sl.dlog(df).fillna(0)

    return df

def plotData(assetSel, set, mode):
    if assetSel == 'raw':
        df = DataSelect(set, 'rets')

    elif assetSel == 'projections':
        allProjections = []
        for pr in range(5):
            exPostProjections = pd.read_sql('SELECT * FROM ' + set + '_exPostProjections_' + str(pr), conn).set_index(
                'Dates', drop=True)
            allProjections.append(sl.rs(exPostProjections))
        df = pd.concat(allProjections, axis=1, ignore_index=True)
        df.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    fig, ax = plt.subplots()
    if mode == 'raw':
        csDf = sl.cs(df)
        csDf.plot(ax=ax, title='Assets')

        from mpl_toolkits.mplot3d import Axes3D
        cols = ['P0', 'P1', 'P2']
        threedee = plt.figure().gca(projection='3d')
        threedee.scatter(csDf[cols[0]], csDf[cols[1]], csDf[cols[2]])
        plt.title('3D Plot of ' + set + ' Projections')
        threedee.set_xlabel(cols[0]);
        threedee.set_ylabel(cols[1]);
        threedee.set_zlabel(cols[2])
        plt.show()

    elif mode == 'rv':
        sl.cs(sl.RV(df)).plot(ax=ax, title=set + ' Relative Values')
    plt.show()

def pyDmapsRun():
    X = df.values

    mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=5, alpha=0.8, epsilon='bgh', k=15)
    # mydmap.fit(X)

    dmap = mydmap.fit_transform(X)
    # dmap_Y = mydmap.transform(Y)

    dmap_DF = pd.DataFrame(dmap, columns=['P0', 'P1', 'P2', 'P3', 'P4'])
    # dmap_DF = sl.cs(dmap_DF)
    # print(dmap_DF)

    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(dmap_DF['P0'], dmap_DF['P1'], dmap_DF['P2'])
    plt.title('3D Plot of DMAPS Projections')
    threedee.set_xlabel('P0');
    threedee.set_ylabel('P1');
    threedee.set_zlabel('P2')
    # dmap_DF.plot()
    plt.show()

df = DataSelect('BasketGekko', 'rets')
df = df.iloc[1500:2000, :]
# df = sl.cs(df.iloc[1000:2000, :])

#eigOut = sl.AI.gDmaps(df.T, nD=5); print(eigOut.iloc[:,0].values)
out = sl.AI.gRollingManifold('DMAPS', df, 50, 5, [0, 1, 2, 3, 4], ProjectionMode='Transpose')
out[2].plot()
plt.show()

