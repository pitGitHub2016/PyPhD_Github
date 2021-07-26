from Slider import Slider as sl
import numpy as np, investpy, json
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

from pydmd import DMD

conn = sqlite3.connect('SmartGlobalAssetAllocation.db')

def test():
    def f1(x, t):
        return 1. / np.cosh(x + 3) * np.exp(2.3j * t)

    def f2(x, t):
        return 2. / np.cosh(x) * np.tanh(x) * np.exp(2.8j * t)

    x = np.linspace(-5, 5, 128)
    t = np.linspace(0, 4 * np.pi, 256)

    xgrid, tgrid = np.meshgrid(x, t)

    X1 = f1(xgrid, tgrid)
    X2 = f2(xgrid, tgrid)
    X = X1 + X2
    titles = ['$f_1(x,t)$', '$f_2(x,t)$', '$f$']
    data = [X1, X2, X]

    fig = plt.figure(figsize=(17, 6))
    for n, title, d in zip(range(131, 134), titles, data):
        plt.subplot(n)
        plt.pcolor(xgrid, tgrid, d.real)
        plt.title(title)
    plt.colorbar()
    plt.show()

    dmd = DMD(svd_rank=2)
    dmd.fit(X.T)

    for eig in dmd.eigs:
        print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag ** 2 + eig.real ** 2 - 1)))

    #dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    for mode in dmd.modes.T:
        plt.plot(x, mode.real)
        plt.title('Modes')
    plt.show()

    for dynamic in dmd.dynamics:
        plt.plot(t, dynamic.real)
        plt.title('Dynamics')
    plt.show()

def financeDataTest():
    pass
    df = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True).iloc[-500:]
    df = df[['Nasdaq', 'DAX', 'USDEUR']]
    X = df.values
    #X = sl.cs(df[['Nasdaq', 'DAX', 'USDEUR']]).values
    print(X)

    dmd = DMD(svd_rank=2)
    dmd.fit(X.T)

    for eig in dmd.eigs:
        print('Eigenvalue {}: distance from unit circle {}'.format(eig, np.abs(eig.imag ** 2 + eig.real ** 2 - 1)))

    #dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    modeList = []
    for mode in dmd.modes.T:
        modeList.append(mode)
        #plt.plot(x, mode.real)
        #plt.title('Modes')
    pd.DataFrame(modeList).plot()
    plt.show()

    for dynamic in dmd.dynamics:
        print(dynamic)
        #plt.plot(t, dynamic.real)
        #plt.title('Dynamics')
    #plt.show()

test()
#financeDataTest()
