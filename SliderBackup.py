import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import math, numpy.linalg as la, sqlite3
from pydiffmap import kernel
import time
from math import acos
from math import sqrt
from math import pi
from numpy.linalg import norm
from numpy import dot, array
from itertools import combinations
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn import linear_model
from hurst import compute_Hc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.callbacks import History
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy import sparse
from scipy.linalg import svd
import warnings, os, tensorflow as tf
from tqdm import tqdm
import math
from sklearn.metrics import mean_squared_error
import pydiffmap

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

class Slider:

    # Operators

    def d(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        return df.diff(nperiods)

    def E(df):
        return df.mean(axis=1)

    def cs(df):
        return df.cumsum()

    def rs(df):
        return df.sum(axis=1)

    def dlog(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        return df.pct_change(nperiods, fill_method='ffill')

    def sign(df):
        df[df > 0] = 1
        df[df < 0] = -1
        df[df == 0] = 0
        return df

    def S(df, **kwargs):

        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1

        out = df.shift(periods=nperiods)

        return out

    def fd(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 0
        if mode == 1:
            df = df.fillna(df.mean())
        elif mode == 0:
            df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

    def svd_flip(u, v, u_based_decision=True):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        v : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v

    ########################

    def cross_product(u, v):
        dim = len(u)
        s = []
        for i in range(dim):
            if i == 0:
                j, k = 1, 2
                s.append(u[j] * v[k] - u[k] * v[j])
            elif i == 1:
                j, k = 2, 0
                s.append(u[j] * v[k] - u[k] * v[j])
            else:
                j, k = 0, 1
                s.append(u[j] * v[k] - u[k] * v[j])
        return s

    def angle_clockwise(A, B):

        def length(v):
            return sqrt(v[0] ** 2 + v[1] ** 2)

        def dot_product(v, w):
            return v[0] * w[0] + v[1] * w[1]

        def determinant(v, w):
            return v[0] * w[1] - v[1] * w[0]

        def inner_angle(v, w):
            cosx = dot_product(v, w) / (length(v) * length(w))
            rad = acos(cosx)  # in radians
            return rad * 180 / pi  # returns degrees

        inner = inner_angle(A, B)
        det = determinant(A, B)
        if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
            return inner
        else:  # if the det > 0 then A is immediately clockwise of B
            return 360 - inner

    def py_ang(v1, v2, method):

        def dotproduct(v1, v2):
            return sum((a * b) for a, b in zip(v1, v2))

        def length(v):
            return math.sqrt(dotproduct(v, v))

        if method == 1:
            return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
        elif method == 2:
            return dotproduct(v1, v2) / (length(v1) * length(v2))
        else:
            arccosInput = dot(v1, v2) / norm(v1) / norm(v2)
            # arccosInput = 1.0 if arccosInput > 1.0 else arccosInput
            # arccosInput = -1.0 if arccosInput < -1.0 else arccosInput
            return math.acos(arccosInput)

    def Hurst(df):
        hurstMat = []
        for col in df.columns:
            H, c, data = compute_Hc(df.loc[:, col], kind='change', simplified=True)
            hurstMat.append(H)
        return hurstMat

    def roller(df, func, n):
        ROLL = df.rolling(window=n, center=False).apply(lambda x: func(x), raw=True)
        return ROLL

    def rollerVol(df, rvn):
        return Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def rollVol(df, rvn):
        return np.sqrt(len(df) / 14) * Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def CorrMatrix(df):
        return df.corr()

    def rollSh(df, window, **kwargs):
        if 'L' in kwargs:
            L = kwargs['L']
        else:
            L = len(df)
        if 'T' in kwargs:
            T = kwargs['T']
        else:
            T = 14
        return np.sqrt(L / T) * Slider.roller(df, Slider.sharpe, window)

    def expander(df, func, n):
        EXPAND = df.expanding(min_periods=n, center=False).apply(lambda x: func(x))
        return EXPAND

    def expSh(df, window, **kwargs):
        if 'L' in kwargs:
            L = kwargs['L']
        else:
            L = len(df)
        if 'T' in kwargs:
            T = kwargs['T']
        else:
            T = 14
        return np.sqrt(L / T) * Slider.expander(df, Slider.sharpe, n=window)

    def sharpe(df):
        return df.mean() / df.std()

    def topSharpe(pnl, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 40
        if 'uniqueTeamSelection' in kwargs:
            uniqueTeamSelection = kwargs['uniqueTeamSelection']
        else:
            uniqueTeamSelection = 0
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 0
        if 'rvunique' in kwargs:
            rvunique = kwargs['rvunique']
        else:
            n = 40

        if mode == 0:
            MSharpe = Slider.sharpe(pnl).sort_values(ascending=False)
            MSharpe = MSharpe[MSharpe > 0]
            posNeg = MSharpe < 0
        elif mode == 1:
            shM = Slider.sharpe(pnl)
            MSharpe = np.absolute(shM).sort_values(ascending=False)
            posNeg = shM < 0
        elif mode == 2:
            MSharpe = Slider.rollSh(pnl, window=100).mean()
            MSharpe = MSharpe[MSharpe > 0]
            posNeg = MSharpe < 0
        elif mode == 3:
            shM = Slider.rollSh(pnl, window=100).mean()
            MSharpe = np.absolute(shM).sort_values(ascending=False)
            posNeg = shM < 0
        TOP = MSharpe.index[:n]
        # Filter the best Teams-Selections combinations with CounterIntuitive Selections = Over vs Under, H vs D vs A etc...
        if uniqueTeamSelection == 1:
            topList = TOP.tolist()
            a = []
            for i in topList:
                itemToListM = i.split('-')
                if len(itemToListM) > 2:
                    # print('-'.join(itemToListM[0:-1]))
                    itemToList = ['-'.join(itemToListM[0:-1]), itemToListM[-1]]
                else:
                    itemToList = itemToListM
                a.append(itemToList)
            aDF = pd.DataFrame(a, columns=['team', 'selection'])
            aDF['TeamGroup'] = aDF['team'] + aDF['selection'].str.replace('FTU25', 'G1').replace('FTO25', 'G1').replace(
                'H', 'G2').replace('D', 'G2').replace('A', 'G2')
            aDF = aDF.set_index('TeamGroup', drop=True)
            aDF = aDF[~aDF.index.duplicated(keep='first')]
            aDF['index'] = aDF['team'] + '-' + aDF['selection']
            aDF = aDF.set_index('index', drop=True)

            # Updated TOP indexes!
            TOP = aDF.index
        elif uniqueTeamSelection == 2:
            topList = TOP.tolist()
            a = []
            for i in topList:
                itemToListM = i.split('_')
                if len(itemToListM) > 2:
                    # print('-'.join(itemToListM[0:-1]))
                    itemToList = ['_'.join(itemToListM[0:-1]), itemToListM[-1]]
                    print(itemToList)
                else:
                    itemToList = itemToListM
                a.append(itemToList)
            aDF = pd.DataFrame(a, columns=['BACK', 'LAY'])
            print(aDF)
            aDF = aDF.groupby(by=['BACK'], as_index=False).head(rvunique)
            # aDF = aDF.drop_duplicates(subset='BACK')
            print(aDF)
            aDF = aDF.groupby(by=['LAY'], as_index=False).head(rvunique)
            # aDF = aDF.drop_duplicates(subset='LAY')
            print(aDF)
            aDF['index'] = aDF['BACK'] + '_' + aDF['LAY']
            print(aDF)
            aDF = aDF.set_index('index', drop=True)

            # Updated TOP indexes!
            TOP = aDF.index
            print(TOP)

        posSwitchTOP = posNeg[TOP]

        return [pnl.loc[:, TOP], TOP, posSwitchTOP]

    def CorrMatrix(df):
        return df.corr()

    def correlation_matrix_plot(df):
        import seaborn as sns
        plt.figure(figsize=(100, 100))
        # play with the figsize until the plot is big enough to plot all the columns
        # of your dataset, or the way you desire it to look like otherwise

        sns.set(font_scale=0.5)
        sns.heatmap(df.corr())
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.show()

    def RV(df):
        cc = list(combinations(df.columns, 2))
        df0 = pd.concat([df[c[0]].sub(df[c[1]]) for c in cc], axis=1, keys=cc)
        df0.columns = df0.columns.map('_'.join)
        return df0

    def sma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        MA = pd.DataFrame(df.rolling(nperiods, min_periods=nperiods).mean()).fillna(0)
        return MA

    def ema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = pd.DataFrame(df.ewm(span=nperiods, min_periods=nperiods).mean()).fillna(0)
        return EMA

    def ExPostOpt(pnl):
        MSharpe = Slider.sharpe(pnl)
        switchFlag = np.array(MSharpe) < 0
        pnl.iloc[:, np.where(switchFlag)[0]] = pnl * (-1)
        return [pnl, switchFlag]

    def adf(df):
        X = df.values
        result = adfuller(X)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        keys = []; values = []
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            keys.append(key); values.append(value)

        return [result, keys, values]

    def gRollingADF(df0, st, **kwargs):

        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        adfData = []
        for i in range(st, len(df0) + 1):
            try:

                print("Step:", i, " of ", len(df0))
                if RollMode == 'RollWindow':
                    df = df0.iloc[i - st:i, :]
                else:
                    df = df0.iloc[0:i, :]

                adfData.append()
            except Exception as e:
                print(e)

    'ARIMA Operators'
    'Optimize order based on AIC or BIC fit'
    def get_optModel(data, opt, **kwargs):
        if 'orderList' in kwargs:
            orderList = kwargs['orderList']
        else:
            orderList = 0
        #ACFAsset = pd.DataFrame(acf(Asset, nlags=len(Asset)), columns=[AssetName])
        best_score, best_cfg = float("inf"), None

        if orderList == 0:
            for p in range(0, 7):
                for d in [0, 1]:
                    for q in range(0, 7):
                        order = (p, d, q)
                        if p!=q:
                            try:
                                model = ARIMA(data, order=(p, d, q))
                                model_fit = model.fit(disp=0)
                                if opt == 'AIC':
                                    aicVal = model_fit.aic
                                elif opt == 'BIC':
                                    aicVal = model_fit.bic
                                if aicVal < best_score:
                                    best_score, best_cfg = aicVal, order
                                #print('ARIMA%s AIC=%.3f' % (order, aicVal))
                            except:
                                continue
        else:
            for order in orderList:
                try:
                    model = ARIMA(data, order=order)
                    model_fit = model.fit(disp=0)
                    if opt == 'AIC':
                        aicVal = model_fit.aic
                    elif opt == 'BIC':
                        aicVal = model_fit.bic

                    if aicVal < best_score:
                        best_score, best_cfg = aicVal, order
                    print('ARIMA%s AIC=%.3f' % (order, aicVal))
                except:
                    continue
        print(best_cfg)
        return best_cfg

    'Arima Dataframe process'
    'Input: list of Dataframe, start: start/window, mode: rolling/expanding, opt: AIC, BIC, (p,d,q)'
    def ARIMA_process(datalist):
        Asset = datalist[0]; start = datalist[1]; mode = datalist[2]; opt = datalist[3]; orderList=datalist[4]
        AssetName = Asset.columns[0]

        X = Asset.to_numpy()
        predictions = []
        stderrList = []
        err = []
        confList = []
        for t in tqdm(range(start, len(X)), desc=AssetName):

            if mode == 'roll':
                history = X[t-start:t]
            elif mode == 'exp':
                history = X[0:t]
            try:
                if opt == 'AIC' or opt == 'BIC':
                    Orders = Slider.get_optModel(history, opt, orderList=orderList)
                else:
                    Orders = opt
                model = ARIMA(history, order=Orders)
                model_fit = model.fit(disp=0)
                #print(model_fit.resid)
                forecast, stderr, conf = model_fit.forecast()
                yhat = forecast
                c1 = conf[0][0]
                c2 = conf[0][1]
            except Exception as e:
                print(e)
                yhat = np.zeros(1) + X[t-1]
                stderr = np.zeros(1)
                c1 = np.nan; c2 = np.nan

            predictions.append(yhat)
            stderrList.append(stderr)
            obs = X[t]
            err.append((yhat-obs)[0])
            confList.append((c1,c2))

        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderrList, index=Asset[start:].index, columns=[AssetName+'_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName+'_err'])
        confDF = pd.DataFrame(confList, index=Asset[start:].index, columns=[AssetName+'_conf1', AssetName+'_conf2'])

        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_Walk(df, start, orderIn):
        X = df.values
        size = int(len(X) * start)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = [0] * len(history)
        for t in tqdm(range(len(test))):
            model = ARIMA(history, order=orderIn)
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        testDF = pd.DataFrame(history, index=df.index)
        PredictionsDF = pd.DataFrame(predictions, index=df.index)

        return [testDF, PredictionsDF]

    def ARIMA_static(datalist):
        Asset = datalist[0]; start = datalist[1]; opt = datalist[3]; orderList = datalist[4]
        AssetName = Asset.columns[0]
        X = Asset.to_numpy()
        history = X[start:-1]
        try:
            if opt == 'AIC' or opt == 'BIC':
                Orders = Slider.get_optModel(history, opt, orderList=orderList)
            else:
                Orders = opt
            model = ARIMA(history, order=Orders)
            model_fit = model.fit(disp=0)
            forecast, stderr, conf = model_fit.forecast()
            yhat = forecast
            c1 = conf[0][0]
            c2 = conf[0][1]
        except Exception as e:
            print(e)
            yhat = np.zeros(1) + history[-1]
            stderr = np.zeros(1)
            c1 = np.nan; c2 = np.nan

        obs = history[-1]
        err = (yhat-obs)
        predictions = yhat[0]
        PredictionsDF = pd.DataFrame(columns=[AssetName], index=Asset.index)
        stderrDF = pd.DataFrame(columns=[AssetName+'_stderr'], index=Asset.index)
        errDF = pd.DataFrame(columns=[AssetName+'_err'], index=Asset.index)
        confDF = pd.DataFrame(columns=[AssetName+'_conf1', AssetName+'_conf2'], index=Asset.index)

        PredictionsDF.loc[Asset.index[-1], AssetName] = predictions
        stderrDF.loc[Asset.index[-1], AssetName+'_stderr'] = stderr[0]
        errDF.loc[Asset.index[-1], AssetName+'_err'] = err[0]
        confDF.loc[Asset.index[-1], AssetName+'_conf1'] = c1; confDF.loc[Asset.index[-1], AssetName+'_conf2'] = c2

        return [PredictionsDF.iloc[-1, :], stderrDF.iloc[-1, :], errDF.iloc[-1, :], confDF.iloc[-1, :]]

    def ARIMA_predictions(df, mode, opt, **kwargs):
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0
        if 'multi' in kwargs:
            multi = kwargs['multi']
        else:
            multi = 0
        if 'indextype' in kwargs:
            indextype = kwargs['indextype']
        else:
            indextype = 0
        if 'orderList' in kwargs:
            orderList = kwargs['orderList']
        else:
            orderList = 0

        if indextype == 1:
            df.index = pd.to_datetime(df.index)
            try:
                frequency = pd.infer_freq(df.index)
                df.index.freq = frequency
                print(frequency)
                df.loc[df.index.max() + pd.to_timedelta(frequency)] = None
            except:
                df.loc[df.index.max() + (df.index[-1] - df.index[-2])] = None

        else:
            df.loc[df.index.max() + 1] = None

        print(df)

        Assets = df.columns.tolist()
        dflist = [[pd.DataFrame(df[Asset]), start, mode, opt, orderList] for Asset in Assets]
        if multi == 0:
            pool = multiprocessing.Pool(processes=len(df.columns.tolist()))
            if mode!='static':
                resultsDF = pool.map(Slider.ARIMA_process, dflist)
                predictions = [x[0] for x in resultsDF]
                stderr = [x[1] for x in resultsDF]
                err = [x[2] for x in resultsDF]
                conf = [x[3] for x in resultsDF]
                PredictionsDF = pd.concat(predictions, axis=1, sort=True)
                stderrDF = pd.concat(stderr, axis=1, sort=True)
                errDF = pd.concat(err, axis=1, sort=True)
                confDF = pd.concat(conf, axis=1, sort=True)
            else:
                resultsDF = pool.map(Slider.ARIMA_static, dflist)
                PredictionsDF = [x[0] for x in resultsDF]
                stderrDF = [x[1] for x in resultsDF]
                errDF = [x[2] for x in resultsDF]
                confDF = [x[3] for x in resultsDF]

        elif multi == 1:
            resultsDF = []
            for i in range(len(Assets)):
                resultsDF.append(Slider.ARIMA_multiprocess(dflist[i]))

            predictions = [x[0] for x in resultsDF]
            stderr = [x[1] for x in resultsDF]
            err = [x[2] for x in resultsDF]
            conf = [x[3] for x in resultsDF]
            PredictionsDF = pd.concat(predictions, axis=1, sort=True)
            stderrDF = pd.concat(stderr, axis=1, sort=True)
            errDF = pd.concat(err, axis=1, sort=True)
            confDF = pd.concat(conf, axis=1, sort=True)

        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_predict(historyOpt):
        history = historyOpt[0]; opt=historyOpt[1]; orderList=historyOpt[2]
        if opt == 'AIC' or opt == 'BIC':
            Orders = Slider.get_optModel(history, opt, orderList=orderList)
        else:
            Orders = opt
        try:
            model = ARIMA(history, order=Orders)
            model_fit = model.fit(disp=0)
            forecast, stderr, conf = model_fit.forecast()
            yhat = forecast
            c1 = conf[0][0]
            c2 = conf[0][1]
        except Exception as e:
            print(e)
            yhat = np.zeros(1) + history[-1]
            stderr = np.zeros(1)
            c1 = np.nan; c2 = np.nan
        obs = history[-1]
        err = (yhat-obs)
        return [yhat, stderr, err, (c1, c2)]

    def ARIMA_multiprocess(df):
        Asset = df[0];
        start = df[1];
        mode = df[2];
        opt = df[3];
        orderList = df[4]
        AssetName = str(Asset.columns[0])
        print('Running Multiprocess Arima: ', AssetName)

        X = Asset.to_numpy()

        if mode == 'roll':
            history = [[X[t - start:t], opt, orderList] for t in range(start, len(X))]
        elif mode == 'exp':
            history = [[X[0:t], opt, orderList] for t in range(start, len(X))]

        p = multiprocessing.Pool(multiprocessing.cpu_count() - 2)
        results = p.map(Slider.ARIMA_predict, tqdm(history, desc=AssetName))
        p.close()
        p.join()

        predictions = [x[0] for x in results]
        stderr = [x[1] for x in results]
        err = [x[2] for x in results]
        conf = [x[3] for x in results]
        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderr, index=Asset[start:].index, columns=[AssetName + '_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName + '_err'])
        confDF = pd.DataFrame(conf, index=Asset[start:].index, columns=[AssetName + '_conf1', AssetName + '_conf2'])
        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_multipredictions(df, mode, opt, **kwargs):
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0
        if 'multi' in kwargs:
            multi = kwargs['multi']
        else:
            multi = 0
        if 'indextype' in kwargs:
            indextype = kwargs['indextype']
        else:
            indextype = 0
        if 'orderList' in kwargs:
            orderList = kwargs['orderList']
        else:
            orderList = 0

        if indextype == 1:
            frequency = pd.infer_freq(df.index)
            df.index.freq = frequency
            print(frequency)
            df.loc[df.index.max() + pd.to_timedelta(frequency)] = None

        Assets = df.columns.tolist()
        dflist = [[pd.DataFrame(df[Asset]), start, mode, opt, orderList] for Asset in Assets]
        resultsDF = []
        for i in range(len(Assets)):
            resultsDF.append(Slider.ARIMA_multiprocess(dflist[i]))

        predictions = [x[0] for x in resultsDF]
        stderr = [x[1] for x in resultsDF]
        err = [x[2] for x in resultsDF]
        conf = [x[3] for x in resultsDF]
        PredictionsDF = pd.concat(predictions, axis=1, sort=True)
        stderrDF = pd.concat(stderr, axis=1, sort=True)
        errDF = pd.concat(err, axis=1, sort=True)
        confDF = pd.concat(conf, axis=1, sort=True)

        return [PredictionsDF, stderrDF, errDF, confDF]

    'Stationarity Operators'

    def Stationarity(df, start, mode, multi):
        Assets = df.columns.tolist()
        dflist = [[pd.DataFrame(df[Asset]), start, mode] for Asset in Assets]

        if multi == 0:
            pool = multiprocessing.Pool(processes=len(df.columns.tolist()))
            resultsDF = pool.map(Slider.Stationarity_process, dflist)

        elif multi == 1:
            resultsDF = []
            for i in range(len(Assets)):
                resultsDF.append(Slider.Stationarity_multiprocess(dflist[i]))

        ADFStat = [x[0] for x in resultsDF]
        pval = [x[1] for x in resultsDF]
        CritVal = [x[2] for x in resultsDF]

        ADFStatDF = pd.concat(ADFStat, axis=1, sort=True)
        pvalDF = pd.concat(pval, axis=1, sort=True)
        CritValDF = pd.concat(CritVal, axis=1, sort=True)

        return [ADFStatDF, pvalDF, CritValDF]

    def Stationarity_test(history):
        #print(history)
        result = adfuller(history)
        ADFStat = result[0]
        pval = result[1]
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        cval = []
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            cval.append(value)
        print(cval)
        CritVal = cval

        return [ADFStat, pval, CritVal]

    def Stationarity_multiprocess(df):
        print('Running Multiprocess Arima')
        print(df)
        Asset = df[0];
        start = df[1];
        mode = df[2]
        AssetName = Asset.columns[0]

        X = Asset[AssetName].to_numpy()
        print(X)

        if mode == 'roll':
            history = [X[t - start:t] for t in range(start, len(X))]
        elif mode == 'exp':
            history = [X[0:t] for t in range(start, len(X))]
        print(history)
        p = multiprocessing.Pool(processes=8)
        results = p.map(Slider.Stationarity_test, history)
        p.close()
        p.join()

        ADFStat = [x[0] for x in results]
        pval = [x[1] for x in results]
        CritVal = [x[2] for x in results]

        ADFStatDF = pd.DataFrame(ADFStat, index=Asset[start:].index, columns=[AssetName + '_ADFstat'])
        pvalDF = pd.DataFrame(pval, index=Asset[start:].index, columns=[AssetName + '_pval'])
        CritValDF = pd.DataFrame(CritVal, index=Asset[start:].index,
                                 columns=[AssetName + '_cv1', AssetName + '_cv5', AssetName + '_cv10'])

        return [ADFStatDF, pvalDF, CritValDF]

    def Stationarity_process(datalist):
        Asset = datalist[0];
        start = datalist[1];
        mode = datalist[2]
        AssetName = Asset.columns[0]
        print(AssetName)

        X = Asset[AssetName].to_numpy()
        pval = []
        ADFStat = []
        CritVal = []
        end = len(X)
        for t in range(start, end):

            if mode == 'roll':
                history = X[t - start:t]
            elif mode == 'exp':
                history = X[0:t]
            try:
                result = adfuller(history)
                ADFStat.append(result[0])
                pval.append(result[1])
                print('ADF Statistic: %f' % result[0])
                print('p-value: %f' % result[1])
                print('Critical Values:')
                cval = []
                for key, value in result[4].items():
                    print('\t%s: %.3f' % (key, value))
                    cval.append(value)
                print(cval)
                CritVal.append(cval)
            except Exception as e:
                print(e)

        ADFStatDF = pd.DataFrame(ADFStat, index=Asset[start:].index, columns=[AssetName + '_ADFstat'])
        pvalDF = pd.DataFrame(pval, index=Asset[start:].index, columns=[AssetName + '_pval'])
        CritValDF = pd.DataFrame(CritVal, index=Asset[start:].index,
                                 columns=[AssetName + '_cv1', AssetName + '_cv5', AssetName + '_cv10'])

        return [ADFStatDF, pvalDF, CritValDF]

    # Machine Learning Functions
    class AI:

        def Pca(df, **kwargs):

            Dates = df.index

            if 'nD' not in kwargs:
                nD = len(df.columns)
            else:
                nD = kwargs['nD']

            features = df.columns.values
            # Separating out the features
            x = df.loc[:, features].values
            # Separating out the target
            # y = df.loc[:, ['target']].values
            # Standardizing the features
            x = MinMaxScaler().fit_transform(x)

            pca = PCA(n_components=nD)
            principalComponents = pca.fit_transform(x)
            principalDf = pd.DataFrame(data=principalComponents)

            principalDf['Date'] = Dates

            principalDf = principalDf.set_index('Date', drop=True)
            principalDf.columns = ['y' + str(i) for i in range(len(principalDf.columns))]

            dfX = pd.DataFrame(x, index=df.index, columns=df.columns)
            dfX.plot()
            plt.show()

            return [dfX, principalDf]

        def gDmaps(df, **kwargs):

            Dates = df.index

            if 'nD' not in kwargs:
                nD = 2
            else:
                nD = kwargs['nD']

            if 'sigma' not in kwargs:
                sigma = 'std'
            else:
                sigma = kwargs['sigma']

            if 'dataMode' not in kwargs:
                dataMode = 'normalize'
            else:
                dataMode = kwargs['sigma']

            if 'gammaCO' not in kwargs:
                gammaCO = 0.5
            else:
                gammaCO = kwargs['gammaCO']

            if 'coFlag' not in kwargs:
                coFlag = 0
            else:
                coFlag = kwargs['coFlag']

            if dataMode == 'standard':
                df = pd.DataFrame(MinMaxScaler().fit_transform(df.values))

            Ddists = squareform(pdist(df))

            if sigma == 'std':
                sigmaDMAPS = df.std()
            elif sigma == 'MaxMin':
                sigmaDMAPS = df.max() - df.min()
            elif sigma == 'bgh':
                sigmaDMAPS = kernel.choose_optimal_epsilon_BGH(Ddists)[0]
            else:
                sigmaDMAPS = sigma

            # K = np.exp(-pd.DataFrame(Ddists) / (2 * sigmaDMAPS ** 2))
            K = np.exp(-pd.DataFrame(Ddists) / (sigmaDMAPS))
            a1 = np.sqrt(K.sum())
            A = K / (a1.dot(a1))
            threshold = 5E-60
            sparseK = pd.DataFrame(sparse.csr_matrix((A * (A > threshold).astype(int)).as_matrix()).todense())
            U, s, VT = svd(sparseK)
            U, VT = Slider.svd_flip(U, VT)

            U = pd.DataFrame(U)
            # s = pd.DataFrame(s)
            VT = pd.DataFrame(VT)
            psi = U
            phi = VT
            for col in U.columns:
                'Building psi and phi projections'
                # psi[col] = s * (U[col] / U.iloc[:, 0])
                psi[col] = U[col] / U.iloc[:, 0]
                phi[col] = VT[col] * VT.iloc[:, 0]

            eigOut = psi.iloc[:, 1:nD + 1].fillna(0)
            eigOut.columns = [str(x) for x in range(nD)]
            eigOut.index = df.index

            # Slider.cs(eigOut).plot()
            # plt.show()

            'Building Contractive Observer Data'
            if coFlag == 1:
                aMat = []
                for z in df.columns:
                    aSubMat = []
                    for eig in eigOut:
                        ajl = df[z] * eigOut[eig]
                        aSubMat.append(ajl.sum())
                    aMat.append(aSubMat)

                aMatDF = pd.DataFrame(aMat)
                a_inv = pd.DataFrame(np.linalg.pinv(aMatDF.values), aMatDF.columns, aMatDF.index).T
                s = (-1) * s  # impose negative eigenvalues to guarantee contraction
                lMat = pd.DataFrame(np.diag(s[:nD]))
                glA = gammaCO * pd.DataFrame(np.dot(a_inv, lMat))
                glA.index = df.columns

                eq0List = []
                coutCo = 0
                for idx, row in eigOut.iterrows():
                    if coutCo == 0:
                        eq0List.append(eigOut.iloc[0, :].to_numpy())
                    else:
                        eq0List.append(eq0List[-1] + (1 - gammaCO) * np.dot(eigOut.loc[idx, :], lMat))
                    coutCo += 1
                eq0 = pd.DataFrame(eq0List, index=eigOut.index, columns=eigOut.columns)

                eq1 = pd.DataFrame(np.dot(df, glA), index=eigOut.index, columns=eigOut.columns)

                cObserver = (eq0 + eq1).fillna(0)

                # fig = plt.figure()
                # cObserver.iloc[:,0].plot(ax=fig.add_subplot(221), title='1st')
                # cObserver.iloc[:,1].plot(ax=fig.add_subplot(222), title='2nd')
                # cObserver.iloc[:,2].plot(ax=fig.add_subplot(223), title='3rd')
                # cObserver.iloc[:,3].plot(ax=fig.add_subplot(224), title='4th')
                # plt.show()

            else:
                lMat = pd.DataFrame([])
                dfPsiT1 = pd.DataFrame([])

            return [eigOut, sigmaDMAPS, s[:nD], glA, cObserver]

        def pyDmapsRun(df):
            X = df.values
            print(X)

            mydmap = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=5, epsilon='bgh')
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

        def gRollingManifold(manifoldIn, df0, st, NumProjections, eigsPC, **kwargs):
            if 'RollMode' in kwargs:
                RollMode = kwargs['RollMode']
            else:
                RollMode = 'RollWindow'

            if 'Scaler' in kwargs:
                Scaler = kwargs['Scaler']
            else:
                Scaler = 'Standard'

            if 'ProjectionMode' in kwargs:
                ProjectionMode = kwargs['ProjectionMode']
            else:
                ProjectionMode = 'NoTranspose'

            if 'contractiveObserver' in kwargs:
                contractiveObserver = kwargs['contractiveObserver']
            else:
                contractiveObserver = 0

            if 'LLE_n_neighbors' in kwargs:
                n_neighbors = kwargs['LLE_n_neighbors']
            else:
                n_neighbors = 2

            if 'LLE_Method' in kwargs:
                LLE_Method = kwargs['LLE_n_neighbors']
            else:
                LLE_Method = 'standard'

            if 'DMAPS_sigma' in kwargs:
                sigma = kwargs['DMAPS_sigma']
            else:
                sigma = 'std'

            eigDf = []
            eigCoeffs = [[] for j in range(len(eigsPC))]
            Comps = [[] for j in range(len(eigsPC))]
            sigmaList = []
            lambdasList = []
            sigmaDF = pd.DataFrame([])
            lambdasDF = pd.DataFrame([])
            cObserverList = []
            # st = 50; pcaN = 5; eigsPC = [0];
            for i in range(st, len(df0) + 1):
                # try:

                print("Step:", i, " of ", len(df0) + 1)
                if RollMode == 'RollWindow':
                    df = df0.iloc[i - st:i, :]
                else:
                    df = df0.iloc[0:i, :]

                if ProjectionMode == 'Transpose':
                    df = df.T

                features = df.columns.values
                x = df.loc[:, features].values

                if Scaler == 'Standard':
                    x = StandardScaler().fit_transform(x)

                if manifoldIn == 'PCA':
                    pca = PCA(n_components=NumProjections)
                    X_pca = pca.fit_transform(x)
                    lambdasList.append(list(pca.singular_values_))
                    sigmaList.append(list(pca.explained_variance_ratio_))
                    c = 0
                    for eig in eigsPC:
                        # print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                        Comps[eig].append(list(pca.components_[eig]))
                        c += 1

                elif manifoldIn == 'ICA':
                    X, _ = load_digits(return_X_y=True)
                    ica = FastICA(n_components=NumProjections,  random_state=0)
                    X_ica = ica.fit_transform(X)
                    lambdasList.append(1)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        print(list(ica.components_[eig]))
                        Comps[eig].append(list(ica.components_[eig]))
                        c += 1

                elif manifoldIn == 'DMAP':
                    dMapsOut = Slider.AI.gDmaps(df, nD=NumProjections, coFlag=contractiveObserver,
                                                sigma=sigma)  # [eigOut, sigmaDMAPS, s[:nD], glA]
                    eigDf.append(dMapsOut[0].iloc[-1, :])
                    glAout = dMapsOut[3]
                    cObserverList.append(dMapsOut[4].iloc[-1, :])
                    sigmaList.append(dMapsOut[1])
                    lambdasList.append(dMapsOut[2])
                    for gl in glAout:
                        Comps[gl].append(glAout[gl])
                        eigCoeffs[gl].append(
                            linear_model.LinearRegression(normalize=True).fit(df, dMapsOut[0].iloc[:, gl]).coef_)

                elif manifoldIn == 'LLE':
                    lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections,
                                                          method=LLE_Method, n_jobs=-1)
                    X_lle = lle.fit_transform(x)  # ; print(X_lle.shape)
                    lambdasList.append(1)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        # print(eig, ',', len(X_lle[:, eig])) # 0 , 100 , 5
                        Comps[c].append(list(X_lle[:, eig]))
                        c += 1

                # except Exception as e:
                #    print(e)
                #    for c in len(eigsPC):
                #        Comps[c].append(list(np.zeros(len(df0.columns), 1)))
                #        eigCoeffs[c].append(list(np.zeros(len(df0.columns), 1)))

            sigmaDF = pd.concat([pd.DataFrame(np.zeros((st - 1, 1))), pd.DataFrame(sigmaList)], axis=0,
                                ignore_index=True).fillna(0)
            sigmaDF.index = df0.index
            try:
                sigmaDF.columns = ['sigma']
            except Exception as e:
                print(e)

            lambdasDF = pd.concat(
                [pd.DataFrame(np.zeros((st - 1, pd.DataFrame(lambdasList).shape[1]))), pd.DataFrame(lambdasList)],
                axis=0, ignore_index=True).fillna(0)
            lambdasDF.index = df0.index

            if contractiveObserver == 0:
                principalCompsDf = [[] for j in range(len(Comps))]
                exPostProjections = [[] for j in range(len(Comps))]
                for k in range(len(Comps)):
                    # principalCompsDf[k] = pd.DataFrame(pcaComps[k], columns=df0.columns, index=df1.index)

                    principalCompsDf[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Comps[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf[k].index = df0.index
                    principalCompsDf[k] = principalCompsDf[k].fillna(0).replace(0, np.nan).ffill()

                    exPostProjections[k] = df0 * Slider.S(principalCompsDf[k])

                return [df0, principalCompsDf, exPostProjections, sigmaDF, lambdasDF]

            else:

                principalCompsDf = [[] for j in range(len(Comps))]
                principaleigCoeffsDf = [[] for j in range(len(Comps))]
                exPostProjectionsComp = [[] for j in range(len(Comps))]
                exPostProjectionseigCoeffs = [[] for j in range(len(Comps))]
                for k in range(len(Comps)):
                    # principalCompsDf[k] = pd.DataFrame(pcaComps[k], columns=df0.columns, index=df1.index)

                    principalCompsDf[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Comps[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf[k].index = df0.index
                    principalCompsDf[k] = principalCompsDf[k].fillna(0).replace(0, np.nan).ffill()

                    principaleigCoeffsDf[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(eigCoeffs[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principaleigCoeffsDf[k].index = df0.index
                    principaleigCoeffsDf[k] = principaleigCoeffsDf[k].fillna(0).replace(0, np.nan).ffill()

                    exPostProjectionsComp[k] = df0 * Slider.S(principalCompsDf[k])
                    exPostProjectionseigCoeffs[k] = df0 * Slider.S(principaleigCoeffsDf[k])

                return [df0, principalCompsDf, principaleigCoeffsDf, exPostProjectionsComp, exPostProjectionseigCoeffs,
                        sigmaDF, lambdasDF]

        def gANN(X_train, X_test, y_train, params):
            epochsIn = params[0]
            batchSIzeIn = params[1]

            history = History()
            # Feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # Part 2 - Now let's make the ANN!
            # Importing the Keras libraries and packages

            # Initialising the ANN
            classifier = Sequential()

            # Adding the input layer and the first hidden layer
            classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu',
                                 input_dim=X_train.shape[1]))
            # Adding a Dropout
            classifier.add(Dropout(0.25))

            # Adding the second hidden layer
            classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
            # Adding a Dropout
            classifier.add(Dropout(0.25))

            # Adding the third hidden layer
            classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
            # Adding a Dropout
            classifier.add(Dropout(0.25))

            # Adding the output layer
            try:
                shapeIn = y_train.shape[1]
            except Exception as e:
                print(e)
                shapeIn = 1

            classifier.add(Dense(units=shapeIn, kernel_initializer='uniform', activation='sigmoid'))

            # Compiling the ANN
            classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

            # Fitting the ANN to the Training set
            classifier.fit(X_train, y_train, batch_size=batchSIzeIn, epochs=epochsIn, callbacks=[history])

            # Part 3 - Making predictions and evaluating the model

            # Predicting the Test set results
            y_pred = classifier.predict(X_test)

            return [y_pred, classifier]

        def gRNN(dataset_all, params):

            ################################### Data Preprocessing ###################################################
            history = History()

            SingleSample = False
            if isinstance(dataset_all, pd.Series):
                SingleSample = True

            TrainEnd = int(params["TrainEndPct"] * len(dataset_all))
            dataVals = dataset_all.values

            #################################### Feature Scaling #####################################################
            sc = MinMaxScaler(feature_range=(0, 1))
            if SingleSample:
                dataVals = sc.fit_transform(dataVals.reshape(-1, 1))
                FeatSpaceDims = 1
                outNaming = [dataset_all.name]
                print(outNaming)
            else:
                dataVals = sc.fit_transform(dataVals)
                FeatSpaceDims = len(dataset_all.columns)
                outNaming = dataset_all.columns

            #################### Creating a data structure with N timesteps and 1 output #############################
            X = []
            y = []
            for i in range(params["TrainWindow"], len(dataset_all)):
                X.append(dataVals[i - params["TrainWindow"]:i - params["HistLag"]])
                y.append(dataVals[i])
            X, y = np.array(X), np.array(y)
            idx = dataset_all.iloc[params["TrainWindow"]:].index

            ####################################### Reshaping ########################################################
            "Samples : One sequence is one sample. A batch is comprised of one or more samples."
            "Time Steps : One time step is one point of observation in the sample."
            "Features : One feature is one observation at a time step."
            X = np.reshape(X, (X.shape[0], X.shape[1], FeatSpaceDims))

            X_train, y_train = X[:TrainEnd], y[:TrainEnd]
            X_test, y_test = X[TrainEnd:], y[TrainEnd:]

            df_real_price_train = pd.DataFrame(sc.inverse_transform(y_train), index=idx[:TrainEnd],
                                               columns=outNaming)
            df_real_price_test = pd.DataFrame(sc.inverse_transform(y_test), index=idx[TrainEnd:],
                                              columns=outNaming)

            print("len(idx)=", len(idx), ", idx.tail[:10]=", idx[:10], ", X.shape=", X.shape, ", TrainWindow=",
                  params["TrainWindow"],
                  ", TrainEnd=", TrainEnd, ", X_train.shape=", X_train.shape, ", y_train.shape=", y_train.shape,
                  ", X_test.shape=", X_test.shape, ", y_test.shape=", y_test.shape)

            ####################################### Initialising the LSTM #############################################
            regressor = Sequential()
            # Adding the first LSTM layer and some Dropout regularisation
            for layer in range(len(params["medSpecs"])):
                if params["medSpecs"][layer]["units"] == 'xShape1':
                    unitsIn = X_train.shape[1]
                    if params["medSpecs"][layer]["LayerType"] == "LSTM":
                        regressor.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                           unit_forget_bias=True, bias_initializer='ones',
                                           input_shape=(X_train.shape[1], FeatSpaceDims)))
                    elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                        regressor.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                                unit_forget_bias=True, bias_initializer='ones',
                                                input_shape=(X_train.shape[1], FeatSpaceDims)))
                    regressor.add(Dropout(params["medSpecs"][layer]["Dropout"]))
                else:
                    unitsIn = params["medSpecs"][layer]["units"]
                    if params["medSpecs"][layer]["LayerType"] == "LSTM":
                        regressor.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                    elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                        regressor.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                    regressor.add(Dropout(params["medSpecs"][layer]["Dropout"]))
            # Adding the output layer
            regressor.add(Dense(units=y_train.shape[1]))

            ######################################## Compiling the RNN ###############################################
            regressor.compile(optimizer=params["CompilerSettings"][0], loss=params["CompilerSettings"][1])
            # Fitting the RNN to the Training set
            regressor.fit(X_train, y_train, epochs=params["epochsIn"], batch_size=params["batchSIzeIn"],
                          callbacks=[history])

            ########################## Get Predictions for Static or Online Learning #################################
            predicted_price_train = sc.inverse_transform(regressor.predict(X_train))

            scoreList = []
            if params["LearningMode"] == 'static':
                predicted_price_test = sc.inverse_transform(regressor.predict(X_test))
                scoreDF = pd.DataFrame(history.history['loss'], columns=['loss'])
                scoreDF.plot()
                plt.show()

            elif params["LearningMode"] == 'static_MultiStep_Ahead':

                predicted_price_test = []  #
                print(1, ", X_test[0]=", X_test[0],
                      ", reShaped_X_test[0]=", np.reshape(X_test[0][0], (1, 1, 1)),
                      ", UnConvertedPrediction=",
                      regressor.predict(np.reshape(X_test[0], (1, X_test.shape[1], FeatSpaceDims))))

                predicted_price_test.append(
                    regressor.predict(np.reshape(X_test[0], (1, 1, 1))))
                print("Predicting ", len(df_real_price_test) - 1, " steps ahead...")
                for obs in range(params["static_MultiStep_Ahead_Horizon"]):
                    predicted_price_test.append(
                        regressor.predict(np.reshape(predicted_price_test[-1], (1, 1, 1)))[0])
                print(predicted_price_test)
                predicted_price_test = sc.inverse_transform(predicted_price_test)
                print("predicted_price_test : ", predicted_price_test)
                scoreDF = pd.DataFrame(history.history['loss'], columns=['loss'])
                scoreDF.plot()
                plt.show()

            elif params["LearningMode"] == 'online':

                # X_test, y_test
                predicted_price_test = []
                for i in range(len(X_test)):
                    # X_test[i], y_test[i] = np.array(X_test[i]), np.array(y_test[i])

                    print('Calculating: ' + str(round(i / len(X_test) * 100, 2)) + '%')
                    print("X_test.shape=", X_test.shape, ", y_test.shape=", y_test.shape)
                    print("X_test[i].shape=", X_test[i].shape, ", y_test[i].shape=", y_test[i].shape)
                    print("X_test[i]=", X_test[i])
                    print("y_test[i]=", y_test[i])

                    print("(1, X_test[i].shape[0], len(dataset_all.columns)) = ",
                          (1, X_test[i].shape[0], FeatSpaceDims))
                    indXtest = np.reshape(X_test[i], (1, X_test[i].shape[0], FeatSpaceDims))
                    # print("(X_test[i].shape[0], 1, len(dataset_all.columns)) = ", (X_test[i].shape[0], 1, len(dataset_all.columns)))
                    # indXtest = np.reshape(X_test[i], (X_test[i].shape[0], 1, len(dataset_all.columns)))

                    print("indXest.shape=", indXtest.shape)
                    print("predicted_price_test=", sc.inverse_transform(regressor.predict(indXtest))[0])
                    predicted_price_test.append(sc.inverse_transform(regressor.predict(indXtest))[0])

                    indYtest = np.reshape(y_test[i], (1, FeatSpaceDims))

                    try:
                        sc.partial_fit(dataVals[i + TrainEnd])
                    except Exception as e:
                        print(e)
                    regressor.train_on_batch(indXtest, indYtest)
                    scores = regressor.evaluate(indXtest, indYtest, verbose=0)
                    scoreList.append(scores)
                    print(scores)
                scoreDF = pd.DataFrame(scoreList)

            df_predicted_price_train = pd.DataFrame(predicted_price_train, index=df_real_price_train.index,
                                                    columns=['PredictedPrice_Train_' + str(c) for c in
                                                             df_real_price_train.columns])
            df_predicted_price_test = pd.DataFrame(predicted_price_test,
                                                   columns=['PredictedPrice_Test_' + str(c) for c in
                                                            df_real_price_test.columns])

            if len(df_predicted_price_test) <= len(df_real_price_test):
                df_predicted_price_test.index = df_real_price_test.index

            if 'writeLearnStructure' in params:
                xList = []
                for bX in X:
                    xList.append(pd.DataFrame(bX).T.astype(str).agg('-'.join, axis=1).values)
                dataDF = pd.concat(
                    [dataset_all, pd.DataFrame(dataVals, index=dataset_all.index), pd.DataFrame(xList, index=idx),
                     pd.DataFrame(y, index=idx),
                     df_real_price_train, df_real_price_test, df_predicted_price_train, df_predicted_price_test],
                    axis=1)
                dataDF.to_csv('LearnStructure.csv')

            return [df_real_price_test, df_predicted_price_test, scoreDF, regressor, history]

    class Models:

        def __init__(self, df, **kwargs):
            'The initial data input in the Models Class '
            self.df = df

            'The meta-data on which the model calculates the betting signal matrix : d, dlog, raw data'
            if 'retmode' in kwargs:
                self.retmode = kwargs['retmode']
            else:
                self.retmode = 2

            if self.retmode == 0:
                self.ToPnL = Slider.d(self.df)
            elif self.retmode == 1:
                self.ToPnL = Slider.dlog(self.df)
            else:
                self.ToPnL = self.df

            self.sig = float('nan')

        def ARIMA_signal(self, **kwargs):
            if 'start' in kwargs:
                start = kwargs['start']
            else:
                start = 0
            if 'mode' in kwargs:
                mode = kwargs['mode']
            else:
                mode = 'exp'
            if 'opt' in kwargs:
                opt = kwargs['opt']
            else:
                opt = (1,0,0)
            if 'multi' in kwargs:
                multi = kwargs['multi']
            else:
                multi = 0
            if 'indextype' in kwargs:
                indextype = kwargs['indextype']
            else:
                indextype = 0
            print(multi)
            self.sig = Slider.sign(Slider.ARIMA_predictions(self.ToPnL, start=start, mode=mode, opt=opt, multi=multi, indextype=indextype)[0] - Slider.S(self.ToPnL))
            self.ToPnL = self.ToPnL
            return self

    class BacktestPnL:

        def ModelPnL(model, **kwargs):

            if 'retmode' in kwargs:
                retmode = kwargs['retmode']
            else:
                retmode = 0

            if retmode == 1:
                #model.ToPnL = Slider.d(model.ToPnL)
                return model.sig * model.ToPnL.diff()
            else:
                return model.sig * model.ToPnL

