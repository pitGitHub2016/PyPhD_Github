import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import math, numpy.linalg as la, sqlite3
from math import acos
from math import sqrt
from math import pi
from numpy.linalg import norm
from numpy import dot
from itertools import combinations
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from hurst import compute_Hc
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import History
import warnings, os, tensorflow as tf
from tqdm import tqdm
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
        return pd.DataFrame.shift(df, nperiods)

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

    def sema(df, **kwargs):
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
        for t in range(start, len(X)):

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
        print('Running Multiprocess Arima')

        Asset = df[0]; start = df[1]; mode = df[2]; opt=df[3]; orderList=df[4]
        AssetName = Asset.columns[0]

        X = Asset.to_numpy()

        if mode == 'roll':
            history = [[X[t-start:t], opt, orderList] for t in range(start, len(X))]
        elif mode == 'exp':
            history = [[X[0:t], opt, orderList] for t in range(start, len(X))]

        p = multiprocessing.Pool(processes=12)
        results = p.map(Slider.ARIMA_predict, history)
        p.close()
        p.join()

        predictions = [x[0] for x in results]
        stderr = [x[1] for x in results]
        err = [x[2] for x in results]
        conf = [x[3] for x in results]
        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderr, index=Asset[start:].index, columns=[AssetName+'_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName+'_err'])
        confDF = pd.DataFrame(conf, index=Asset[start:].index, columns=[AssetName+'_conf1', AssetName+'_conf2'])
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
                nD = 2
            else:
                nD = kwargs['nD']

            features = df.columns.values
            # Separating out the features
            x = df.loc[:, features].values
            # Separating out the target
            # y = df.loc[:, ['target']].values
            # Standardizing the features
            x = StandardScaler().fit_transform(x)

            pca = PCA(n_components=nD)
            principalComponents = pca.fit_transform(x)
            principalDf = pd.DataFrame(data=principalComponents)

            principalDf['Date'] = Dates

            principalDf = principalDf.set_index('Date', drop=True)

            return principalDf

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

            if 'LLE_n_neighbors' in kwargs:
                n_neighbors = kwargs['LLE_n_neighbors']
            else:
                n_neighbors = 2

            if 'LLE_Method' in kwargs:
                LLE_Method = kwargs['LLE_n_neighbors']
            else:
                LLE_Method = 'standard'

            Comps = [[] for j in range(len(eigsPC))]
            # st = 50; pcaN = 5; eigsPC = [0];
            for i in range(st, len(df0)+1):
                try:

                    print("Step:", i, " of ", len(df0))
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
                        # if i == st:
                        #    print(pca.explained_variance_ratio_)
                        c = 0
                        for eig in eigsPC:
                            #print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                            Comps[eig].append(list(pca.components_[eig]))
                            c += 1
                    elif manifoldIn == 'LLE':
                        lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections, method=LLE_Method, n_jobs=-1)
                        X_lle = lle.fit_transform(x)  # ; print(X_lle.shape)
                        c = 0
                        for eig in eigsPC:
                            # print(eig, ',', len(X_lle[:, eig])) # 0 , 100 , 5
                            Comps[c].append(list(X_lle[:, eig]))
                            c +=1
                except Exception as e:
                    print(e)
                    for c in len(eigsPC):
                        Comps[c].append(list(np.zeros(len(df0.columns), 1)))

            principalCompsDf = [[] for j in range(len(Comps))]
            exPostProjections = [[] for j in range(len(Comps))]
            for k in range(len(Comps)):
                #principalCompsDf[k] = pd.DataFrame(pcaComps[k], columns=df0.columns, index=df1.index)

                principalCompsDf[k] = pd.concat([pd.DataFrame(np.zeros((st-1, len(df0.columns))), columns=df0.columns), pd.DataFrame(Comps[k], columns=df0.columns)], axis=0, ignore_index=True)
                principalCompsDf[k].index = df0.index
                principalCompsDf[k] = principalCompsDf[k].fillna(0).replace(0, np.nan).ffill()

                exPostProjections[k] = df0 * Slider.S(
                    principalCompsDf[k])  #exPostProjections[k] = df0.mul(Slider.S(principalCompsDf[k]), axis=0)

            return [df0, principalCompsDf, exPostProjections]

        def gRNN(dataset_all, params):

            #HistLag = 0; TrainWindow = 50; epochsIn = 50; batchSIzeIn = 49; medBatchTrain = 49; HistoryPlot = 0; PredictionsPlot = 0
            HistLag = params[0];
            TrainWindow = params[1];
            epochsIn = params[2];
            batchSIzeIn = params[3];
            medBatchTrain = params[4];
            HistoryPlot = params[5];
            PredictionsPlot = params[6];
            LearningMode = params[7]

            history = History()

            TrainEnd = int(0.3 * len(dataset_all))
            #InputArchitecture = pd.DataFrame([['LSTM', 0.2], ['LSTM', 0.2], ['LSTM', 0.2], ['Dense', 0.2]], columns=['LayerType', 'LayerDropout'])
            #OutputArchitecture = pd.DataFrame([['Dense', 0.2]], columns=['LayerType', 'LayerDropout'])
            #CompilerParams = pd.DataFrame([['adam', 'mean_squared_error']], columns=['optimizer', 'loss'])

            training_set = dataset_all.iloc[:TrainEnd].values

            # Feature Scaling
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            # training_set_scaled = training_set

            # Creating a data structure with N timesteps and 1 output
            X_train = [];
            y_train = []
            for i in range(TrainWindow, TrainEnd):
                X_train.append(training_set_scaled[i - TrainWindow:i - HistLag])
                y_train.append(training_set_scaled[i])
            X_train, y_train = np.array(X_train), np.array(y_train)

            # Reshaping
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(dataset_all.columns)))

            # Part 2 - Building the RNN
            # Importing the Keras libraries and packages

            # Initialising the RNN
            regressor = Sequential()
            # Adding the first LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=X_train.shape[1], return_sequences=True,
                               input_shape=(X_train.shape[1], len(dataset_all.columns))))
            regressor.add(Dropout(0.2))
            # Adding a second LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=X_train.shape[1], return_sequences=True))
            regressor.add(Dropout(0.2))
            # Adding a third LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=X_train.shape[1], return_sequences=True))
            regressor.add(Dropout(0.2))
            # Adding a fourth LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=X_train.shape[1]))
            # regressor.add(Dropout(0.2))
            # Adding the output layer
            regressor.add(Dense(units=y_train.shape[1]))

            # Compiling the RNN
            regressor.compile(optimizer='adam', loss='mean_squared_error')
            # regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
            # regressor.compile(optimizer='adam', loss='categorical_crossentropy')

            # Fitting the RNN to the Training set
            regressor.fit(X_train, y_train, epochs=epochsIn, batch_size=batchSIzeIn, callbacks=[history])

            print(history.history.keys())

            # Part 3 - Making the predictions and visualising the results
            # Getting the test set
            dataset_test = dataset_all.iloc[TrainEnd:].values
            inputs = sc.transform(dataset_test.reshape(-1, len(dataset_all.columns)))
            # inputs = dataset_test.reshape(-1, len(dataset_all.columns))

            # Getting the predicted prices
            dataset_total = pd.concat([pd.DataFrame(training_set_scaled), pd.DataFrame(inputs)], axis=0).reset_index(
                drop=True).values

            scoreList = []
            if LearningMode == 'static':
                X_test = []
                for i in range(TrainEnd, len(dataset_total)):  # TrainEnd+100):
                    print('Calculating: ' + str(round(i / len(dataset_total) * 100, 2)) + '%')
                    X_test.append(dataset_total[i - TrainWindow:i - HistLag])
                X_test_array = np.array(X_test)
                X_test_array = np.reshape(X_test_array,
                                          (X_test_array.shape[0], X_test_array.shape[1], len(dataset_all.columns)))

                predicted_price = sc.inverse_transform(regressor.predict(X_test_array))

            elif LearningMode == 'online':
                X_test = [];
                y_test = [];
                for i in range(TrainEnd, len(dataset_total)):  # TrainEnd+100):
                    print('Calculating: ' + str(round(i / len(dataset_total) * 100, 2)) + '%')
                    # Formalize the rolling input prices
                    X_test.append(dataset_total[i - TrainWindow:i - HistLag])
                    X_test_array = np.array(X_test)
                    X_test_array = np.reshape(X_test_array,
                                              (X_test_array.shape[0], X_test_array.shape[1], len(dataset_all.columns)))

                    # Formalize the rolling output prices
                    y_test.append(dataset_total[i])
                    y_test_array = np.array(y_test)
                    # Get the out-of-sample predicted prices
                    predicted_price = sc.inverse_transform(regressor.predict(X_test_array))

                    if len(X_test) > medBatchTrain:
                        X_test_Batch = X_test[-medBatchTrain:]
                        X_test_array_Batch = np.array(X_test_Batch)
                        X_test_array_Batch = np.reshape(X_test_array_Batch, (
                        X_test_array_Batch.shape[0], X_test_array_Batch.shape[1], len(dataset_all.columns)))
                        y_test_Batch = y_test[-medBatchTrain:]
                        y_test_array_Batch = np.array(y_test_Batch)
                    try:
                        # Retrain the RNN on batch as new info comes into play
                        regressor.train_on_batch(X_test_array_Batch, y_test_array_Batch)
                        # Final evaluation of the model
                        scores = regressor.evaluate(X_test_array_Batch, y_test_array_Batch, verbose=0)
                        scoreList.append(scores)
                        print(scores)
                    except Exception as e:
                        print(e)
                        print("Breaking on the Training ...")

            # df_real_price = pd.DataFrame(dataset_test)
            df_real_price = dataset_all.iloc[TrainEnd - 1:]
            # df_predicted_price = pd.DataFrame(predicted_price)
            df_predicted_price = pd.DataFrame(predicted_price, index=dataset_all.iloc[TrainEnd - 1:-1].index, columns=['PredictedPrice'])

            if HistoryPlot == 1:
                print(history.history.keys())
                plt.plot(history.history['loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()

            if PredictionsPlot[0] == 1:
                if PredictionsPlot[1] == 'cs':
                    pd.concat([df_real_price, df_predicted_price], axis=1).plot()
                elif PredictionsPlot[1] == 'NoCs':
                    Slider.cs(pd.concat([df_real_price, df_predicted_price], axis=1)).plot()
                plt.show()

            return [df_real_price, df_predicted_price, pd.DataFrame(scoreList)]

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

