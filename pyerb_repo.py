from datetime import datetime
import pandas as pd, numpy as np, blpapi, sqlite3, time, matplotlib.pyplot as plt, itertools, types, multiprocessing, ta, sqlite3, xlrd
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, t
from scipy.stats import skew, kurtosis, entropy
from sklearn import linear_model
from sklearn.decomposition import PCA
from itertools import combinations
from ta.volume import *
from optparse import OptionParser

class pyerb:

    # Math Operators

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
        return df.pct_change(nperiods).fillna(0)

    def dret(df, **kwargs):
        if 'AUM' in kwargs:
            AUM = kwargs['AUM']
        else:
            AUM = 100000
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1

        dfOut = pyerb.d(df).fillna(0) / AUM
        return dfOut

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

        out = df.shift(periods=nperiods).fillna(0)

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
        return df

    def roller(df, func, n):
        ROLL = df.rolling(window=n, center=False).apply(lambda x: func(x), raw=True)
        return ROLL

    def gapify(df, **kwargs):
        if 'steps' in kwargs:
            steps = kwargs['steps']
        else:
            steps = 5

        gapifiedDF = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
        gapifiedDF.iloc[::steps, :] = df.iloc[::steps, :]

        gapifiedDF = gapifiedDF.ffill()

        return gapifiedDF

    def expander(df, func, n):
        EXPAND = df.expanding(min_periods=n, center=False).apply(lambda x: func(x))
        return EXPAND

    # Other operators (file readers, chunks etc.)

    def read_date(date):
        return xlrd.xldate.xldate_as_datetime(date, 0)

    def chunkReader(name):
        df = pd.read_csv(name, delimiter=';', chunksize=10000)
        return df

    # Quantitative Finance

    def sharpe(df):
        return df.mean() / df.std()

    def drawdown(pnl):
        """
        calculate max drawdown and duration
        Input:
            pnl, in $
        Returns:
            drawdown : vector of drawdwon values
            duration : vector of drawdown duration
        """
        cumret = pnl.cumsum()

        highwatermark = [0]

        idx = pnl.index
        drawdown = pd.Series(index=idx)
        drawdowndur = pd.Series(index=idx)

        for t in range(1, len(idx)):
            highwatermark.append(max(highwatermark[t - 1], cumret[t]))
            drawdown[t] = (highwatermark[t] - cumret[t])
            drawdowndur[t] = (0 if drawdown[t] == 0 else drawdowndur[t - 1] + 1)

        return drawdown, drawdowndur

    def rollVol(df, nIn):
        rollVolDF = pyerb.roller(df, np.std, nIn)
        return rollVolDF

    def rollNormalise(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'standardEMA'
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 25

        if mode == 'standardEMA':
            rollNormaliserDF = pyerb.ema(df, nperiods=nIn) / pyerb.rollVol(df, nIn=nIn)
        return rollNormaliserDF

    def rollStatistics(df, method, **kwargs):
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
       else:
            nIn = 25
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.01

        if method == 'Vol':
            rollStatisticDF = pyerb.roller(df, np.std, nIn)
        elif method == 'Skewness':
            rollStatisticDF = pyerb.roller(df, skew, nIn)
        elif method == 'Kurtosis':
            rollStatisticDF = pyerb.roller(df, kurtosis, nIn)
        elif method == 'VAR':
            rollStatisticDF = norm.ppf(1 - alpha) * pyerb.rollVol(df, nIn=nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'CVAR':
            rollStatisticDF = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * pyerb.rollVol(df, nIn=nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'Sharpe':
            rollStatisticDF = pyerb.roller(df, pyerb.sharpe, nIn)

        return rollStatisticDF

    def maxDD_DF(df):
        ddList = []
        for j in df:
            df0 = pyerb.d(df[j])
            maxDD = pyerb.drawdown(df0)[0].max()
            ddList.append([j, -maxDD])

        ddDF = pd.DataFrame(ddList)
        ddDF.columns = ['Strategy', 'maxDD']
        ddDF = ddDF.set_index('Strategy')
        return ddDF

    def profitRatio(pnl):
        '''
        calculate profit ratio as sum(pnl)/drawdown
        Input: pnl  - daily pnl, Series or DataFrame
        '''

        def processVector(pnl):  # process a single column
            s = pnl.fillna(0)
            dd = pyerb.drawdown(s)[0]
            p = s.sum() / dd.max()
            return p

        if isinstance(pnl, pd.Series):
            return processVector(pnl)

        elif isinstance(pnl, pd.DataFrame):

            p = pd.Series(index=pnl.columns)

            for col in pnl.columns:
                p[col] = processVector(pnl[col])

            return p
        else:
            raise TypeError("Input must be DataFrame or Series, not " + str(type(pnl)))

    # Technical Analysis Operators

    def sma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        SMA = df.rolling(nperiods).mean().fillna(0)
        return SMA

    def ema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = df.ewm(span=nperiods, min_periods=nperiods).mean().fillna(0)
        return EMA

    def bb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2
        dfBB = pd.DataFrame(df)
        dfBB['Price'] = df
        dfBB['rolling_mean'] = ema(df, nperiods)
        dfBB['rolling_std'] = df.rolling(nperiods).std()

        dfBB['MIDDLE'] = dfBB['rolling_mean']
        dfBB['UPPER'] = dfBB['MIDDLE'] + (dfBB['rolling_std'] * no_of_std)
        dfBB['LOWER'] = dfBB['MIDDLE'] - (dfBB['rolling_std'] * no_of_std)

        return dfBB[['Price', 'UPPER', 'MIDDLE', 'LOWER']]

    def rsi(df, n):
        i = 0
        UpI = [0]
        DoI = [0]
        while i + 1 < len(df):
            UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
            DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
            if UpMove > DoMove and UpMove > 0:
                UpD = UpMove
            else:
                UpD = 0
            UpI.append(UpD)
            if DoMove > UpMove and DoMove > 0:
                DoD = DoMove
            else:
                DoD = 0
            DoI.append(DoD)
            i = i + 1
        UpI = pd.Series(UpI)
        DoI = pd.Series(DoI)
        PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
        NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
        RSI = pd.DataFrame(PosDI / (PosDI + NegDI))
        RSI.columns = ['RSI']
        return RSI

    # Signals
    def sbb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        if 'rev' in kwargs:
            rev = kwargs['rev']
        else:
            rev = 1

        signalList = []
        for c in df.columns:
            if c != 'Date':
                cBB = pyerb.bb(df[c], nperiods=nperiods)
                cBB['Position'] = None
                cBB['Position'][(cBB['Price'] > cBB['UPPER']) & (pyerb.S(cBB['Price']) < pyerb.S(cBB['UPPER']))] = 1
                cBB['Position'][(cBB['Price'] < cBB['LOWER']) & (pyerb.S(cBB['Price']) > pyerb.S(cBB['LOWER']))] = -1
                cBB[c] = cBB['Position']
                signalList.append(cBB[c])
        s = pd.concat(signalList, axis=1)
        if rev == 1:
            s = s * (-1)
        return s

    def sign_rsi(rsi, r, ob, os, **kwargs):
        if 'fix' in kwargs:
            fix = kwargs['fix']
        else:
            fix = 5
        df = rsi.copy()
        print(rsi.copy())
        print('RSI=' + str(len(df)))
        print('RSI=' + str(len(rsi)))
        df[df > ob] = 1
        df[df < os] = -1
        df.iloc[0, 0] = -1
        print(df)
        for i in range(1, len(rsi)):
            if df.iloc[i, 0] != 1 and df.iloc[i, 0] != -1:
                df.iloc[i, 0] = df.iloc[i - 1, 0]

        df = np.array(df)
        df = np.repeat(df, fix)  # fix-1 gives better sharpe
        df = pd.DataFrame(df)
        print('ASSET=' + str(r))
        print('RSI x ' + str(fix) + '=' + str(len(df)))
        c = r - len(df)  # pnl - rsi diff
        # c = r%len(df)
        print('----------------------DIFF=' + str(c))
        df = df.append(df.iloc[[-1] * c])
        df = df.reset_index(drop=True)
        print(df)
        return df

    # Advanced Operators for Portfolio Management and Optimization

    def ExPostOpt(pnl):
        MSharpe = pyerb.sharpe(pnl)
        switchFlag = np.array(MSharpe) < 0
        pnl.iloc[:, np.where(switchFlag)[0]] = pnl * (-1)
        return [pnl, switchFlag]

    def RV(df, **kwargs):
        if "RVspace" in kwargs:
            RVspace = kwargs["RVspace"]
        else:
            RVspace = "classic"
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'linear'
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 25

        if RVspace == "classic":
            cc = list(combinations(df.columns, 2))
            if mode == 'linear':
                df0 = pd.concat([df[c[0]].sub(df[c[1]]) for c in cc], axis=1, keys=cc)
            elif mode == 'priceRatio':
                df0 = pd.concat([df[c[0]]/df[c[1]] for c in cc], axis=1, keys=cc)
            elif mode == 'priceRatio_zScore':
                lDF = []
                for c in cc:
                    PrRatio = df[c[0]] / df[c[1]]
                    emaPrRatio = pyerb.ema(PrRatio, nperiods=n)
                    volPrRatio = pyerb.expander(PrRatio, np.std, n)
                    PrZScore = (PrRatio-emaPrRatio) / volPrRatio
                    lDF.append(PrZScore)
                df0 = pd.concat(lDF, axis=1, keys=cc)
            elif mode == 'beta':
                df0 = pd.concat([df[c[0]].sub((pd.rolling_cov(df[c[0]], df[c[1]], window=25) / pd.rolling_var(df[c[1]], window=25)) * df[c[1]]) for c in cc], axis=1, keys=cc)

            df0.columns = df0.columns.map('_'.join)

        else:
            print("Projection based on RVSpace asset ... ")
            rvList = []
            for c in df.columns:
                rvDF = df[c].sub(df[RVspace])
                rvDF.name = c + "_" + RVspace
            df0 = pd.concat([], axis=1)

        return df0.fillna(method='ffill').fillna(0)

    def Baskets(df):
        cc = list(combinations(df.columns, 2))
        df0 = pd.concat([df[c[0]].add(df[c[1]]) for c in cc], axis=1, keys=cc)
        df0.columns = df0.columns.map('_'.join)
        return df0

    def RVSignalHandler(sigDF):
        assetSignList = []
        for c in sigDF.columns:
            medSigDF = pd.DataFrame(sigDF[c])
            assetNames = c.split("_")
            medSigDF[assetNames[0]] = sigDF[c]
            medSigDF[assetNames[1]] = sigDF[c] * (-1)
            assetSignList.append(medSigDF[[assetNames[0], assetNames[1]]])
        assetSignDF = pd.concat(assetSignList, axis=1)
        assetSignDFgroupped = assetSignDF.groupby(assetSignDF.columns, axis=1).sum()
        return assetSignDFgroupped

    # Metric Build

    def Metric(df, **kwargs):
       if 'statistic' in kwargs:
            statistic = kwargs['statistic']
        else:
            statistic = None
        if "metric" in kwargs:
            metric = kwargs['metric']
        else:
            metric = "euclidean"
        if "minkowskiOrder" in kwargs:
            minkowskiOrder = kwargs['minkowskiOrder']
        else:
            minkowskiOrder = 3
        if "wminkowskiWeight" in kwargs:
            wminkowskiWeight = kwargs['wminkowskiWeight']
        else:
            wminkowskiWeight = 0.25
        if "seuclideanV" in kwargs:
            seuclideanV = kwargs['seuclideanV']
        else:
            seuclideanV = 1

        MetricMat = pd.DataFrame(index=df.columns, columns=df.columns)

        "CALCULATE ROLLING STATISTIC"
        if statistic is not None:
            metaDF = pyerb.rollStatistics(df.copy(), statistic)

        for c1 in metaDF.columns:
            for c2 in metaDF.columns:
                if metric == "euclidean":
                    MetricMat.loc[c1,c2] = np.sqrt(((metaDF[c1] - metaDF[c2])**2).sum())
                elif metric == "manhattan":
                    MetricMat.loc[c1, c2] = (metaDF[c1] - metaDF[c2]).abs().sum()
                elif metric == "chebyshev":
                    MetricMat.loc[c1, c2] = (metaDF[c1] - metaDF[c2]).abs().max()
                elif metric == "minkowski":
                    MetricMat.loc[c1, c2] = ((((metaDF[c1] - metaDF[c2]).abs())**minkowskiOrder).sum())**(1/minkowskiOrder)
                elif metric == "wminkowski":
                    MetricMat.loc[c1, c2] = ((((metaDF[c1] - metaDF[c2]) * wminkowskiWeight)**minkowskiOrder).sum())**(1/minkowskiOrder)
                elif metric == "seuclidean":
                    MetricMat.loc[c1, c2] = np.sqrt(((metaDF[c1] - metaDF[c2])**2 / seuclideanV).sum())

        eigVals, eigVecs = np.linalg.eig(MetricMat.apply(pd.to_numeric, errors='coerce').fillna(0))

        return [eigVals, eigVecs]

    # Folders Body & Plots

    def Navigate(DB, module):

        navigatorData = []
        for a in dir(module):
            if isinstance(getattr(module, a), types.FunctionType):
                # print(inspect.getfullargspec(getattr(module, a)))
                navigatorData.append(['ParentFunction', a, ','.join(getattr(module, a).__code__.co_varnames)])
            elif isinstance(getattr(module, a), types.ModuleType):
                subModule = getattr(module, a)
                for b in dir(subModule):
                    if isinstance(getattr(subModule, b), types.FunctionType):
                        navigatorData.append(
                            ['SubModuleFunction', b, ','.join(getattr(subModule, b).__code__.co_varnames)])

        navigatorDataDF = pd.DataFrame(navigatorData, columns=['FunctionType', 'FunctionName', 'Parameters'])
        navigatorDataDF.to_sql("ModuleNavigator", sqlite3.connect(DB), if_exists='replace')

    def Plot(df, **kwargs):
        if 'title' in kwargs:
            titleIn = kwargs['title']
        else:
            titleIn = 'PyERB Chart'

       if titleIn == 'ew_sharpe':
            df.plot(title="Strategy Sharpe Ratio = " + str(pyerb.sharpe(pyerb.E(df)).round(2)))
        elif titleIn == 'cs_ew_sharpe':
            fig, axes = plt.subplots(nrows=2, ncols=1)
            pyerb.cs(df).plot(ax=axes[0], title="Individual Contributions")
            pyerb.cs(pyerb.E(df)).plot(ax=axes[1])
        else:
            df.plot(title=titleIn)
        plt.show()

    def RefreshableFile(dfIn, filename, refreshSecs):
        dfIn.to_html(filename)

        append_copy = open(filename, "r")
        original_text = append_copy.read()
        append_copy.close()

        append_copy = open(filename, "w")
        append_copy.write('<meta http-equiv="refresh" content="' + str(refreshSecs) + '">\n')
        append_copy.write(original_text)
        append_copy.close()

# SubClasses

class BackTester:

    def backTestReturnKernel(kernel, tradedAssets, **kwargs):

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'directionalPredictability'

        if 'TrShift' in kwargs:
            TrShift = kwargs['TrShift']
        else:
            TrShift = 1

        if 'reverseFlag' in kwargs:
            reverseFlag = kwargs['reverseFlag']
        else:
            reverseFlag = 1

        if 'scanAll' in kwargs:
            scanAll = kwargs['scanAll']
        else:
            scanAll = 'no'

        if mode == 'directionalPredictability':
            kernel = pyerb.sign(kernel)
        else:
            print("Using defaulet 'Direct trading kernel projection' to traded assets ...")

        if isinstance(kernel, pd.Series):
            kernel = pd.DataFrame(kernel)
        if isinstance(tradedAssets, pd.Series):
            tradedAssets = pd.DataFrame(tradedAssets)

        if (len(kernel.columns) != len(tradedAssets.columns)) | (scanAll == 'yes'):
            print("Kernel's dimension is not the same with the dimension of the Traded Assets matrix - building BT crosses...")
            cc = []
            for ck in kernel.columns:
                for c in tradedAssets.columns:
                    cc.append((ck, c))
            pnl = pd.concat([pyerb.S(kernel[c[0]], nperiods=TrShift) * pyerb.dlog(tradedAssets[c[1]]) * reverseFlag for c in cc], axis=1, keys=cc)
        else:
            print("Straight BT Projection...")
            pnl = pyerb.S(kernel, nperiods=TrShift) * pyerb.dlog(tradedAssets) * reverseFlag
        return pnl

class AI:

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

        if 'CustomMetric' in kwargs:
            CustomMetric = kwargs['CustomMetric']
        else:
            CustomMetric = "euclidean"

        if 'CustomMetricStatistic' in kwargs:
            CustomMetricStatistic = kwargs['CustomMetricStatistic']
        else:
            CustomMetricStatistic = None

        eigDf = []
        eigCoeffs = [[] for j in range(len(eigsPC))]
        Comps = [[] for j in range(len(eigsPC))]
        sigmaList = []
        lambdasList = []
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

            if manifoldIn == 'CustomMetric':
                customMetric = pyerb.Metric(df, statistic=CustomMetricStatistic, metric=CustomMetric)
                lambdasList.append(list(customMetric[0]))
                sigmaList.append(list(customMetric[0]))
                c = 0
                for eig in eigsPC:
                    #print(eig, ', customMetric[1][eig] =', customMetric[1][eig]) # 0 , 100 , 5
                    Comps[c].append(list(customMetric[1][eig]))
                    c += 1

            elif manifoldIn == 'PCA':
                pca = PCA(n_components=NumProjections)
                X_pca = pca.fit_transform(x)
                lambdasList.append(list(pca.singular_values_))
                sigmaList.append(list(pca.explained_variance_ratio_))
                c = 0
                for eig in eigsPC:
                    #print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                    Comps[c].append(list(pca.components_[eig]))
                    c += 1

            elif manifoldIn == 'DMAPS':
                dMapsOut = pyerb.AI.gDmaps(df, nD=NumProjections, coFlag=contractiveObserver,
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
            if len(sigmaDF.columns) <= 1:
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

                exPostProjections[k] = df0 * pyerb.S(principalCompsDf[k])

            return [df0, principalCompsDf, exPostProjections, sigmaDF, lambdasDF]

        else:

            return [df0, pd.DataFrame(eigDf), pd.DataFrame(cObserverList), sigmaDF, lambdasDF, Comps, eigCoeffs]

class PyBloomberg:

    def __init__(self, DB):
        self.DB = DB

    def CustomDataFetch(TargetDB, name, assetsToProcess, fieldsIn):

        dataOutCustomData = []

        EXCEPTIONS = blpapi.Name("exceptions")
        FIELD_ID = blpapi.Name("fieldId")
        REASON = blpapi.Name("reason")
        CATEGORY = blpapi.Name("category")
        DESCRIPTION = blpapi.Name("description")
        ERROR_INFO = blpapi.Name("ErrorInfo")

        class Window(object):
            def __init__(self, name):
                self.name = name

            def displaySecurityInfo(self, msg):
                print("%s: %s" % (self.name, msg))
                # print("%s:" % (self.name))

                d = msg.getElement('securityData')
                size = d.numValues()
                fieldDataList = [[d.getValueAsElement(i).getElement("security").getValueAsString(),
                                  d.getValueAsElement(i).getElement("fieldData")] for i in range(0, size)]
                for x in fieldDataList:
                    subData = []
                    # print(x, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    for fld in fieldsIn:
                        try:
                            subData.append([fld, x[1].getElement(fld).getValueAsString()])
                        except Exception as e:
                            # print(e)
                            subData.append([fld, np.nan])
                        subDataDF = pd.DataFrame(subData, columns=["field", x[0]]).set_index('field', drop=True)
                    dataOutCustomData.append(subDataDF)

                dfOUT = pd.concat(dataOutCustomData, axis=1).T
                dfOUT.to_sql(name, sqlite3.connect(TargetDB), if_exists='replace')

        def parseCmdLine():
            parser = OptionParser(description="Retrieve reference data.")
            parser.add_option("-a",
                              "--ip",
                              dest="host",
                              help="server name or IP (default: %default)",
                              metavar="ipAddress",
                              default="localhost")
            parser.add_option("-p",
                              dest="port",
                              type="int",
                              help="server port (default: %default)",
                              metavar="tcpPort",
                              default=8194)

            (options, args) = parser.parse_args()

            return options

        def startSession(session):
            if not session.start():
                print("Failed to connect!")
                return False

            if not session.openService("//blp/refdata"):
                print("Failed to open //blp/refdata")
                session.stop()
                return False

            return True

        global options
        options = parseCmdLine()

        # Fill SessionOptions
        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost(options.host)
        sessionOptions.setServerPort(options.port)

        print("Connecting to %s:%d" % (options.host, options.port))

        # Create a Session
        session = blpapi.Session(sessionOptions)

        # Start a Session
        if not startSession(session):
            return

        refDataService = session.getService("//blp/refdata")
        request = refDataService.createRequest("ReferenceDataRequest")
        for asset in assetsToProcess:
            request.append("securities", asset)
        for fld in fieldsIn:
            request.append("fields", fld)

        secInfoWindow = Window("SecurityInfo")
        cid = blpapi.CorrelationId(secInfoWindow)

        # print("Sending Request:", request)
        session.sendRequest(request, correlationId=cid)

        try:
            # Process received events
            while (True):
                # We provide timeout to give the chance to Ctrl+C handling:
                event = session.nextEvent(500)
                for msg in event:
                    if event.eventType() == blpapi.Event.RESPONSE or \
                            event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                        window = msg.correlationIds()[0].value()
                        window.displaySecurityInfo(msg)

                # Response completly received, so we could exit
                if event.eventType() == blpapi.Event.RESPONSE:
                    break
        finally:
            # Stop the session
            session.stop()