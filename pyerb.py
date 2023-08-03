from datetime import datetime
import pandas as pd, numpy as np, math, sqlite3, time, matplotlib.pyplot as plt, itertools, types, multiprocessing, ta, sqlite3, xlrd
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, t
from scipy.stats import skew, kurtosis, entropy
from pykalman import pykalman
import statsmodels.tsa.stattools as ts
from scipy.linalg import svd
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn import linear_model
from fracdiff2 import frac_diff_ffd
from sklearn.decomposition import PCA
from itertools import combinations, permutations
from ta.volume import *
from optparse import OptionParser
from hurst import compute_Hc

class pyerb:

    "Math Operators"
    def d(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        out = df.diff(nperiods)
        return out

    def dlog(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        if 'fillna' in kwargs:
            fillna = kwargs['fillna']
        else:
            fillna = "yes"

        out = pyerb.d(np.log(df), nperiods=nperiods)

        if fillna == "yes":
            out = out.fillna(0)
        return out

    def frac(df, **kwargs):
        if 'fracOrder' in kwargs:
            fracOrder = kwargs['fracOrder']
        else:
            fracOrder = 1
        if 'fillna' in kwargs:
            fillna = kwargs['fillna']
        else:
            fillna = "yes"

        fracDF = pd.DataFrame(index=df.index, columns=df.columns)
        for c in df.columns:
            fracDF[c] = frac_diff_ffd(df[c].values, d=fracOrder)

        if fillna == "yes":
            fracDF = fracDF.fillna(0)

        return fracDF

    def r(df, **kwargs):
        if 'calcMethod' in kwargs:
            calcMethod = kwargs['calcMethod']
        else:
            calcMethod = 'Continuous'
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        if 'fillna' in kwargs:
            fillna = kwargs['fillna']
        else:
            fillna = "yes"

        if calcMethod == 'Continuous':
            out = pyerb.d(np.log(df), nperiods=nperiods)
        elif calcMethod == 'Discrete':
            out = df.pct_change(nperiods)
        if calcMethod == 'Linear':
            diffDF = pyerb.d(df, nperiods=nperiods)
            out = diffDF.divide(df.iloc[0])

        if fillna == "yes":
            out = out.fillna(0)
        return out

    def E(df):
        out = df.mean(axis=1)
        return out

    def rs(df):
        out = df.sum(axis=1)
        return out

    def ew(df):
        out = np.log(pyerb.E(np.exp(df)))
        return out

    def cs(df):
        out = df.cumsum()
        return out

    def ecs(df):
        out = np.exp(df.cumsum())
        return out

    def rp(df, **kwargs):
        if "nIn" in kwargs:
            nIn = kwargs["nIn"]
        else:
            nIn = 250
        RollVol = np.sqrt(252) * pyerb.rollStatistics(df, method="Vol", nIn=nIn) * 100
        out = (df / pyerb.S(RollVol)).fillna(0)
        return out

    def pb(df):
        out = np.log(pyerb.rs(np.exp(df)))
        return out

    def beta(df,yVar,xVar,**kwargs):
        if "n" in kwargs:
            n = kwargs["n"]
        else:
            n = 250
        out = df[xVar].rolling(n).corr(df[yVar]) * (pyerb.roller(df[xVar], np.std, n) / pyerb.roller(df[yVar], np.std, n))
        return out

    def sign(df):
        #df[df > 0] = 1
        #df[df < 0] = -1
        #df[df == 0] = 0
        out = np.sign(df)
        return out

    def S(df, **kwargs):

        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1

        out = df.shift(periods=nperiods)

        return out

    def fd(df):
        out = df.replace([np.inf, -np.inf], 0).fillna(0)
        return out

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

    def rowStoch(df):
        out = df.div(df.abs().sum(axis=1), axis=0)
        return out

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

    "Other operators (file readers, chunks etc.)"

    def read_date(date):
        return xlrd.xldate.xldate_as_datetime(date, 0)

    def chunkReader(name):
        df = pd.read_csv(name, delimiter=';', chunksize=10000)
        return df

    "Quantitative Finance"

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

    def rollNormalise(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'standardSMA'
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 250

        if mode == 'standardSMA':
            rollNormaliserDF = (df - df.rolling(nIn).mean()) / df.rolling(nIn).std()
        if mode == 'standardEMA':
            rollNormaliserDF = (df - pyerb.ema(df, nperiods=nIn)) / df.rolling(nIn).std()
        elif mode == 'MinMax':
            rollNormaliserDF = (df - df.rolling(nIn).min()) / (df.rolling(nIn).max() - df.rolling(nIn).min())
        elif mode == 'standardExpandingSMA':
            rollNormaliserDF = (df - df.expanding(nIn).mean()) / df.expanding(nIn).std()
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
            rollStatisticDF = norm.ppf(1 - alpha) * pyerb.roller(df, np.std, nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'CVAR':
            rollStatisticDF = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * pyerb.roller(df, np.std, nIn) - pyerb.ema(df, nperiods=nIn)
        elif method == 'Sharpe':
            rollStatisticDF = pyerb.roller(df, pyerb.sharpe, nIn)
        elif method == 'Hurst':
            rollStatisticDF = pyerb.roller(df, pyerb.gHurst, nIn)
        elif method == 'DirectionalStatistic':
            signedDF = pyerb.sign(df)
            rollStatisticDF = pyerb.cs(signedDF)
        elif method == 'Cointegration':
            if 'specificPairs' in kwargs:
                pairs = kwargs['specificPairs']
            else:
                pairs = []
                for c1 in df.columns:
                    for c2 in df.columns:
                        pairs.append((c1, c2))

            dataList = []
            for pair in pairs:
                subCoints = []
                for i in tqdm(range(nIn, len(df) + 1)):
                    subX = df.iloc[i-nIn:i, list(df.columns).index(pair[0])]
                    subY = df.iloc[i-nIn:i, list(df.columns).index(pair[1])]
                    coint_p_Value = ts.coint(subX, subY)[1]
                    subCoints.append([subX.index[-1],coint_p_Value])
                subDF = pd.DataFrame(subCoints, columns=["Dates", pair[0]+"_"+pair[1]]).set_index("Dates", drop=True)
                dataList.append(subDF)
            rollStatisticDF = pd.concat(dataList, axis=1).sort_index()

        return rollStatisticDF

    def rollOLS(df, **kwargs):
        if "selList" in kwargs:
            selList = kwargs['selList']
        else:
            selList = df.columns
        if "Y" in kwargs:
            Y = kwargs["Y"]
        else:
            Y = selList[0]
        if "X" in kwargs:
            X = kwargs["X"]
        else:
            X = selList
        if "n" in kwargs:
            n = kwargs["n"]
        else:
            n = 250

        model = RollingOLS(endog=df[Y], exog=df[X], window=n)
        rres = model.fit()
        return rres.params.sort_index()

    def roll_OLS_Regress(df, targetAsset, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 250
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = "1-to-1"
        if "X" in kwargs:
            X = kwargs["X"]
        else:
            X = [x for x in df.columns if x != targetAsset]

        if mode == "1-to-1":
            OLS_Betas_List = []
            for c in X:
                OLS_Betas_List.append(pyerb.rollOLS(df, Y=[targetAsset], X=c, n=n))
            OLS_Betas_DF = pd.concat(OLS_Betas_List, axis=1)
        elif mode == "MultipleRegression":
            OLS_Betas_DF = pyerb.rollOLS(df, Y=[targetAsset], X=X, n=n)

        OLS_Mappings_DF = OLS_Betas_DF * df[X]

        return [OLS_Betas_DF, OLS_Mappings_DF]

    def gRollingHurst(df0, **kwargs):

        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 25

        HurstDF = pd.DataFrame(None, index=df0.index, columns=df0.columns)
        for i in tqdm(range(st, len(df0) + 1)):

            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]

            for c in df.columns:
                try:
                    H, Hc, Hdata = compute_Hc(df[c], kind='change', simplified=True)
                except Exception as e:
                    H = None

                HurstDF.loc[df.index[-1], c] = H

        return HurstDF

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

    def Roll_Max_DD(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 25

        ddDF = df.copy()
        for c in df.columns:
            ddDF[c] = df[c].rolling(nperiods).apply(lambda s: (s - s.cummax()).min())

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

    def SelectLookBack(df, **kwargs):
        if "method" in kwargs:
            method = kwargs['method']
        else:
            method = "EMA"

        if method == "EMA":
            columnSet = ["Asset", "Lag", "shiftLag", "Sharpe"]

        outList = []
        for c in df.columns:
            if method == "EMA":
                for l in [10, 25, 50, 125, 250, 500]:#[10, 25, 50, 125, 250, 500], range(5,750, 5):
                    for s in [2]:#[1,2]:
                        subPnl = pyerb.S(pyerb.sign(pyerb.ema(df[c], nperiods=l)), nperiods=s) * df[c]
                        subSh = np.sqrt(252) * pyerb.sharpe(subPnl)
                        outList.append([c,l,s,subSh])
        ###############################################################################################################
        outDF = pd.DataFrame(outList, columns=columnSet)
        outDF = outDF.sort_values(['Asset', 'Sharpe'], ascending=False).groupby('Asset').head()
        outDF = outDF.set_index("Asset", drop=True)
        ###############################################################################################################
        BestLagList = []
        for c in df.columns:
            bestLag = outDF.loc[c,"Lag"].values[0]
            BestLagList.append([c,bestLag])
        BestLagDF = pd.DataFrame(BestLagList, columns=["Asset", "Lag"]).set_index("Asset", drop=True)
        ###############################################################################################################
        return BestLagDF

    "Smoothers"

    def DynamicSelectLookBack(df0, **kwargs):
        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 500+1
        if st < 500:
            print("You need at least 500 observations for the 'SelectLookBack' function to run .. setting st to 500+1")
            st = 500+1

        outList = []
        for i in tqdm(range(st, len(df0) + 1)):
            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]

            out = pyerb.SelectLookBack(df, method="EMA")
            out.columns = [df.index[-1]]
            outList.append(out)

        outDF = pd.concat(outList,axis=1).T.sort_index()
        outDF.index.names = df0.index.names
        return outDF

    def RollKalman(df, **kwargs):

        if "mode" in kwargs:
            mode = kwargs['mode']
        else:
            mode = "Roll"

        if "st" in kwargs:
            st = kwargs['st']
        else:
            st = 25

        if "ocMode" in kwargs:
            ocMode = kwargs['ocMode']
        else:
            ocMode = ["fixed",1]

        if "tcMode" in kwargs:
            tcMode = kwargs['tcMode']
        else:
            tcMode = ["fixed",1]

        rollKalmanDF = pd.DataFrame(None, index=df.index, columns=df.columns)
        rollKalmanTotalDF = pd.DataFrame(None, index=df.index, columns=df.columns)
        for c in df.columns:
            print(c)
            for i in tqdm(range(st,len(df)+1)):
                # //////////////////////////////////////////////////////////////////////////////////////////////
                if mode == "Roll":
                    subDF = df[c].iloc[i-st:i]
                elif mode == "Exp":
                    subDF = df[c].iloc[0:i]
                # //////////////////////////////////////////////////////////////////////////////////////////////
                if ocMode[0] == "fixed":
                    observation_covariance_In = ocMode[1]
                elif ocMode[0] == "10-SigmaRule":
                    observation_covariance_In = abs(subDF.std())*10
                elif ocMode[0] == "100-SigmaRule":
                    observation_covariance_In = abs(subDF.std())*100
                # //////////////////////////////////////////////////////////////////////////////////////////////
                if tcMode[0] == "fixed":
                    transition_covariance_In = tcMode[1]
                elif tcMode[0] == "1-SigmaRule":
                    transition_covariance_In = abs(subDF.std())
                elif tcMode[0] == "3-SigmaRule":
                    transition_covariance_In = abs(subDF.std()) * 3
                elif tcMode[0] == "MinMaxRule":
                    transition_covariance_In = abs(subDF.max()-subDF.min()/subDF.min())
                # //////////////////////////////////////////////////////////////////////////////////////////////
                kf = pykalman.KalmanFilter(transition_matrices=[1],  # The value for At. It is a random walk so is set to 1.0
                                           observation_matrices=[1],  # The value for Ht.
                                           initial_state_mean=0,  # Any initial value. It will converge to the true state value.
                                           initial_state_covariance=1,# Sigma value for the Qt in Equation (1) the Gaussian distribution
                                           observation_covariance=observation_covariance_In,# Sigma value for the Rt in Equation (2) the Gaussian distribution
                                           transition_covariance=transition_covariance_In)  # A small turbulence in the random walk parameter 1.0

                state_means, _ = kf.filter(subDF)
                if i == len(df):
                    rollKalmanTotalDF[c] = pd.DataFrame(state_means, index=subDF.index).iloc[:,0]

                rollKalmanDF.loc[subDF.index[-1],c] = state_means[-1][0]

        SMAs = pyerb.sma(df, nperiods=st); SMAs.columns = [x+"_SMA" for x in SMAs.columns]
        EMAs = pyerb.ema(df, nperiods=st); EMAs.columns = [x+"_EMA" for x in EMAs.columns]

        rollKalmanTotalDF.columns = [x+"_KalmanTotal" for x in rollKalmanTotalDF.columns]

        return rollKalmanTotalDF

    def BBcompress(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 25

        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2

        if "KillMode" in kwargs:
            KillMode = kwargs['KillMode']
        else:
            KillMode = "Outwards"

        if "SmootherBase" in kwargs:
            SmootherBase = kwargs['SmootherBase']
        else:
            SmootherBase = "Raw"

        out = pd.DataFrame(None, index=df.index, columns=df.columns)
        for c in df.columns:
            subBB = pyerb.bb(df[c], nperiods=nperiods, no_of_std=no_of_std)

            if KillMode == "Outwards":

                if SmootherBase == "Raw":
                    subBB[c] = subBB["Price"]
                elif SmootherBase == "bbMA":
                    subBB[c] = subBB["MIDDLE"]
                subBB.loc[(subBB["Price"] > subBB["UPPER"])|(subBB["Price"] < subBB["LOWER"]), c] = None

            elif KillMode == "Inwards":
                subBB[c] = None
                subBB.loc[subBB["Price"] > subBB["UPPER"], c] = subBB["UPPER"]
                subBB.loc[subBB["Price"] < subBB["LOWER"], c] = subBB["LOWER"]

            out[c] = subBB[c]

        out = out.ffill().bfill()

        return out

    "Technical Analysis Operators"

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
        EMA = df.ewm(span=nperiods, min_periods=nperiods).mean().bfill()
        return EMA

    def dema(df, LookBacks, **kwargs):
        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            mode = "mean"

        #pd.set_option('max_row', None)
        uniqueLags = []
        for c in LookBacks.columns:
            for x in list(set(LookBacks[c].values)):
                uniqueLags.append(x)
        uniqueLags = list(set(uniqueLags))
        #########################################################################
        demaDF = pd.DataFrame(None, index=df.index, columns=df.columns)
        for l in uniqueLags:
            if mode == "mean":
                demaDF[LookBacks == l] = df.ewm(span=l, min_periods=l).mean()
            elif mode == "AnnVol":
                demaDF[LookBacks == l] = np.sqrt(252) * df.ewm(span=l, min_periods=l).std() * 100

        return demaDF.fillna(0)

    def bb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2
        dfBB = pd.DataFrame(df, index=df.index)
        dfBB['Price'] = df
        dfBB['rolling_mean'] = pyerb.sma(dfBB['Price'], nperiods=nperiods)
        dfBB['rolling_std'] = dfBB['Price'].rolling(window=nperiods).std()

        dfBB['MIDDLE'] = dfBB['rolling_mean']
        dfBB['UPPER'] = dfBB['MIDDLE'] + dfBB['rolling_std'] * no_of_std
        dfBB['LOWER'] = dfBB['MIDDLE'] - dfBB['rolling_std'] * no_of_std

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

    "Signals"
    def sbb(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3

        signalList = []
        for c in df.columns:
            if c != 'Date':
                cBB = pyerb.bb(df[c], nperiods=nperiods)
                cBB['Position'] = np.nan
                cBB['Position'][(cBB['Price'] > cBB['UPPER']) & (pyerb.S(cBB['Price']) <= pyerb.S(cBB['UPPER']))] = 1
                cBB['Position'][(cBB['Price'] < cBB['LOWER']) & (pyerb.S(cBB['Price']) >= pyerb.S(cBB['LOWER']))] = -1
                cBB[c] = cBB['Position']
                signalList.append(cBB[c])
        s = pd.concat(signalList, axis=1).ffill().fillna(0)
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

    "Advanced Operators for Portfolio Management and Optimization"

    def ExPostOpt(pnl):
        MSharpe = pyerb.sharpe(pnl)
        switchFlag = np.array(MSharpe) < 0
        pnl.iloc[:, np.where(switchFlag)[0]] = pnl * (-1)
        out = [pnl, switchFlag]
        return out

    def StaticHedgeRatio(df, targetAsset):
        HedgeRatios = []
        for c in df.columns:
            subHedgeRatio = df[targetAsset].corr(df[c]) * (df[targetAsset].std()/df[c].std())
            HedgeRatios.append(subHedgeRatio)
        HedgeRatiosDF = pd.Series(HedgeRatios, index=df.columns).drop(df[targetAsset].name)
        return HedgeRatiosDF

    def BetaRegression(df, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 250

        if 'SpecY' in kwargs:
            SpecY = kwargs['SpecY']
        else:
            SpecY = df.columns

        if 'SpecX' in kwargs:
            SpecX = kwargs['SpecX']
        else:
            SpecX = df.columns

        RollVols = pyerb.rollStatistics(df, 'Vol', nIn=n) ** 2

        BetaList = []
        for cY in SpecY:
            for cX in SpecX:
                Beta_c = (df[cY].rolling(n).cov(df[cX])).divide(RollVols[cX], axis=0).replace([np.inf, -np.inf], 0)
                Beta_c.name = cY+"_"+cX
                BetaList.append(Beta_c)
        BetaDF = pd.concat(BetaList, axis=1).fillna(0)

        return BetaDF

    def BetaKernel(df):

        BetaMatDF = pd.DataFrame(np.cov(df.T), index=df.columns, columns=df.columns)
        for idx, row in BetaMatDF.iterrows():
            BetaMatDF.loc[idx] /= row[idx]

        return BetaMatDF

    def MultiRegressKernel(df):

        dataList = []
        for c in df.columns:
            regr = linear_model.LinearRegression()
            regr.fit(df, df[c])
            dataList.append(regr.coef_)

        dataDF = pd.DataFrame(dataList, columns=df.columns, index=df.columns)

        return dataDF

    def RV(df, **kwargs):
        if "RVspace" in kwargs:
            RVspace = kwargs["RVspace"]
        else:
            RVspace = "classicPermutations"
        if "noDims" in kwargs:
            noDims = kwargs["noDims"]
        else:
            noDims = [2]
        if noDims[0] != 2:
            RVspace = "specificDriverPermutations_"+noDims[1]
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'Linear'
        if 'HedgeRatioConnectionMode' in kwargs:
            HedgeRatioConnectionMode = kwargs['HedgeRatioConnectionMode']
        else:
            HedgeRatioConnectionMode = "Spreads"
        if HedgeRatioConnectionMode == "Baskets":
            HedgeRatioConnectionSign = 1
        else:
            HedgeRatioConnectionSign = -1
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 25

        if RVspace == "classicPermutations":
            cc = list(permutations(df.columns, noDims[0]))
        elif RVspace == "classicCombos":
            cc = list(combinations(df.columns, noDims[0]))
        elif RVspace.split("_")[0] == "specificDriverclassicPermutations":
            cc = [c for c in list(permutations(df.columns, noDims[0])) if c[0] == RVspace.split("_")[1]]
        elif RVspace.split("_")[0] == "specificDriverPermutations":
            cc = [c for c in list(permutations(df.columns, noDims[0])) if c[0] == RVspace.split("_")[1]]
        elif RVspace.split("_")[0] == "specificDriverCombos":
            cc = [c for c in list(combinations(df.columns, noDims[0])) if c[0] == RVspace.split("_")[1]]
        elif RVspace.split("_")[0] == "specificLaggerclassicPermutations":
            cc = [c for c in list(permutations(df.columns, noDims[0])) if c[1] == RVspace.split("_")[1]]
        elif RVspace.split("_")[0] == "specificLaggerclassicCombos":
            cc = [c for c in list(combinations(df.columns, noDims[0])) if c[1] == RVspace.split("_")[1]]
        elif RVspace == "specificPairs":
            cc = kwargs["targetPairs"]

        if noDims[0] == 2:
            if mode == 'Linear':
                df0 = pd.concat([df[c[0]].sub(df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'Baskets':
                df0 = pd.concat([df[c[0]].add(df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'PriceRatio':
                df0 = pd.concat([df[c[0]]/df[c[1]] for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'PriceMultiply':
                df0 = pd.concat([df[c[0]] * df[c[1]] for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'PriceRatio_zScore':
                lDF = []
                for c in tqdm(cc):
                    PrRatio = df[c[0]] / df[c[1]]
                    emaPrRatio = pyerb.ema(PrRatio, nperiods=n)
                    volPrRatio = pyerb.expander(PrRatio, np.std, n)
                    PrZScore = (PrRatio-emaPrRatio) / volPrRatio
                    lDF.append(PrZScore)
                df0 = pd.concat(lDF, axis=1, keys=cc)
            elif mode == 'HedgeRatio':
                df0 = pd.concat([df[c[0]].rolling(n).corr(df[c[1]]) * (pyerb.roller(df[c[0]], np.std, n) / pyerb.roller(df[c[1]], np.std, n)) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatio_Expanding':
                df0 = pd.concat([df[c[0]].expanding(n).corr(df[c[1]]) * (pyerb.expander(df[c[0]], np.std, n) / pyerb.expander(df[c[1]], np.std, n)) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioPnL_Roll':
                df0 = pd.concat([(df[c[0]] + HedgeRatioConnectionSign * pyerb.S(df[c[0]].rolling(n).corr(df[c[1]]) * (
                            pyerb.roller(df[c[0]], np.std, n) / pyerb.roller(df[c[1]], np.std, n)))
                                             * df[c[1]]).fillna(0) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioPnL_Expand':
                df0 = pd.concat([(df[c[0]] + HedgeRatioConnectionSign * pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]) * (
                            pyerb.expander(df[c[0]], np.std, n) / pyerb.expander(df[c[1]], np.std, n)))
                                             * df[c[1]]).fillna(0) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioSimpleCorr':
                df0 = pd.concat([df[c[0]] + HedgeRatioConnectionSign * (pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]), nperiods=2) * df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
            df0.columns = df0.columns.map('_'.join)
        else:
            if mode == 'OLSpnl':
                df0 = pd.concat([pyerb.rs(pyerb.S(pyerb.rollOLS(df, selList=list(c), Y=c[0], X=list(c[1:]), n=n)) * df[list(c[1:])]).rename('_'.join(list(c))) for c in tqdm(cc)], axis=1)

        return df0.fillna(0).sort_index()

    def RVSignalHandler(sigDF, **kwargs):
        if 'HedgeRatioDF' in kwargs:
            HedgeRatioDF = kwargs['HedgeRatioDF']
        else:
            HedgeRatioDF = pd.DataFrame(1, index=sigDF.index, columns=sigDF.columns)
        assetSignList = []
        for c in sigDF.columns:
            medSigDF = pd.DataFrame(sigDF[c])
            HedgeRatio = HedgeRatioDF[c]
            assetNames = c.split("_")
            medSigDF[assetNames[0]] = sigDF[c]
            medSigDF[assetNames[1]] = sigDF[c] * (-1) * HedgeRatio
            subSigDF = medSigDF[[assetNames[0], assetNames[1]]]
            #print(subSigDF)
            assetSignList.append(subSigDF)
        assetSignDF = pd.concat(assetSignList, axis=1)
        #print(assetSignDF)
        assetSignDFgroupped = assetSignDF.groupby(assetSignDF.columns, axis=1).sum()
        #print(assetSignDFgroupped)
        #time.sleep(3000)
        return assetSignDFgroupped

    "Pricing"

    def black_scholes(S, K, T, r, rf, sigma, option_type):
        d1 = (np.log(S / K) + (r - rf + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            option_price = S * math.exp(-rf * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-rf * T) * norm.cdf(-d1)

        return option_price

    def black_scholes_greeks(S, K, T, r, rf, sigma, option_type):
        d1 = (np.log(S / K) + (r - rf + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        if option_type == "call":
            delta = np.exp(-rf * T) * N_d1
            gamma = np.exp(-rf * T - d1 ** 2 / 2) / (S * sigma * np.sqrt(T) * np.sqrt(2 * np.pi))
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T))) - (rf * K * np.exp(-rf * T) * N_d2)
            vega = S * np.exp(-rf * T) * n_d1 * np.sqrt(T)
            rho = K * T * np.exp(-r * T) * N_d2
        elif option_type == "put":
            delta = -np.exp(-rf * T) * (1 - N_d1)
            gamma = np.exp(-rf * T - d1 ** 2 / 2) / (S * sigma * np.sqrt(T) * np.sqrt(2 * np.pi))
            theta = (-S * n_d1 * sigma / (2 * np.sqrt(T))) + (rf * K * np.exp(-rf * T) * (1 - N_d2))
            vega = S * np.exp(-rf * T) * n_d1 * np.sqrt(T)
            rho = -K * T * np.exp(-r * T) * (1 - N_d2)

        return delta, gamma / 100, theta / 365, vega / 100, rho / 100

    "Metric Build"

    def Metric(metaDF, **kwargs):
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
        if "skipEigCalc" in kwargs:
            skipEigCalc = "Yes"
        else:
            skipEigCalc = "No"
        if "ExcludeSet" in kwargs:
            ExcludeSet = kwargs['ExcludeSet']
        else:
            ExcludeSet = []

        MetricMat = pd.DataFrame(index=metaDF.columns, columns=metaDF.columns)

        for c1 in metaDF.columns:
            for c2 in metaDF.columns:
                if (c1 in ExcludeSet)&(c2 in ExcludeSet):
                    MetricMat.loc[c1, c2] = None
                else:
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
                    elif metric == "MI":
                        MetricMat.loc[c1, c2] = mutual_info_score(metaDF[c1].values, metaDF[c2].values)
                    elif metric == "AdjMI":
                        MetricMat.loc[c1, c2] = adjusted_mutual_info_score(metaDF[c1].values, metaDF[c2].values)
                    elif metric == "NormMI":
                        MetricMat.loc[c1, c2] = normalized_mutual_info_score(metaDF[c1].values, metaDF[c2].values)

        if skipEigCalc == "No":
            eigVals, eigVecs = np.linalg.eig(MetricMat.apply(pd.to_numeric, errors='coerce').fillna(0))
        else:
            eigVals = []
            eigVecs = []

        return [eigVals, eigVecs, MetricMat]

    def RollMetric(df0, **kwargs):

        if "ID" in kwargs:
            ID = kwargs['ID']
        else:
            ID = "default"

        if "metric" in kwargs:
            metric = kwargs['metric']
        else:
            metric = "euclidean"

        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'ExpWindow'

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 25

        if "ExcludeSet" in kwargs:
            ExcludeSet = kwargs['ExcludeSet']
        else:
            ExcludeSet = []

        MetricMatCols = []
        for c1 in df0.columns:
            for c2 in df0.columns:
                MetricMatCols.append(c1+"_"+c2)
        MetricMatDF = pd.DataFrame(None, index=df0.index, columns=MetricMatCols)

        for i in tqdm(range(st, len(df0) + 1)):
            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]

            subMetric = pyerb.Metric(df, metric=metric, skipEigCalc="Yes", ExcludeSet=ExcludeSet)[2]

            for c1 in subMetric.columns:
                for c2 in subMetric.columns:
                    MetricMatDF.loc[df.index[-1],c1+"_"+c2] = subMetric.loc[c1,c2]

        return MetricMatDF

    "Folders Body & Plots"
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

    def RefreshableFile(dfList, filename, refreshSecs, **kwargs):
        pd.options.display.float_format = '{:,}'.format
        pd.set_option('colheader_justify', 'center')

        if 'addButtons' in kwargs:
            addButtons = kwargs['addButtons']
        else:
            addButtons = None
        if 'addPlots' in kwargs:
            addPlots = kwargs['addPlots']
        else:
            addPlots = None
        if 'cssID' in kwargs:
            cssID = kwargs['cssID']
        else:
            cssID = ''
        if 'specificID' in kwargs:
            specificID = kwargs['specificID']
        else:
            specificID = None

        dfListNew = []
        for x in dfList:
            dfListNew.append(x[0].to_html(index=False, table_id=x[1]) + "\n\n" + "<br>")

        with open(filename, 'w') as _file:
            _file.write(''.join(dfListNew))

        append_copy = open(filename, "r")
        original_text = append_copy.read()
        append_copy.close()

        append_copy = open(filename, "w")
        if specificID not in ["DeskPnLSinceInception", "BOFFICE_GATOS_DailyPnLDF_DERIV"]:
            append_copy.write('<meta http-equiv="refresh" content="' + str(refreshSecs) + '">\n')
        append_copy.write('<meta charset="UTF-8">')
        append_copy.write('<link rel="stylesheet" href="style/df_style.css"><link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css"><link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inconsolata">')
        append_copy.write('<div id="footerText"> LAST UPDATE : ' + datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " - Derivatives and FX Trading Desk</div>")

        if addButtons == 'GreenBoxMain':
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write('</div>')
        elif addButtons == 'RiskStatus':
            append_copy.write('<div>')
            append_copy.write("The table displays the probability of the assets under consideration, entering or already experiencing a 'Risk Off / Short' trend. The 'strength' of the signal is the probability level itself, with 50% being the threshold between up and down trends. 'Fairly' strong level to buy hedges would be the 80% and above.")
            append_copy.write('<br><img src="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/Credit_ETFs_RiskOffProbabilities.jpg" alt="Credit ETFs RiskOff Probs">')
            append_copy.write(
                '<br><a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/ERB_ML_RiskOffProbsDF.html">Eurobank ML Risk Off Index Page</a>')
            append_copy.write('</div>')
            ##############################
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_LQD US Equity.html">LQD US Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_IHYG LN Equity.html">IHYG LN Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_IEAC LN Equity.html">IEAC LN Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_HYG US Equity.html">HYG US Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_EMB US Equity.html">EMB US Equity (Alpha)</a>||')
            append_copy.write('</div>')
            ##############################
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_LQD US Equity.html">LQD US Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_IHYG LN Equity.html">IHYG LN Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_IEAC LN Equity.html">IEAC LN Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_HYG US Equity.html">HYG US Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_EMB US Equity.html">EMB US Equity (Hedge)</a>||')
            append_copy.write('</div>')
        elif addButtons == 'RiskStatusTRADINGTEAM':
            append_copy.write('<div>')
            append_copy.write("The table displays the probability of the assets under consideration, entering or already experiencing a 'Risk Off / Short' trend. The 'strength' of the signal is the probability level itself, with 50% being the threshold between up and down trends. 'Fairly' strong level to buy hedges would be the 80% and above.")
            append_copy.write('<br><img src="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/Credit_ETFs_RiskOffProbabilities_Daily.jpg" alt="Credit ETFs RiskOff Probs">')
            append_copy.write(
                '<br><a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/ERB_ML_Daily.html">Eurobank ML Risk Off Index Page</a>')
            append_copy.write('</div>')
            ##############################
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_LQD US Equity.html">LQD US Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_IHYG LN Equity.html">IHYG LN Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_IEAC LN Equity.html">IEAC LN Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_HYG US Equity.html">HYG US Equity (Alpha)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Alpha_EMB US Equity.html">EMB US Equity (Alpha)</a>||')
            append_copy.write('</div>')
            ##############################
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_LQD US Equity.html">LQD US Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_IHYG LN Equity.html">IHYG LN Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_IEAC LN Equity.html">IEAC LN Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_HYG US Equity.html">HYG US Equity (Hedge)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/EcoHedge_Hedge_EMB US Equity.html">EMB US Equity (Hedge)</a>||')
            append_copy.write('</div>')
        elif 'DeskPnLSinceInception' in addButtons:
            progress = addButtons.split("_")[1]
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/SDExcelFXPricer_FX_2_Options.html">SD Pricer (FX_2)</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/BOFFICE_GATOS_DailyPnLDF_DERIV.html">BOFFICE PnL</a>')
            append_copy.write('</div>')
            append_copy.write(
                '<div class="progress"> <div class="progress-bar progress-bar-success progress-bar-striped" role="progressbar" aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" style="width:'+str(progress)+'%"> '+str(progress)+'% (of EUR 5M) </div></div>')
        elif addButtons == "QuantStrategies":
            append_copy.write('<br><div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">EMSX Expiries</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/DeskPnLSinceInception.html">Desk PnL</a>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance.html">Endurance</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Coast.html">Coast</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Brotherhood.html">Brotherhood</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Shore.html">Shore</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Valley.html">Valley</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Dragons.html">Dragons</a><br>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/VolatilityScannerPnL.html">VolatilityScanner</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance_Coast_Brotherhood_Valley_Shore_Dragons_TimeStory.html">Total CTA Book</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/TOTAL_SYSTEMATIC.html">TOTAL SYSTEMATIC</a>')
            append_copy.write(
                '<a href="F:\Dealing\Panagiotis Papaioannou\MT5\HTML_Reports\LIVE\PyReports">HFT Strategies</a><br>')
            #append_copy.write('<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/BetaEyeDF_MajorFutures.html">Betas</a>')
            append_copy.write('</div><br>')
        elif addButtons == "SalesOrderBook":
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="http://127.0.0.1/UpdateOrdersButton">Update Orders</a>')
            append_copy.write('</div><br>')
        elif addButtons == "aggregatedFills":
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a><br>')
            append_copy.write('</div>')

            append_copy.write('<div class="topnav">')
            ### STRATEGIES PAGES ###
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/CKATSILEROS1_trader_aggregatedFills.html">C. Katsileros</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/NASIMAKOPOU1_trader_aggregatedFills.html">N. Assimakopoulos</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/SDALIAKOPOUL_trader_aggregatedFills.html">S. Daliakopoulos</a>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">P. Papaioannou</a>')

            append_copy.write('</div>')

        if addPlots is not None:
            append_copy.write('<p>')
            append_copy.write('<img src="'+addPlots+'" alt="'+addPlots+'Img" width="500" height="400">')
            append_copy.write('</p>')

        if specificID == "DeskPnLSinceInception":
            append_copy.write(original_text)
            append_copy.write(
                '<br><div id="chartDiv" style="width:80%; height:650px; margin:0 auto;"></div><br>'
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script>"
                '<script src="https://code.jscharting.com/2.9.0/jscharting.js"></script>'
                '<script src="js/rsPyDeskPnL.js"></script>'
            )
            append_copy.write(
                "<script>  $(function() {$('table td').each(function(){var txt = $(this).text(); if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script><script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()
        elif specificID == "BOFFICE_GATOS_DailyPnLDF_DERIV":
            append_copy.write(original_text)
            append_copy.write(
                '<br><div id="chartDiv" style="width:80%; height:650px; margin:0 auto;"></div><br>'
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script>"
                '<script src="https://code.jscharting.com/2.9.0/jscharting.js"></script>'
                '<script src="js/BOFFICE_GATOS_DailyPnLDF_DERIV.js"></script>'
            )
            append_copy.write(
                "<script>  $(function() {$('table td').each(function(){var txt = $(this).text(); if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script><script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()
        elif specificID == 'BetaEyeDF':
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/Beta_RX1%20Comdty.html"  id="BetaBundTS">(RX1 Comdty) Bund Betas</a>&nbsp &nbsp &nbsp')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/Beta_Endurance.html"  id="BetaEnduranceTS">(Macros) Endurance Betas</a>&nbsp &nbsp &nbsp')
            append_copy.write('<br>')
            append_copy.write(original_text)
            append_copy.write(
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script><script>  $(function() {$('table td').each(function(){var txt = $(this).text(); "
                "if(parseInt(txt) < 0)  $(this).css('color', 'red');  else if(parseInt(txt) > 0) $(this).css('color', '#1193fa');});});</script>")
            append_copy.write(
                "<script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('index')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('color', 'white');}})})}});});</script>")
            append_copy.close()
        elif specificID == 'ML_RiskOff':
            append_copy.write(original_text)
            append_copy.write(
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script><script>  $(function() {$('table td').each(function(){var txt = $(this).text(); "
                "if(parseInt(txt) >= 50 && !txt.includes('-'))  $(this).css('color', 'red'); else if (parseInt(txt) < 50 && !txt.includes('-')) $(this).css('color', 'green');});});</script>")
            append_copy.close()
        else:
            append_copy.write(original_text)
            append_copy.write(
                "<script src='http://code.jquery.com/jquery-1.9.1.js'></script><script>  $(function() {$('table td').each(function(){var txt = $(this).text(); "
                "if(txt.includes('Endurance')) $(this).css('color', '#2fd10f'); else if(txt.includes('Coast')) $(this).css('color', '#21ebd3'); "
                "if(parseInt(txt) < 0 && !txt.includes('M') && !txt.includes('Y'))  $(this).css('color', 'red'); else if (txt.includes('NEED TO ROLL !!!'))  $(this).css('color', 'green'); else if (txt.includes('Expired'))  $(this).css('color', 'white'); else if(parseInt(txt) > 0 && !txt.includes('M') && !txt.includes('Y')) $(this).css('color', '#1193fa');});});</script>")
            append_copy.write(
                "<script>$('table').each (function(){var tableElementID = $(this).closest('table').attr('id'); $('#'+tableElementID+' thead>tr>th').each (function(index){var txt = $(this).text(); if(txt.includes('Asset')||txt.includes('TOTAL')) {var SumCol = index; $('#'+tableElementID+' tr').each(function() { $(this).find('td').each (function(index) {if(index === SumCol) {$(this).css('background', 'black'); $(this).css('color', ' #f2a20d'); $(this).css('border', '2px solid white');}})})}});});</script>")
            append_copy.close()

    "BLOOMBERG TRADING RELATED"

    def getFutureTicker(FxCur):
        if FxCur == "EUR":
            out = "EC1 Curncy"
        elif FxCur == "JPY":
            out = "JY1 Curncy"
        elif FxCur == "GBP":
            out = "BP1 Curncy"
        elif FxCur == "CAD":
            out = "CD1 Curncy"
        elif FxCur == "CHF":
            out = "SF1 Curncy"
        elif FxCur == "AUD":
            out = "AD1 Curncy"
        elif FxCur == "BRL":
            out = "BR1 Curncy"
        elif FxCur == "NZD":
            out = "NV1 Curncy"
        elif FxCur == "RUB":
            out = "RU1 Curncy"
        elif FxCur == "MXN":
            out = "PE1 Curncy"
        elif FxCur == "ZAR":
            out = "RA1 Curncy"
        elif FxCur == "USD":
            out = "USD"
        return out

    def getMoreFuturesCurvePoints(InputPointsIn, FuturesTable, whichCurvePoints):

        outList = list(FuturesTable.loc[InputPointsIn].index)
        for c in whichCurvePoints:
            outList += FuturesTable.loc[InputPointsIn, "Point_"+str(c)].tolist()

        return outList

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
