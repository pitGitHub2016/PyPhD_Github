from datetime import datetime
import pandas as pd, numpy as np, math, sqlite3, time, matplotlib.pyplot as plt, itertools, types, multiprocessing, ta, sqlite3, xlrd, pickle
from statsmodels.regression.rolling import RollingOLS
from tqdm import tqdm
import persim
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import QuantLib as ql
from scipy.stats import norm, t
from scipy.stats import skew, kurtosis, entropy
import statsmodels.tsa.stattools as ts
from scipy.linalg import svd
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn import linear_model
from sklearn.decomposition import PCA
from itertools import combinations, permutations
from ta.volume import *
from optparse import OptionParser
try:
    from ripser import Rips
    from pykalman import pykalman
    from fracdiff2 import frac_diff_ffd
    from hurst import compute_Hc
    from pypfopt.expected_returns import mean_historical_return
    from pypfopt.risk_models import CovarianceShrinkage
    from pypfopt.efficient_frontier import EfficientFrontier
except Exception as e:
    print(e)

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

        if "mode" in kwargs:
            mode = kwargs["mode"]
        else:
            mode = "RollWin"

        if "nShift" in kwargs:
            nShift = kwargs["nShift"]
        else:
            nShift = 2

        if "gapify" in kwargs:
            gapify = kwargs["gapify"]
        else:
            gapify = None

        if mode == "RollWin":
            RollVol = np.sqrt(252) * pyerb.rollStatistics(df, method="Vol", nIn=nIn) * 100
        elif mode == "ExpWin":
            RollVol = np.sqrt(252) * df.expanding(25).std().ffill().bfill() * 100

        if gapify is not None:
            RollVol = pyerb.gapify(RollVol, steps=gapify)

        out = df / pyerb.S(RollVol, nperiods=nShift)
        out = pyerb.fd(out)
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

    def sign(df, **kwargs):
        if "FillNA" in kwargs:
            FillNA = kwargs['FillNA']
        else:
            FillNA = False

        if FillNA:
            df = df.fillna(0)

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
        ################################################################################################
        #tqdm.pandas(desc=None)
        #EXPAND = df.expanding(min_periods=n, center=False).progress_apply(lambda x: func(x))
        return EXPAND

    def expanderMetric(df, metric, n):
        if metric == "Sharpe":
            out = df.expanding(n).mean() / df.expanding(n).std()
        return out

    def DeactivateAssets(df, targetAssets, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = "Neutralise"

        if mode == "Neutralise":
            df.loc[:, targetAssets] = 0

        return df

    def SubSpaceIgnore(df, TargetSubSpace, Condition):
        out = df
        for c in TargetSubSpace:
            if Condition == "IgnorePositives":
                out.loc[out[c] > 0,c] = None
            elif Condition == "IgnoreNegatives":
                out.loc[out[c] < 0,c] = None
        return out

    def SubSpaceIgnoreConnections(df, mode, Condition):
        outMat = pd.DataFrame(0, index=df.columns, columns=df.columns)

        for c1 in tqdm(df.columns):
            for c2 in df.columns:
                if c1 != c2:
                    subDF = pyerb.SubSpaceIgnore(df[[c1, c2]], [c1], Condition).dropna(subset=[c1])
                    try:
                        if mode == "Corr":
                            outMat.loc[c1, c2] = subDF.corr().iloc[0, 1]
                        elif mode == "MI":
                            outMat.loc[c1, c2] = mutual_info_score(subDF[c1].values, subDF[c2].values)

                    except Exception as e:

                        outMat.loc[c1, c2] = None

        return outMat

    "Rollers"

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
                    #print(c, e)
                    H = None

                HurstDF.loc[df.index[-1], c] = H

        return HurstDF

    "Quantitative Finance"

    def Inactivity(df):
        df = df.fillna(0)
        zeroedDF = df[df==0]+1
        InactivityPct = zeroedDF.sum() / df.shape[0]

        return InactivityPct

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

    def max_drawdown(return_series):
        comp_ret = (return_series + 1).cumprod()
        peak = comp_ret.expanding(min_periods=1).max()
        dd = (comp_ret / peak) - 1
        return dd.min()

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

    def AnnSortinoRatio(series, **kwargs):
        if "N" in kwargs:
            N = kwargs['N']
        else:
            N = 252

        if "rf" in kwargs:
            rf = kwargs['rf']
        else:
            rf = 0

        mean = series.mean() * N - rf
        std_neg = series[series < 0].std() * np.sqrt(N)

        return mean / std_neg

    def AnnCalmar(df):
        max_drawdowns = pyerb.max_drawdown(df)
        calmars = df.mean() * 255 / abs(max_drawdowns)
        return calmars

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

    "Smoothers"

    def SelectLookBack(df, **kwargs):

        def getLocalMetric(l_OptimizedMetric, l_OptimizationMode, l_subPnl):

            if l_OptimizedMetric == "Sharpe":
                l_rawMetric = pyerb.sharpe(l_subPnl)
            elif l_OptimizedMetric == "Sortino":
                l_rawMetric = pyerb.AnnSortinoRatio(l_subPnl)
            elif l_OptimizedMetric == "Calmar":
                l_rawMetric = pyerb.AnnCalmar(l_subPnl)
            ###########################################
            if l_OptimizationMode == "Raw":
                l_subMetric = np.sqrt(252) * l_rawMetric
            elif l_OptimizationMode == "Abs":
                l_subMetric = np.sqrt(252) * np.abs(l_rawMetric)

            return [l_subMetric, l_rawMetric]

        if "DynamicsSpace" in kwargs:
            DynamicsSpace = kwargs['DynamicsSpace']
        else:
            DynamicsSpace = "SlowTrendFollowingOnly"

        if "LookBackSelectorMode" in kwargs:
            LookBackSelectorMode = kwargs["LookBackSelectorMode"]
        else:
            LookBackSelectorMode = "SingleLookBack"

        if "TopCalcPct" in kwargs:
            TopCalcPct = kwargs["TopCalcPct"]
        else:
            TopCalcPct = 0.1

        if "TopSelectPct" in kwargs:
            TopSelectPct = kwargs["TopSelectPct"]
        else:
            TopSelectPct = 0.8

        if "stdList" in kwargs:
            stdList = kwargs["stdList"]
        else:
            stdList = [0.5, 1, 1.5, 2, 2.5, 3]

        ##########################################################
        if DynamicsSpace == "SlowTrendFollowingOnly":
            LagList = [10, 25, 50, 125, 250, 375, 500]
            sigShift = 2
        elif DynamicsSpace == "FastTrendFollowingOnly":
                LagList = [10, 25, 50]
                sigShift = 2
        elif DynamicsSpace == "MeanReversionOnlyShift2":
            LagList = [2, 3, 5, 10, 15, 25]
            sigShift = 2
        elif DynamicsSpace == "MeanReversionOnlyShift1":
            LagList = [2, 3, 5, 10, 15, 25]
            sigShift = 1
        elif DynamicsSpace == "CreditTrader":
            LagList = [3, 5, 10, 25, 50, 125, 250]
            sigShift = 1
        elif DynamicsSpace == "SovTrader":
            LagList = [5, 10, 25, 50, 100, 200]
            sigShift = 1
        elif "Range-" in DynamicsSpace:
            DynamicsSpaceSplit = DynamicsSpace.split("-")
            LagList = range(int(DynamicsSpaceSplit[1]),int(DynamicsSpaceSplit[2]), int(DynamicsSpaceSplit[3]))
            sigShift = 2
        ##########################################################
        if "OptimizedMetric" in kwargs:
            OptimizedMetric = kwargs['OptimizedMetric']
        else:
            OptimizedMetric = "Sharpe"
        ##########################################################
        if "OptimizationMode" in kwargs:
            OptimizationMode = kwargs['OptimizationMode']
        else:
            OptimizationMode = "Abs"
        ##########################################################
        if "SortAscending" in kwargs:
            SortAscending = kwargs['SortAscending']
        else:
            SortAscending = False

        if "method" in kwargs:
            method = kwargs['method']
        else:
            method = "EMA"

        columnSet = ["Asset", "Lag", "shiftLag", "Sharpe", 'Direction']

        outList = []
        for c in df.columns:
            if method == "EMA":
                for l in LagList:
                    for s in [sigShift]:#[1,2]:
                        subPnl = pyerb.S(pyerb.sign(pyerb.ema(df[c], nperiods=l)), nperiods=s) * df[c]
                        [subMetric, rawMetric] = getLocalMetric(OptimizedMetric, OptimizationMode, subPnl)
                        outList.append([c,l,s,subMetric,np.sign(rawMetric)])
            elif method == "BB":
                for l in LagList:
                    for no_of_std_In in stdList:
                        for s in [sigShift]:#[1,2]:
                            subPnl = pyerb.S(pyerb.sbb(pd.DataFrame(df[c], columns=[c]), nperiods=int(l), no_of_std=no_of_std_In)[c], nperiods=s) * df[c]
                            [subMetric, rawMetric] = getLocalMetric(OptimizedMetric, OptimizationMode, subPnl)
                            outList.append([c, str(l)+"_"+str(no_of_std_In), s, subMetric, np.sign(rawMetric)])
        ###############################################################################################################
        outDF = pd.DataFrame(outList, columns=columnSet)
        outDF = outDF.sort_values(['Asset', 'Sharpe'], ascending=SortAscending).groupby('Asset').head(round(TopCalcPct * outDF.shape[0]))
        outDF = outDF.set_index("Asset", drop=True)
        ###############################################################################################################
        BestLagList = []
        DirectionList = []
        for c in df.columns:
            #############################################################################
            if LookBackSelectorMode == "SingleLookBack":
                bestLag = outDF.loc[c,"Lag"].values[0]
                direction = outDF.loc[c,"Direction"].values[0]
            elif LookBackSelectorMode == "LookBacksList":
                sub_AssetData = outDF.loc[c,:]
                sub_AssetData["SharpeNormaliser"] = sub_AssetData["Sharpe"]/sub_AssetData["Sharpe"].max()
                sub_AssetData = sub_AssetData[sub_AssetData["SharpeNormaliser"] >= TopSelectPct].round(4)
                sub_AssetData["Lag"] = sub_AssetData["Lag"].astype(str)+":"+sub_AssetData["Sharpe"].astype(str)
                bestLag = ','.join([str(x) for x in sub_AssetData["Lag"].values])
                direction = ','.join([str(x) for x in sub_AssetData["Direction"].values])
            ##############################################################################
            BestLagList.append([c,bestLag])
            DirectionList.append([c,direction])
        BestLagDF = pd.DataFrame(BestLagList, columns=["Asset", "Lag"]).set_index("Asset", drop=True)
        DirectionDF = pd.DataFrame(DirectionList, columns=["Asset", "Direction"]).set_index("Asset", drop=True)
        ###############################################################################################################
        return [BestLagDF, DirectionDF]

    def DynamicSelectLookBack(df0, **kwargs):
        if 'LookBackSelectorMode' in kwargs:
            LookBackSelectorMode = kwargs['LookBackSelectorMode']
        else:
            LookBackSelectorMode = 'SingleLookBack' # SingleLookBack, LookBacksList

        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        if "method" in kwargs:
            method = kwargs['method']
        else:
            method = "EMA"

        if "stdList" in kwargs:
            stdList = kwargs["stdList"]
        else:
            stdList = [0.5, 1, 1.5, 2, 2.5, 3]

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 500+1
        if st < 500:
            print("You need at least 500 observations for the 'SelectLookBack' function to run .. setting st to 500+1")
            st = 500+1
        ##########################################################
        if "DynamicsSpace" in kwargs:
            DynamicsSpace = kwargs['DynamicsSpace']
        else:
            DynamicsSpace = "SlowTrendFollowingOnly"
        ##########################################################
        if "OptimizedMetric" in kwargs:
            OptimizedMetric = kwargs['OptimizedMetric']
        else:
            OptimizedMetric = "Sharpe"
        ##########################################################
        if "OptimizationMode" in kwargs:
            OptimizationMode = kwargs['OptimizationMode']
        else:
            OptimizationMode = "Abs"
        ##########################################################

        LookbacksList = []
        LookbacksDirectionsList = []
        for i in tqdm(range(st, len(df0) + 1)):
            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]
            ##########################################################
            out = pyerb.SelectLookBack(df, method=method, stdList=stdList, DynamicsSpace=DynamicsSpace,
                                       OptimizedMetric=OptimizedMetric, OptimizationMode=OptimizationMode,
                                       LookBackSelectorMode=LookBackSelectorMode)
            ##########################################################
            LookbacksOut = out[0]
            LookbacksDirectionsOut = out[1]
            LookbacksOut.columns = [df.index[-1]]
            LookbacksDirectionsOut.columns = [df.index[-1]]
            LookbacksList.append(LookbacksOut)
            LookbacksDirectionsList.append(LookbacksDirectionsOut)

        ##############################################################################################
        LookbacksDF = pd.concat(LookbacksList,axis=1).T.sort_index()
        LookbacksDF.index.names = df0.index.names
        ##############################################################################################
        LookbacksDirectionsDF = pd.concat(LookbacksDirectionsList,axis=1).T.sort_index()
        LookbacksDirectionsDF.index.names = df0.index.names
        ##############################################################################################
        return [LookbacksDF,LookbacksDirectionsDF]

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

        if "LookBacksDirections" in kwargs:
            LookBacksDirections = kwargs["LookBacksDirections"]
        else:
            LookBacksDirections = pd.DataFrame(1, index=df.index, columns=df.columns)
        "Be sure to have the LookbackDirections 'on' ONLY for the rolling mean case"
        if mode not in ["mean"]:
            LookBacksDirections = pd.DataFrame(1, index=df.index, columns=df.columns)

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

        out = demaDF * LookBacksDirections

        return out

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

    "Gekko Custom Technicals"

    def hmp(df, **kwargs):
        if 'channelMethod' in kwargs:
            channelMethod = kwargs['channelMethod']
        else:
            channelMethod = "BB"

        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3

        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2

        outlist = []
        for c in df.columns:
            if channelMethod == "BB":
                ch = pyerb.bb(df[c], nperiods=nperiods, no_of_std=no_of_std)
                ch["InBands"] = 0
                ch.loc[(ch["Price"] < ch["UPPER"])&(ch["Price"] >= ch["LOWER"]),"InBands"] = 1
                ch["pctPointsOut"] = 0
                for i in tqdm(range(nperiods, len(ch) + 1)):
                    subCh = ch.iloc[i - nperiods:i, :]
                    ch.loc[subCh.index[-1],"pctPointsOut"] = 1-subCh['InBands'].mean()
            ################################################################################
            subDF = ch["pctPointsOut"]
            subDF.name = c
            outlist.append(subDF)

        outDF = pd.concat(outlist,axis=1).sort_index()

        return outDF

    "Signals"

    def sbb(df, **kwargs):

        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3

        if 'no_of_std' in kwargs:
            no_of_std = kwargs['no_of_std']
        else:
            no_of_std = 2

        signalList = []
        for c in df.columns:
            if c != 'Date':
                cBB = pyerb.bb(df[c], nperiods=nperiods, no_of_std=no_of_std)
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
        if 'ConnectionMode' in kwargs:
            ConnectionMode = kwargs['ConnectionMode']
        else:
            ConnectionMode = "-"
        if ConnectionMode == "+":
            ConnectionSign = 1
        else:
            ConnectionSign = -1
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
            elif mode == 'HedgeRatio_Rolling':
                df0 = pd.concat([df[c[0]].rolling(n).corr(df[c[1]]) * (pyerb.roller(df[c[0]], np.std, n) / pyerb.roller(df[c[1]], np.std, n)) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatio_Expanding':
                df0 = pd.concat([df[c[0]].expanding(n).corr(df[c[1]]) * (pyerb.expander(df[c[0]], np.std, n) / pyerb.expander(df[c[1]], np.std, n)) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioPnL_Roll':
                df0 = pd.concat([(df[c[0]] + ConnectionSign * pyerb.S(df[c[0]].rolling(n).corr(df[c[1]]) * (
                            pyerb.roller(df[c[0]], np.std, n) / pyerb.roller(df[c[1]], np.std, n)), nperiods=1)
                                             * df[c[1]]).fillna(0) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioPnL_Expand':
                df0 = pd.concat([(df[c[0]] + ConnectionSign * pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]) * (
                            pyerb.expander(df[c[0]], np.std, n) / pyerb.expander(df[c[1]], np.std, n)), nperiods=1)
                                             * df[c[1]]).fillna(0) for c in tqdm(cc)], axis=1, keys=cc)
            elif mode == 'HedgeRatioSimpleCorr':
                df0 = pd.concat([df[c[0]] + ConnectionSign * (pyerb.S(df[c[0]].expanding(n).corr(df[c[1]]), nperiods=1) * df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
            df0.columns = df0.columns.map('_'.join)
        else:
            if mode == 'OLSpnl':
                df0 = pd.concat([pyerb.rs(pyerb.S(pyerb.rollOLS(df, selList=list(c), Y=c[0], X=list(c[1:]), n=n)) * df[list(c[1:])]).rename('_'.join(list(c))) for c in tqdm(cc)], axis=1)

        return df0.fillna(0).sort_index()

    def RVSignalHandler(sigDF, **kwargs):

        if 'HedgeRatioMul' in kwargs:
            HedgeRatioMul = kwargs['HedgeRatioMul']
        else:
            HedgeRatioMul = 1

        if 'HedgeRatioDF' in kwargs:
            HedgeRatioDF = kwargs['HedgeRatioDF']
        else:
            HedgeRatioDF = pd.DataFrame(HedgeRatioMul, index=sigDF.index, columns=sigDF.columns)

        assetSignList = []
        for c in sigDF.columns:
            medSigDF = pd.DataFrame(sigDF[c])
            assetNames = c.split("_")
            medSigDF[assetNames[0]] = sigDF[c]
            medSigDF[assetNames[1]] = sigDF[c] * HedgeRatioDF[c]
            subSigDF = medSigDF[[assetNames[0], assetNames[1]]]
            #print(subSigDF)
            assetSignList.append(subSigDF)
        assetSignDF = pd.concat(assetSignList, axis=1)
        assetSignDFgroupped = assetSignDF.groupby(assetSignDF.columns, axis=1).sum()

        return assetSignDFgroupped

    "Pricing"
    "Custom Pricers"
    def black_scholes(S, K, T, rf, sigma, option_type):
        "https://carlolepelaars.github.io/blackscholes/4.the_greeks_black76/#parameters"
        d1 = (np.log(S / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == "call":
            option_price = math.exp(-rf * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
        elif option_type == "put":
            option_price = math.exp(-rf * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        return option_price

    def black_scholes_greeks(S, K, T, rf, sigma, option_type):
        "https://carlolepelaars.github.io/blackscholes/4.the_greeks_black76/#parameters"
        d1 = (np.log(S / K) + (sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        #Phi_d1 = norm.cdf(d1)
        #phi_d1 = norm.pdf(d1)

        if option_type == "call":
            delta = np.exp(-rf * T) * norm.cdf(d1)
            gamma = np.exp(-rf * T) * (norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            theta = np.exp(-rf * T) * ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - (rf * K * norm.cdf(d2)) + (rf * S * norm.cdf(d1)))
            vega = S * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T)
            rho = -T * np.exp(-rf * T) * (S * norm.cdf(d1) - K * norm.cdf(d2))
        elif option_type == "put":
            delta = -np.exp(-rf * T) * norm.cdf(-d1)
            gamma = np.exp(-rf * T) * (norm.pdf(d1) / (S * sigma * np.sqrt(T)))
            theta = np.exp(-rf * T) * ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) + (rf * K * norm.cdf(-d2)) - (rf * S * norm.cdf(-d1)))
            vega = S * np.exp(-rf * T) * norm.pdf(d1) * np.sqrt(T)
            rho = -T * np.exp(-rf * T) * (K * norm.cdf(-d2) - S * norm.cdf(-d1))

        return delta, gamma / 100, theta / 365, vega / 100, rho / 100

    "QuantLib Stuff"
    def quantLib_DateConverter(dateInput):
        ####################################################
        if dateInput.month == 1:
            qlmonth = ql.January
        elif dateInput.month == 2:
            qlmonth = ql.February
        elif dateInput.month == 3:
            qlmonth = ql.March
        elif dateInput.month == 4:
            qlmonth = ql.April
        elif dateInput.month == 5:
            qlmonth = ql.May
        elif dateInput.month == 6:
            qlmonth = ql.June
        elif dateInput.month == 7:
            qlmonth = ql.July
        elif dateInput.month == 8:
            qlmonth = ql.August
        elif dateInput.month == 9:
            qlmonth = ql.September
        elif dateInput.month == 10:
            qlmonth = ql.October
        elif dateInput.month == 11:
            qlmonth = ql.November
        elif dateInput.month == 12:
            qlmonth = ql.December
        ####################################################
        return ql.Date(dateInput.day, qlmonth, dateInput.year)

    def Barrier(specs):

        def binomial_barrier_option():
            payoff = ql.CashOrNothingPayoff(option_type, strike, payoff_amt)
            exercise = ql.EuropeanExercise(expiry_dt)
            option = ql.BarrierOption(barrier_type, barrier, 0.0, payoff, exercise)
            process = ql.GarmanKohlagenProcess(
                ql.QuoteHandle(spot_quote),
                ql.YieldTermStructureHandle(foreignTS),
                ql.YieldTermStructureHandle(domesticTS),
                ql.BlackVolTermStructureHandle(expanded_volTS),
            )
            engine = ql.BinomialBarrierEngine(process, "crr", 200)
            option.setPricingEngine(engine)
            return option

        def vanna_volga_barrer_option():
            payoff = ql.CashOrNothingPayoff(option_type, strike, payoff_amt)
            exercise = ql.EuropeanExercise(expiry_dt)
            option = ql.BarrierOption(barrier_type, barrier, 0.0, payoff, exercise)
            engine = ql.VannaVolgaBarrierEngine(
                ql.DeltaVolQuoteHandle(atmVol),
                ql.DeltaVolQuoteHandle(vol25Put),
                ql.DeltaVolQuoteHandle(vol25Call),
                ql.QuoteHandle(spot_quote),
                ql.YieldTermStructureHandle(domesticTS),
                ql.YieldTermStructureHandle(foreignTS),
            )
            option.setPricingEngine(engine)
            return option

        #option_ID = specs['option_ID']#"OVML EURUSD DIKO 1.0000P B0.9500 01/13/23 N1M"

        today = pyerb.quantLib_DateConverter(specs['today'])#ql.Date(7, ql.December, 2023)
        ql.Settings.instance().evaluationDate = today

        "option specification"
        #underlying = specs['underlying']#"EURUSD"
        ############################################################################
        if specs['option_type'] == "call":
            option_type = ql.Option.Call
        elif specs['option_type'] == "put":
            option_type = ql.Option.Put
        ############################################################################
        strike = specs['strike']#1.0779
        if specs['barrier_type'] == "DownOut":
            barrier_type = ql.Barrier.DownOut
        elif specs['barrier_type'] == "DownIn":
            barrier_type = ql.Barrier.DownIn
        elif specs['barrier_type'] == "UpOut":
            barrier_type = ql.Barrier.UpOut
        elif specs['barrier_type'] == "UpIn":
            barrier_type = ql.Barrier.UpIn
        ############################################################################
        barrier = specs['barrier']#1.0237
        payoff_amt = specs['payoff_amt']#1000000.0
        expiry_dt = pyerb.quantLib_DateConverter(specs['expiry_dt'])#ql.Date(14, 12, 2023)
        #trade_dt = ql.Date(7, 12, 2023)
        #settle_dt = ql.Date(9, 12, 2023)
        #delivery_dt = ql.Date(18, 12, 2023)
        ############################################################################
        "market data"
        spot = specs['spot'] #1.0776
        vol_atm = specs['vol_atm']  # 9.528
        vol_rr = specs['vol_rr']  # -0.31
        vol_bf = specs['vol_bf']  # 0.08
        vol_25d_put = vol_bf - vol_rr / 2 + vol_atm
        vol_25d_call = vol_rr / 2 + vol_bf + vol_atm
        rd = specs['rd']#3.950, eur_depo
        rf = specs['rf']#5.335. usd_depo
        ############################################################################
        "simple quotes"
        spot_quote = ql.SimpleQuote(spot)
        vol_atm_quote = ql.SimpleQuote(vol_atm / 100)
        vol_25d_put_quote = ql.SimpleQuote(vol_25d_put / 100)
        vol_25d_call_quote = ql.SimpleQuote(vol_25d_call / 100)
        rd_quote = ql.SimpleQuote(rd / 100)
        rf_quote = ql.SimpleQuote(rf / 100)
        ############################################################################
        "delta quotes"
        atmVol = ql.DeltaVolQuote(
            ql.QuoteHandle(vol_atm_quote),
            ql.DeltaVolQuote.Fwd,
            3.0,
            ql.DeltaVolQuote.AtmFwd,
        )
        vol25Put = ql.DeltaVolQuote(
            -0.25, ql.QuoteHandle(vol_25d_put_quote), 3.0, ql.DeltaVolQuote.Fwd
        )
        vol25Call = ql.DeltaVolQuote(
            0.25, ql.QuoteHandle(vol_25d_call_quote), 3.0, ql.DeltaVolQuote.Fwd
        )
        ############################################################################
        "term structures"
        domesticTS = ql.FlatForward(
            0, ql.UnitedStates(), ql.QuoteHandle(rd_quote), ql.Actual360()
        )
        foreignTS = ql.FlatForward(
            0, ql.UnitedStates(), ql.QuoteHandle(rf_quote), ql.Actual360()
        )
        volTS = ql.BlackConstantVol(
            0, ql.UnitedStates(), ql.QuoteHandle(vol_atm_quote), ql.ActualActual()
        )
        expanded_volTS = ql.BlackConstantVol(
            0, ql.UnitedStates(), ql.QuoteHandle(vol_atm_quote), ql.ActualActual()
        )
        ############################################################################
        try:
            option = binomial_barrier_option()
            optionPricerData = {"Price": option.NPV() / spot}
        except Exception as e:
            print(e)
            option = vanna_volga_barrer_option()
            optionPricerData = {"Price":option.NPV()/spot}
        ############################################################################
        "https://quant.stackexchange.com/questions/70258/quantlib-greeks-of-fx-option-in-python"
        ############################################################################
        try:
            optionPricerData["gamma"] = option.gamma()*spot/100
        except Exception as e:
            pass
            optionPricerData["gamma"] = np.nan
        try:
            optionPricerData["vega"] = option.vega()*(1/100)/spot
        except Exception as e:
            pass
            optionPricerData["vega"] = np.nan
        try:
            optionPricerData["theta"] = option.theta()*(1/365)/spot
        except Exception as e:
            pass
            optionPricerData["theta"] = np.nan
        try:
            optionPricerData["delta"] = option.delta()
        except Exception as e:
            pass
            optionPricerData["delta"] = np.nan
        ############################################################################
        return optionPricerData

    "TDA"

    def compute_wasserstein_distances(log_returns, window_size, rips):
        """Compute the Wasserstein distances."""
        n = len(log_returns) - (2 * window_size) + 1
        distances = np.full((n, 1), np.nan)  # Using np.full with NaN values

        for i in range(n):
            segment1 = log_returns[i:i + window_size].reshape(-1, 1)
            segment2 = log_returns[i + window_size:i + (2 * window_size)].reshape(-1, 1)

            if segment1.shape[0] != window_size or segment2.shape[0] != window_size:
                continue

            dgm1 = rips.fit_transform(segment1)
            dgm2 = rips.fit_transform(segment2)
            distance = persim.wasserstein(dgm1[0], dgm2[0], matching=False)
            distances[i] = distance
            #print(distance)

        return distances

    "Metric Build"

    def Metric(metaDF, **kwargs):
        if "metric" in kwargs:
            metric = kwargs['metric']
        else:
            metric = "euclidean"
        if "SubSpaceIgnoreConnections" in metric:
            metricSplit = metric.split("_")
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
                    elif "SubSpaceIgnoreConnections" in metric:
                        if c1 != c2:
                            subDF = pyerb.SubSpaceIgnore(metaDF[[c1, c2]], [c1], metricSplit[1]).dropna(subset=[c1])
                            try:
                                if metricSplit[2] == "Corr":
                                    MetricMat.loc[c1, c2] = subDF.corr().iloc[0, 1]
                                elif metricSplit[2] == "MI":
                                    MetricMat.loc[c1, c2] = mutual_info_score(subDF[c1].values, subDF[c2].values)
                            except Exception as e:
                                MetricMat.loc[c1, c2] = None
                        else:
                            MetricMat.loc[c1, c2] = 0

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
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/IRS_Risk_Reporting_Management/ALL_IRS_Risk.html">IRS Risk Reporting</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel_ActivePositions.html">Quant Strategies (Hermes)</a><br>')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">EMSX Expiries</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/DeskPnLSinceInception.html">Desk PnL</a><br>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance.html">Endurance</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Coast.html">Coast</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Brotherhood.html">Brotherhood</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/ShoreDM.html">ShoreDM</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/ShoreEM.html">ShoreEM</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Valley.html">Valley</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Dragons.html">Dragons</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Lumen.html">Lumen</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Fidei.html">Fidei</a><br>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/Endurance_Coast_Brotherhood_ShoreDM_ShoreEM_Valley_Dragons_Lumen_Fidei_TimeStory.html">Total CTA Book</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/VolatilityScannerPnL.html">VolatilityScanner</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/IV_HV_Spreads_Plots/IV_HV_Plotter.html">IV HV Plotter</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/TOTAL_SYSTEMATIC.html">TOTAL SYSTEMATIC</a><br>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/HFT+ShoreDM.html">HFT+ShoreDM</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/MT5/HTML_Reports/LIVE/PyReports/Currently_Running_System_ERB_Py_Reporter_TOTAL_GATHERER_VOLSCANNER.html">HFT+VolScanner</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/StrategiesFactSheets/TOTAL_HFT_VolScanner_ShoreDM.html">HFT+VolScanner+ShoreDM</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/MT5/HTML_Reports/LIVE/PyReports/Currently_Running_System_ERB_Py_Reporter_TOTAL.html">HFT TOTAL</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/MT5/HTML_Reports/LIVE/PyReports/Currently_Running_System_ERB_Py_Reporter_YTD_TOTAL.html">HFT TOTAL YTD</a><br>')
            append_copy.write(
                '<a href="F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\StrategiesFactSheets\Galileo_Total.html">Galileo</a>||')
            append_copy.write(
                '<a href="F:\Dealing\Panagiotis Papaioannou\MT5\HTML_Reports\LIVE\PyReports">HFT Strategies</a><br>')
            #append_copy.write('<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/BetaEyeDF_MajorFutures.html">Betas</a>')
            append_copy.write('</div><br>')
        elif addButtons == "QuantStrategiesHermes":
            append_copy.write('<br><div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel_ActivePositions.html">Quant Strategies (Hermes)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">EMSX Expiries</a>')
            append_copy.write('</div>')
            append_copy.write('<div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/TRADING/Govies/QIS/LiveQuantSystems/SOVTraderFactSheets">SOV Trader Factsheets</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/TRADING/High Yield Desk/Credit_Trader_Python/CreditTraderFactSheets">Credit Trader Factsheets</a>')
            append_copy.write('</div><br>')
        elif addButtons == "QIS_Reporter":
            append_copy.write('<br><div class="topnav">')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/GreenBoxHome.html">GreenBox Home</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel.html">Quant Strategies</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/GreenBox/QuantitativeStrategies_ControlPanel_ActivePositions.html">Quant Strategies (Hermes)</a>||')
            append_copy.write(
                '<a href="file:///F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/PPAPAIOANNO1_trader_aggregatedFills.html">EMSX Expiries</a>')
            append_copy.write('</div>')
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
            append_copy.write('<img src="'+addPlots['src']+'" alt="'+addPlots['name']+'Img" width="'+str(addPlots['width'])+'" height="'+str(addPlots['height'])+'">')
            append_copy.write('</p>')

        if specificID == "DeskPnLSinceInception_with_JS_Charts":
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

    "Other operators (DB Handlers, File readers, Chunks etc.)"

    def updateDF(OldDF, NewDF):
        TotalNewDF = pd.concat([OldDF, NewDF], axis=0)
        TotalNewDF = TotalNewDF[~TotalNewDF.index.duplicated(keep='last')]
        TotalNewDF.index.names = ['date']
        return TotalNewDF

    def updateExcelDF(targetFile, NewDF):
        ##################################################################################
        OldDF = pd.read_excel(targetFile)
        OldDF = OldDF.set_index(OldDF.columns[0], drop=True)
        OldDF.index.names = ['date']
        ##################################################################################
        TotalNewDF = pd.concat([OldDF, NewDF], axis=0)
        TotalNewDF = TotalNewDF[~TotalNewDF.index.duplicated(keep='last')]
        TotalNewDF.index.names = ['date']
        return TotalNewDF

    def updateTableDF(tableName, NewDF, TargetDBconn):
        ##################################################################################
        OldDF = pd.read_sql('SELECT * FROM ' + tableName, TargetDBconn)
        OldDF = OldDF.set_index(OldDF.columns[0], drop=True)
        OldDF.index.names = ['date']
        OldDF.to_sql(tableName + "_BeforeUpdate", TargetDBconn, if_exists='replace')
        ##################################################################################
        TotalNewDF = pd.concat([OldDF, NewDF], axis=0)
        TotalNewDF = TotalNewDF[~TotalNewDF.index.duplicated(keep='last')]
        TotalNewDF.index.names = ['date']
        return TotalNewDF

    def updatePickleDF(NewDF, TargetPickleFile):
        ##################################################################################
        OldPickle = open(TargetPickleFile,'rb')
        OldDF = pickle.load(OldPickle)
        OldPickle.close()
        OldDF.index.names = ['date']
        ##################################################################################
        TotalNewDF = pd.concat([OldDF, NewDF], axis=0)
        TotalNewDF = TotalNewDF[~TotalNewDF.index.duplicated(keep='last')]
        TotalNewDF.index.names = ['date']

        return TotalNewDF

    def readPickleDF(TargetPickleFile):
        tempPickle = open(TargetPickleFile, 'rb')
        tempPickleDF = pickle.load(tempPickle)
        tempPickle.close()
        return tempPickleDF

    def savePickleDF(saveData, TargetPickleFile):
        tempPickle = open(TargetPickleFile, 'wb')
        pickle.dump(saveData, tempPickle)
        tempPickle.close()
        return 'done'

    def chunkMaker(df, axisDirection, chunkSize, **kwargs):

        chunksList = []
        if axisDirection == 0:
            for i in range(0, df.shape[0], chunkSize):
                if i+chunkSize <= df.shape[0]:
                    i_end = i+chunkSize
                else:
                    i_end = df.shape[0]
                subDF = df.iloc[i:i_end,:]
                chunksList.append(subDF)
        elif axisDirection == 1:
            for i in range(0, df.shape[1], chunkSize):
                if i+chunkSize <= df.shape[1]:
                    i_end = i+chunkSize
                else:
                    i_end = df.shape[1]
                subDF = df.iloc[:, i:i_end]
                chunksList.append(subDF)

        if "ReturnIntervals" in kwargs:
            IntervalsList = []
            for elem in chunksList:
                IntervalsList.append([df.columns.get_loc(elem.columns[0]), df.columns.get_loc(elem.columns[-1])])
            return [chunksList, IntervalsList]
        else:
            return chunksList

    def getIndexes(dfObj, value):

        # Empty list
        listOfPos = []

        # isin() method will return a dataframe with
        # boolean values, True at the positions
        # where element exists
        result = dfObj.isin([value])

        # any() method will return
        # a boolean series
        seriesObj = result.any()

        # Get list of column names where
        # element exists
        columnNames = list(seriesObj[seriesObj == True].index)

        # Iterate over the list of columns and
        # extract the row index where element exists
        for col in columnNames:
            rows = list(result[col][result[col] == True].index)

            for row in rows:
                listOfPos.append((row, col))

        # This list contains a list tuples with
        # the index of element in the dataframe
        return listOfPos

    def top_bottom_columns(df, **kwargs):

        if "N_Features" in kwargs:
            N_Features = kwargs['N_Features']
        else:
            N_Features = 5

        if "Normaliser" in kwargs:
            Normaliser = kwargs['Normaliser']
        else:
            Normaliser = "Raw"
            #Normaliser = "RowStoch"

        if "Scalers" in kwargs:
            Scalers = kwargs['Scalers']
        else:
            Scalers = [np.sqrt(252),np.sqrt(252)]

        if Normaliser == "RowStoch":
            df = pyerb.rowStoch(df)

        MainIndexName = df.index.name
        # Create a DataFrame to store the results
        result_df = pd.DataFrame()

        # Iterate through each row in the input DataFrame
        for index, row in df.iterrows():
            # Sort the row values and keep track of the original column names
            sorted_values = row.sort_values(ascending=False).dropna()
            if Normaliser == "RowStoch":
                sorted_values *= 100
            top_columns = sorted_values[-N_Features:].index.tolist()  # Top 5 column names
            bottom_columns = sorted_values[:N_Features].index.tolist()  # Bottom 5 column names
            ####################################################################################
            top_values = list(sorted_values[-N_Features:].values * Scalers[0])  # Top 5 column names
            bottom_values = list(sorted_values[:N_Features].values * Scalers[1])  # Bottom 5 column names
            ####################################################################################
            # Create a dictionary with the results for this row
            row_result = {
                MainIndexName : index,
                'Top_'+str(N_Features)+'_Columns': top_columns,
                'Bottom_'+str(N_Features)+'_Columns': bottom_columns,
                'Top_'+str(N_Features)+'_Values': top_values,
                'Bottom_'+str(N_Features)+'_Values': bottom_values
            }
            ####################################################################################
            # Append the row result to the result DataFrame
            result_df = result_df.append(row_result, ignore_index=True)

        result_df = result_df.set_index(MainIndexName, drop=True)

        return result_df

    def read_date(date):
        return xlrd.xldate.xldate_as_datetime(date, 0)

    def chunkCSVReader(name):
        df = pd.read_csv(name, delimiter=';', chunksize=10000)
        return df

    def getAllTablesDB(sqliteConnection):

        sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
        cursor = sqliteConnection.cursor()
        cursor.execute(sql_query)
        return cursor.fetchall()

    def stringReplace(df, stringList, replaceWith):

        out = df.copy().astype(str)
        for c in df.columns:
            try:
                out[c] = out[c].str.replace("|".join(stringList), replaceWith, regex=True)
            except:
                pass

        return out

    "BLOOMBERG TRADING RELATED"

    def EMSX_Kondor_Dict(k):

        if k == "CME-ED":
            return "ED1 Comdty"
        elif k == "ED1 Comdty":
            return "CME-ED"
        ##############################################
        elif k == 'F_3MEURIBOR':
            return 'ER2 Comdty'
        elif k == 'ER2 Comdty':
            return 'F_3MEURIBOR'
        ##############################################
        elif k == "F_EUR":
            return "EC1 Curncy"
        elif k == "EC1 Curncy":
            return "F_EUR"
        ##############################################
        elif k == "F_GBP":
            return "BP1 Curncy"
        elif k == "BP1 Curncy":
            return "F_GBP"
        ##############################################
        elif k == "F_CAD":
            return "CD1 Curncy"
        elif k == "CD1 Curncy":
            return "F_CAD"
        ##############################################
        elif k == "F_NZD":
            return "NV1 Curncy"
        elif k == "NV1 Curncy":
            return "F_NZD"
        ##############################################
        elif k == "F_JPY":
            return "JY1 Curncy"
        elif k == "JY1 Curncy":
            return "F_JPY"
        ##############################################
        elif k == "F_AUD":
            return "AD1 Curncy"
        elif k == "AD1 Curncy":
            return "F_AUD"
        ##############################################
        elif k == "F_CHF":
            return 'SF1 Curncy'
        elif k == 'SF1 Curncy':
            return "F_CHF"
        ##############################################
        elif k == "DXY_FUTURE":
            return 'DX1 Curncy'
        elif k == 'DX1 Curncy':
            return "DXY_FUTURE"
        ##############################################
        elif k in ["FUT_BRL", "F_BRL"]:
            return "BR1 Curncy"
        elif k == "BR1 Curncy":
            return "F_BRL"
        ##############################################
        elif k == "F_MXN":
            return "PE1 Curncy"
        elif k == "PE1 Curncy":
            return "F_MXN"
        ##############################################
        elif k == "RUB_USD_FUT":
            return "RU1 Curncy"
        elif k == "RU1 Curncy":
            return "RUB_USD_FUT"
        ##############################################
        elif k == "F_ZAR":
            return "RA1 Curncy"
        elif k == "RA1 Curncy":
            return "F_ZAR"
        ##############################################
        elif k == "E-MINI_FUTUR":
            return "ES1 Index"
        elif k == "ES1 Index":
            return "E-MINI_FUTUR"
        ##############################################
        elif k == "YM":
            return "DM1 Index"
        elif k == "DM1 Index":
            return "YM"
        ##############################################
        elif k == "CAC40_FUTURE":
            return 'CF1 Index'
        elif k == 'CF1 Index':
            return "CAC40_FUTURE"
        ##############################################
        elif k == "F_EURSTOXX50":
            return 'VG1 Index'
        elif k == 'VG1 Index':
            return "F_EURSTOXX50"
        ##############################################
        elif k == "FDAX_FUTURE":
            return 'GX1 Index'
        elif k == 'GX1 Index':
            return "FDAX_FUTURE"
        ##############################################
        elif k == "ME":
            return 'FA1 Index'
        elif k == 'FA1 Index':
            return "ME"
        ##############################################
        elif k == "OMXH25_FUT":
            return 'OT1 Index'
        elif k == 'OT1 Index':
            return "OMXH25_FUT"
       ###############################################
        elif k == "NAS_100_MINI":
            return 'NQ1 Index'
        elif k == 'NQ1 Index':
            return "NAS_100_MINI"
        ##############################################
        elif k == "F_SMI":
            return "SM1 Index"
        elif k == "SM1 Index":
            return "F_SMI"
        ##############################################
        elif k == "LIF-FTSE100":
            return "Z 1 Index"
        elif k == "Z 1 Index":
            return "LIF-FTSE100"
        ##############################################
        elif k == "F_2Y_T_NOTE":
            return "TU1 Comdty"
        elif k == "TU1 Comdty":
            return "F_2Y_T_NOTE"
        ##############################################
        elif k == "F_TY_T_NOTE":
            return "TY1 Comdty"
        elif k == "TY1 Comdty":
            return "F_TY_T_NOTE"
        ##############################################
        elif k == "F_5Y_T_NOTE":
            return "FV1 Comdty"
        elif k == "FV1 Comdty":
            return "F_5Y_T_NOTE"
        ##############################################
        elif k == "EUREXSCHATZ":
            return 'DU1 Comdty'
        elif k == 'DU1 Comdty':
            return "EUREXSCHATZ"
        ##############################################
        elif k == "EUREXBOBL":
            return 'OE1 Comdty'
        elif k == 'OE1 Comdty':
            return "EUREXBOBL"
        ##############################################
        elif k == "EUREXBUND":
            return 'RX1 Comdty'
        elif k == 'RX1 Comdty':
            return "EUREXBUND"
        ##############################################
        elif k == "EUREXBUXL":
            return 'UB1 Comdty'
        elif k == 'UB1 Comdty':
            return "EUREXBUXL"
        ##############################################
        elif k == "EUREXFOAT":
            return 'OAT1 Comdty'
        elif k == 'OAT1 Comdty':
            return "EUREXFOAT"
        ##############################################
        elif k == "F_30Y_ULTRA":
            return 'WN1 Comdty'
        elif k == 'WN1 Comdty':
            return "F_30Y_ULTRA"
        ##############################################
        elif k == "LONG_GILT":
            return 'G 1 Comdty'
        elif k == 'G 1 Comdty':
            return "LONG_GILT"
        ##############################################
        elif k == "----------------------":
            return 'FF1 Comdty'
        ##############################################
        elif k == "----------------------":
            return 'FVS1 Index'
        ##############################################
        elif k == "VIX_FUTURE":
            return 'UX1 Index'
        elif k == 'UX1 Index':
            return "VIX_FUTURE"

    def TimeOverride(HFT_Data_File, DayToOverride, WhichHourToUse, OutLabel):
        HFT_df = pd.read_csv(HFT_Data_File, delimiter='\t')
        HFT_df["<DATE>"] = pd.to_datetime(HFT_df["<DATE>"])
        HFT_df["DayNumber"] = HFT_df["<DATE>"].dt.weekday.astype(int)
        HFT_df["HH"] = HFT_df["<TIME>"].str.split(":").str[0].astype(int)
        HFT_df["MM"] = HFT_df["<TIME>"].str.split(":").str[1].astype(int)
        HFT_df["SS"] = HFT_df["<TIME>"].str.split(":").str[2].astype(int)

        HFT_df_SubSace = HFT_df[(HFT_df['DayNumber'] == DayToOverride) & (HFT_df["HH"] == WhichHourToUse) & (HFT_df["MM"] == 0) & (HFT_df["SS"] == 0)].set_index("<DATE>",drop=True)["<CLOSE>"]
        HFT_df_SubSace.name = OutLabel

        return HFT_df_SubSace

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

    def PointsBelongToSameCurve(pair, **kwargs):
        if "Delimiter" in kwargs:
            Delimiter = kwargs['Delimiter']
        else:
            Delimiter = "_"

        pairSplit = pair.split(Delimiter)
        FirstFut = pairSplit[0].split(" ")[0]
        SecondFut = pairSplit[1].split(" ")[0]

        FirstFutBase = FirstFut.replace(FirstFut[-1], "")
        SecondFutBase = SecondFut.replace(SecondFut[-1], "")

        if FirstFutBase == SecondFutBase:
            return True
        else:
            return False

    def IndicatorsHandler(IndicatorsDF,ControllerDF,**kwargs):
        if "SelectorHandler" in kwargs:
            SelectorHandler = kwargs['SelectorHandler']
        else:
            SelectorHandler = [[0, "exclude"],[2, "diff"]]

        out = IndicatorsDF
        sel = [0, "exclude"]

        for sel in SelectorHandler:
            if sel[1] == 'exclude':
                out = out[ControllerDF[ControllerDF["Selector"] != sel[0]]["Indicator"].dropna().tolist()]
            if sel[1] in ['diff']:
                out[ControllerDF.loc[ControllerDF["Selector"] == sel[0], "Indicator"].tolist()] = pyerb.d(out[ControllerDF.loc[ControllerDF["Selector"] == sel[0], "Indicator"].tolist()])

        return out

    def getUnderlyingFromDerivative(DerTicker):
        if DerTicker == "SPX Index":
            return "ES1 Index"
        elif DerTicker == "NDX Index":
            return "NQ1 Index"

# SubClasses
class MainStream:

    def __init__(self, ID):
        self.ID = ID

    def PortfolioBuilder(self, df0, **kwargs):

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 25

        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'RollWindow'

        if RollMode == "ExpWindow":
            st = 25

        EmbeddingPackList = []
        for i in tqdm(range(st, len(df0) + 1)):
            try:

                # print("Step:", i, " of ", len(df0) + 1)
                if RollMode == 'RollWindow':
                    df = df0.iloc[i - st:i, :]
                else:
                    df = df0.iloc[0:i, :]

                latestIndex = df.index[-1]

                ef = EfficientFrontier(mean_historical_return(df, log_returns=True),
                                       CovarianceShrinkage(df, log_returns=True).ledoit_wolf(),
                                       weight_bounds=(-1, 1))
                weights = ef.max_sharpe()
                print(weights)
                time.sleep(30000)

            except Exception as e:
                print(e)
