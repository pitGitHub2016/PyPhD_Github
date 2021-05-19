import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
import math, numpy.linalg as la, sqlite3
from pydiffmap import kernel
import time, pickle
import matplotlib as mpl
from math import acos
from math import sqrt
from math import pi
from numpy.linalg import norm
from scipy import stats
from numpy import dot, array
from itertools import combinations
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, minmax_scale, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, \
    QuantileTransformer, PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn import linear_model
import pymc3 as pm
import arviz as az
from statsmodels.tsa.arima_model import ARIMA
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import theano.tensor as tt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis, norm
from scipy import stats as st
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
import logging
logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logger.propagate = False
warnings.filterwarnings('ignore')

class Slider:

    # Operators

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

        out = Slider.d(np.log(df), nperiods=nperiods)

        if fillna == "yes":
            out = out.fillna(0)
        return out

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
            out = Slider.d(np.log(df), nperiods=nperiods)
        elif calcMethod == 'Discrete':
            out = df.pct_change(nperiods)
        if calcMethod == 'Linear':
            diffDF = Slider.d(df, nperiods=nperiods)
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
        out = np.log(Slider.E(np.exp(df)))
        return out

    def cs(df):
        out = df.cumsum()
        return out

    def ecs(df):
        out = np.exp(df.cumsum())
        return out

    def pb(df):
        out = np.log(Slider.rs(np.exp(df)))
        return out

    def sign(df):
        # df[df > 0] = 1
        # df[df < 0] = -1
        # df[df == 0] = 0
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
        out = df.replace([np.inf, -np.inf], 0)
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

    def rowStoch(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'classic'
        if mode == 'classic':
            out = df.div(df.sum(axis=1), axis=0)
        elif mode == 'abs':
            out = df.div(df.abs().sum(axis=1), axis=0)
        return out

    def normStd(df):
        out = df / (100 * np.sqrt(252) * df.std())
        return out

    def rp(df, **kwargs):

        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 250

        SRollVol = np.sqrt(252) * Slider.S(Slider.rollStatistics(df, method='Vol', nIn=nIn)) * 100
        out = (df / SRollVol).replace([np.inf, -np.inf], 0).fillna(0)

        return [out, SRollVol]

    def preCursor(df, preCursorDF, **kwargs):

        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = "roll"

        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 250

        if 'multiplier' in kwargs:
            multiplier = kwargs['multiplier']
        else:
            multiplier = 2

        if mode == 'roll':
            preCursorDF_Upper = preCursorDF.rolling(nIn).mean() + preCursorDF.rolling(nIn).std() * multiplier
            preCursorDF_Lower = preCursorDF.rolling(nIn).mean() - preCursorDF.rolling(nIn).std() * multiplier
        else:
            preCursorDF_Upper = preCursorDF.expanding(nIn).mean() + preCursorDF.expanding(nIn).std() * multiplier
            preCursorDF_Lower = preCursorDF.expanding(nIn).mean() - preCursorDF.expanding(nIn).std() * multiplier
        bbTS = pd.concat([preCursorDF_Lower, preCursorDF, preCursorDF_Upper], axis=1)

        binarizeUpper = Slider.sign(preCursorDF - preCursorDF_Upper)
        binarizeUpper[binarizeUpper > 0] = None
        binarizeUpper = binarizeUpper.abs()
        binarizeLower = Slider.sign(preCursorDF - preCursorDF_Lower)
        binarizeLower[binarizeLower < 0] = None
        binarizeLower = binarizeLower.abs()
        binarySignal = np.sign(binarizeUpper + binarizeLower)

        out = df.mul(binarySignal, axis=0)

        return [out, binarySignal]

    ########################

    def Paperize(df):
        outDF = df.reset_index()
        outDF['PaperText'] = ""
        for x in outDF.columns:
            if x != "PaperText":
                outDF['PaperText'] += outDF[x].astype(str) + " & "
        outDF['PaperText'] += "\\\\"
        return outDF

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

    def Hurst(df, **kwargs):

        def divGtN0(n, n0):
            # Find all divisors of the natural number N greater or equal to N0
            xx = np.arange(n0, math.floor(n / 2) + 1)
            return xx[(n / xx) == np.floor(n / xx)]

        def rscalc(z, n):
            m = int(z.shape[0] / n)
            y = np.reshape(z, (m, n)).T
            mu = np.mean(y, axis=0)
            sigma = np.std(y, ddof=1, axis=0)
            y = np.cumsum(y - mu, axis=0)
            yrng = np.max(y, axis=0) - np.min(y, axis=0)
            return np.mean(yrng / sigma)

        def hurstExponent(x, d=50):
            # Find such a natural number OptN that possesses the largest number of
            # divisors among all natural numbers in the interval [0.99*N,N]
            dmin, N, N0 = d, x.shape[0], math.floor(0.99 * x.shape[0])
            dv = np.zeros((N - N0 + 1,))
            for i in range(N0, N + 1):
                dv[i - N0] = divGtN0(i, dmin).shape[0]
            optN = N0 + np.max(np.arange(0, N - N0 + 1)[max(dv) == dv])
            # Use the first OptN values of x for further analysis
            x = x[:optN]
            d = divGtN0(optN, dmin)

            N = d.shape[0]
            RSe, ERS = np.zeros((N,)), np.zeros((N,))

            # Calculate empirical R/S
            for i in range(N):
                RSe[i] = rscalc(x, d[i])

            # Compute Anis-Lloyd [1] and Peters [3] corrected theoretical E(R/S)
            # (see [4] for details)
            for i in range(N):
                n = d[i]
                K = np.arange(1, n)
                ratio = (n - 0.5) / n * np.sum(np.sqrt((np.ones((n - 1)) * n - K) / K))
                if n > 340:
                    ERS[i] = ratio / math.sqrt(0.5 * math.pi * n)
                else:
                    ERS[i] = (math.gamma(0.5 * (n - 1)) * ratio) / (math.gamma(0.5 * n) * math.sqrt(math.pi))

            # Calculate the Anis-Lloyd/Peters corrected Hurst exponent
            # Compute the Hurst exponent as the slope on a loglog scale
            ERSal = np.sqrt(0.5 * math.pi * d)
            Pal = np.polyfit(np.log10(d), np.log10(RSe - ERS + ERSal), 1)
            Hal = Pal[0]

            # Calculate the empirical and theoretical Hurst exponents
            Pe = np.polyfit(np.log10(d), np.log10(RSe), 1)
            He = Pe[0]
            P = np.polyfit(np.log10(d), np.log10(ERS), 1)
            Ht = P[0]

            # Compute empirical confidence intervals (see [4])
            L = math.log2(optN)
            # R/S-AL (min(divisor)>50) two-sided empirical confidence intervals
            # pval95 = np.array([0.5-exp(-7.33*log(log(L))+4.21) exp(-7.20*log(log(L))+4.04)+0.5])
            lnlnL = math.log(math.log(L))
            c1 = [0.5 - math.exp(-7.35 * lnlnL + 4.06), math.exp(-7.07 * lnlnL + 3.75) + 0.5, 0.90]
            c2 = [0.5 - math.exp(-7.33 * lnlnL + 4.21), math.exp(-7.20 * lnlnL + 4.04) + 0.5, 0.95]
            c3 = [0.5 - math.exp(-7.19 * lnlnL + 4.34), math.exp(-7.51 * lnlnL + 4.58) + 0.5, 0.99]
            C = np.array([c1, c2, c3])

            detail = (d, optN, RSe, ERS, ERSal)
            return (Hal, He, Ht, C, detail)

        if 'conf' in kwargs:
            conf = kwargs['conf']
        else:
            conf = 'no'

        hurstMat = []
        for col in range(len(df.columns)):
            print(df.iloc[:, col].name)
            if conf == 'no':
                try:
                    H, c, data = compute_Hc(df.iloc[:, col], kind='change', simplified=True)
                except Exception as e:
                    print(e, df.iloc[:, col].name)
                    H, c, data = compute_Hc(df.iloc[1000:, col], kind='change', simplified=True)
                hurstMat.append(H)
            else:
                Hal, He, Ht, C, detail = hurstExponent(df.iloc[:, col].values)
                confIntervals = C[0]
                hurstMat.append([Hal, He, Ht, confIntervals[0], confIntervals[1]])

        hurstDF = pd.DataFrame(hurstMat, index=df.columns)

        return hurstDF

    def rollStatistics(df, method, **kwargs):
        if "mode" in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'Roll'
        if 'nIn' in kwargs:
            nIn = kwargs['nIn']
        else:
            nIn = 250
        if 'alpha' in kwargs:
            alpha = kwargs['alpha']
        else:
            alpha = 0.01

        if method == 'Vol':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, np.std, nIn)
            else:
                rollStatisticDF = Slider.expander(df, np.std, nIn)
        elif method == 'Skewness':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, skew, nIn)
            else:
                rollStatisticDF = Slider.expander(df, skew, nIn)
        elif method == 'Kurtosis':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, kurtosis, nIn)
            else:
                rollStatisticDF = Slider.expander(df, kurtosis, nIn)
        elif method == 'VAR':
            if mode == 'Roll':
                rollStatisticDF = norm.ppf(1 - alpha) * Slider.rollVol(df, nIn=nIn) - Slider.ema(df, nperiods=nIn)
            else:
                rollStatisticDF = norm.ppf(1 - alpha) * Slider.expanderVol(df, nIn=nIn) - Slider.ema(df, nperiods=nIn)
        elif method == 'CVAR':
            if mode == 'Roll':
                rollStatisticDF = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * Slider.rollVol(df, nIn=nIn) - Slider.ema(df,
                                                                                                                     nperiods=nIn)
            else:
                rollStatisticDF = alpha ** -1 * norm.pdf(norm.ppf(alpha)) * Slider.expanderVol(df,
                                                                      nIn=nIn) - Slider.ema(df,
                                                                                                                     nperiods=nIn)
        elif method == 'Sharpe':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, Slider.sharpe, nIn)
            else:
                rollStatisticDF = Slider.expander(df, Slider.sharpe, nIn)
        elif method == 'VAR_Quantile':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, Slider.VaR_Quantile, nIn)
            else:
                rollStatisticDF = Slider.expander(df, Slider.VaR_Quantile, nIn)

        elif method == 'Hurst':
            if mode == 'Roll':
                rollStatisticDF = Slider.roller(df, Slider.Hurst, nIn)
            else:
                rollStatisticDF = Slider.expander(df, Slider.Hurst, nIn)
        return rollStatisticDF

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

    def rollerVol(df, rvn):
        outDF = Slider.roller(df, np.std, rvn)
        return outDF

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
        out = np.sqrt(L / T) * Slider.roller(df, Slider.sharpe, window)
        return out

    def expander(df, func, n):
        EXPAND = df.expanding(min_periods=n, center=False).apply(lambda x: func(x))
        return EXPAND

    def expanderVol(df, rvn):
        outDF = Slider.expander(df, np.std, rvn)
        return outDF

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

    def MaximumDD(df):
        maxDDlist = []
        for c in df.columns:
            mdd = (df[c] + 1).cumprod().diff().min()
            maxDDlist.append(mdd)
        maxDDdf = pd.Series(maxDDlist, index=df.columns)

        return maxDDdf

    def annual_returns(returns):
        num_years = len(returns) / 252

        cum_ret_final = (returns + 1).prod().squeeze()
        return cum_ret_final ** (1 / num_years) - 1

    def sub_calmar(returns):
        # max_dd = max_drawdown(cumulative_returns(returns))
        max_dd = (returns + 1).cumprod().diff().min()

        if max_dd < 0:
            return Slider.annual_returns(returns) / abs(max_dd)

        return np.nan

    def Calmar(df):
        calmarlist = []
        for c in df.columns:
            calmar = Slider.sub_calmar(df[c])
            calmarlist.append(calmar)
        calmardf = pd.Series(calmarlist, index=df.columns)

        return calmardf

    def sharpe(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'standard'

        if mode == 'standard':
            out = df.mean() / df.std()
        elif mode == 'processNA':
            shList = []
            for c in df.columns:
                df[c] = df[c].dropna()
                medSh = df[c].mean() / df[c].std()
                shList.append(medSh)

            out = pd.Series(shList, index=df.columns)

        return out

    def VaR_Quantile(df, **kwargs):
        if 'a' in kwargs:
            a = kwargs['a']
        else:
            a = 0.1
        out = df.quantile(a)
        return out

    def tConfDF(df, **kwargs):
        if "scalingFactor" in kwargs:
            scalingFactor = kwargs["scalingFactor"]
        else:
            scalingFactor = 1

        tConfList = []
        for c in df.columns:
            tConfs = [np.round(x * scalingFactor, 2) for x in
                      st.t.interval(0.95, len(df[c].values) - 1, loc=np.mean(df[c].values), scale=st.sem(df[c].values))]
            tConfList.append([c, tConfs])

        tConfDF = pd.DataFrame(tConfList, columns=['index', 'tConf'])
        return tConfDF

    def ttestRV(df):
        cc = list(combinations(df.columns, 2))
        outList = []
        for c in cc:
            ttestPair = stats.ttest_ind(df[c[0]].values, df[c[1]].values, equal_var=False)
            outList.append([c[0], c[1], ttestPair.statistic, ttestPair.pvalue])
        out = pd.DataFrame(outList, columns=["pop1", "pop2", "t_statistic", "t_pvalue"])
        return out

    def downside_risk(returns, risk_free=0):
        adj_returns = returns - risk_free

        sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
        return np.sqrt(np.nanmean(sqr_downside) * 252)

    def sub_sortino(returns, risk_free=0):
        adj_returns = returns - risk_free

        drisk = Slider.downside_risk(adj_returns)
        if drisk == 0:
            return np.nan

        return (np.nanmean(adj_returns) * np.sqrt(252)) / drisk

    def sortino(df, **kwargs):
        if 'risk_free' in kwargs:
            risk_free = kwargs['risk_free']
        else:
            risk_free = 0

        sortinoList = []
        for c in df.columns:
            sortinoList.append(Slider.sub_sortino(df[c], risk_free))

        sortinoDF = pd.Series(sortinoList, index=df.columns)

        return sortinoDF

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
            rvunique = 40

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
        print("TOP = ", TOP)
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

    def BetaRegression(df, X, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 250
        BetaList = []
        for c in df.columns:
            if X not in c:
                RollVar_c = (1 / (Slider.rollerVol(df[c], 250) ** 2))
                # RollVar_c[RollVar_c > 100] = 1
                Beta_c = (df[X].rolling(n).cov(df[c])).mul(RollVar_c, axis=0).replace([np.inf, -np.inf], 0)
                Beta_c.name = c
                BetaList.append(Beta_c)
        BetaDF = pd.concat(BetaList, axis=1)

        return BetaDF

    def BetaKernel(df):

        BetaMatDF = pd.DataFrame(np.cov(df.T), index=df.columns, columns=df.columns)
        for idx, row in BetaMatDF.iterrows():
            BetaMatDF.loc[idx] /= row[idx]

        return BetaMatDF

    def RV(df, **kwargs):
        if "RVspace" in kwargs:
            RVspace = kwargs["RVspace"]
        else:
            RVspace = "classic"
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'Linear'
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 25
        if 'combos' in kwargs:
            combos = kwargs['combos']
        else:
            combos = 2
        if 'shOut' in kwargs:
            shOut = kwargs['shOut']
        else:
            shOut = 'off'

        if RVspace == "classic":
            cc = list(combinations(df.columns, combos))
        else:
            cc = [c for c in list(combinations(df.columns, combos)) if c[0] == RVspace]

        if mode == 'Linear':
            df0 = pd.concat([df[c[0]].sub(df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
        elif mode == 'Baskets':
            if shOut == 'off':
                df0 = pd.concat([df[c[0]].add(df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
            else:
                df0 = pd.concat(
                    [pd.Series(np.sqrt(252) * Slider.sharpe(Slider.rs(df[[col for col in c]]))) for c in tqdm(cc)],
                    axis=1, keys=cc)
        elif mode == 'PriceRatio':
            df0 = pd.concat([df[c[0]] / df[c[1]] for c in tqdm(cc)], axis=1, keys=cc)
        elif mode == 'PriceMultiply':
            df0 = pd.concat([df[c[0]] * df[c[1]] for c in tqdm(cc)], axis=1, keys=cc)
        elif mode == 'PriceRatio_zScore':
            lDF = []
            for c in tqdm(cc):
                PrRatio = df[c[0]] / df[c[1]]
                emaPrRatio = Slider.ema(PrRatio, nperiods=n)
                volPrRatio = Slider.expander(PrRatio, np.std, n)
                PrZScore = (PrRatio - emaPrRatio) / volPrRatio
                lDF.append(PrZScore)
            df0 = pd.concat(lDF, axis=1, keys=cc)
        elif mode == 'HedgeRatioPair':
            df0 = pd.concat([df[c[0]] - (Slider.S(df[c[0]].expanding(n).corr(df[c[1]]) * (
                    Slider.expander(df[c[0]], np.std, n) / Slider.expander(df[c[1]], np.std, n)), nperiods=2)
                                         * df[c[1]]) for c in cc], axis=1, keys=cc)
        elif mode == 'HedgeRatioBasket':
            df0 = pd.concat([df[c[0]] - (Slider.S(df[c[0]].expanding(n).corr(df[c[1]]) * (
                    Slider.expander(df[c[0]], np.std, n) / Slider.expander(df[c[1]], np.std, n)), nperiods=2)
                                         * df[c[1]]) for c in tqdm(cc)], axis=1, keys=cc)
        elif mode == 'HedgeRatioSimpleCorr':
            df0 = pd.concat(
                [df[c[0]] - (Slider.S(df[c[0]].expanding(n).corr(df[c[1]]), nperiods=2) * df[c[1]]) for c in tqdm(cc)],
                axis=1,
                keys=cc)

        df0.columns = df0.columns.map('_'.join)

        if shOut != 'Off':
            df0 = df0.T

        return df0.fillna(method='ffill').fillna(0)

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
        switchFlag = pd.Series(switchFlag, index=pnl.columns)
        return [pnl, switchFlag]

    def adf(df):
        X = df.values
        result = adfuller(X)
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        keys = [];
        values = []
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))
            keys.append(key);
            values.append(value)

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

    ############ PAPER RELATED #############

    def dateFilter(df):
        df.index = [x.replace("00:00:00", "").strip() for x in df.index]
        out = df
        return out

    def PaperSinglePlot(df, **kwargs):

        if "yLabelIn" in kwargs:
            yLabelIn = kwargs['yLabelIn']
        else:
            yLabelIn = 'Temp label'

        if "legendType" in kwargs:
            legendType = kwargs['legendType']
        else:
            legendType = 'out'

        if "positions" in kwargs:
            positions = kwargs['positions']
        else:
            positions = [0.95, 0.2, 0.85, 0.12, 0, 0]

        try:
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
        except:
            pass
        fig, ax = plt.subplots()
        mpl.pyplot.locator_params(axis='x', nbins=35)
        df.plot(ax=ax)
        for label in ax.get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(45)
        ax.set_xlim(xmin=0.0, xmax=len(df) + 1)
        mpl.pyplot.ylabel(yLabelIn, fontsize=32)
        if legendType == 'in':
            plt.legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'size': 24})
        elif legendType == 'out':
            plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        plt.subplots_adjust(top=positions[0], bottom=positions[1], right=positions[2], left=positions[3],
                            hspace=positions[4], wspace=positions[5])
        plt.margins(0, 0)
        plt.grid()
        plt.show()

    def plotCumulativeReturns(dfList, yLabelIn, **kwargs):

        if len(dfList) == 1:
            df = dfList[0] - 1
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            (df * 100).plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(df) + 1)
            mpl.pyplot.ylabel(yLabelIn[0], fontsize=32)
            plt.legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'size': 24})
            # plt.subplots_adjust(top=0.95, bottom=0.2, right=0.99, left=0.08, hspace=0, wspace=0)
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.12, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()
        elif len((dfList)) == 2:
            fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
            mpl.pyplot.locator_params(axis='x', nbins=35)
            titleList = ['(a)', '(b)']
            c = 0
            for df in dfList:
                df.index = [x.replace("00:00:00", "").strip() for x in df.index]
                df -= 1
                (df * 100).plot(ax=ax[c])
                for label in ax[c].get_xticklabels():
                    label.set_fontsize(25)
                    label.set_ha("right")
                    label.set_rotation(40)
                ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
                # ax[c].set_title(titleList[c], y=1.0, pad=-20)
                ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
                ax[c].set_ylabel(yLabelIn[c], fontsize=24)
                ax[c].legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'weight': 'bold', 'size': 24})
                ax[c].grid()
                c += 1
            # plt.subplots_adjust(top=0.95, bottom=0.2, right=0.99, left=0.08, hspace=0.1, wspace=0)
            plt.subplots_adjust(top=0.95, bottom=0.15, right=0.85, left=0.12, hspace=0.1, wspace=0)
            plt.show()
        else:
            print("More than 4 dataframes provided! Cant use this function for subplotting them - CUSTOMIZE ...")

    'ARIMA Operators'
    'Optimize order based on AIC or BIC fit'

    def get_optModel(data, opt, **kwargs):
        if 'orderList' in kwargs:
            orderList = kwargs['orderList']
        else:
            orderList = 0
        # ACFAsset = pd.DataFrame(acf(Asset, nlags=len(Asset)), columns=[AssetName])
        best_score, best_cfg = float("inf"), None

        if orderList == 0:
            for p in range(0, 7):
                for d in [0, 1]:
                    for q in range(0, 7):
                        order = (p, d, q)
                        if p != q:
                            try:
                                model = ARIMA(data, order=(p, d, q))
                                model_fit = model.fit(disp=0)
                                if opt == 'AIC':
                                    aicVal = model_fit.aic
                                elif opt == 'BIC':
                                    aicVal = model_fit.bic
                                if aicVal < best_score:
                                    best_score, best_cfg = aicVal, order
                                # print('ARIMA%s AIC=%.3f' % (order, aicVal))
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

    'Gaussian Process Regressors'

    def GPR_Walk(df, start, Kernel, rw):

        time_X = np.array(range(len(df))).reshape(len(df), 1)
        X = df.values.reshape(len(df), 1)
        size = int(len(X) * start)
        X_train, X_test = time_X[0:size], time_X[size:len(X)]
        y_train, y_test = X[0:size], X[size:len(X)]
        idx_train, idx_test = df.index[0:size], df.index[size:len(X)]

        if Kernel == "RBF":
            mainKernel = 1 * RBF()
        elif Kernel == "DotProduct":
            mainKernel = 1 * DotProduct()
        elif Kernel == "Matern":
            mainKernel = 1 * Matern()
        elif Kernel == "RationalQuadratic":
            mainKernel = 1 * RationalQuadratic()
        elif Kernel == "WhiteKernel":
            mainKernel = 1 * WhiteKernel()
        elif Kernel == "Matern_WhiteKernel":
            mainKernel = 1 * Matern() + 1 * WhiteKernel()

        gpcparamList = []
        test_price = []
        predictions = []

        c = 0
        for t in tqdm(range(len(X_test))):

            if c > 0:
                subhistory_X = X_train[-rw:-1]
                subhistory_y = y_train[-rw:-1]
            else:
                subhistory_X = X_train  # [-5:-1]
                subhistory_y = y_train  # [-5:-1]

            model = GaussianProcessRegressor(
                kernel=mainKernel)  # alpha=0.0, kernel=...,  n_restarts_optimizer=10, normalize_y=True

            model_prior_samples = model.sample_y(X=subhistory_X, n_samples=100)
            model_fit = model.fit(subhistory_X, subhistory_y)

            newX = X_test[t].reshape(1, -1)
            y_pred, y_std = model_fit.predict(newX, return_std=True)

            predictions.append(y_pred[0][0])
            X_train = np.append(X_train, newX, axis=0)
            y_train = np.append(y_train, [y_test[t]], axis=0)

            test_price.append(y_test[t][0])

            ###
            gpcparamList.append(y_std[-1])

            c += 1

        #print("len(X_test) = ", len(X_test), ", len(y_test) = ", len(y_test), ", len(idx_test) = ", len(idx_test))
        #print("len(test_price) = ", len(test_price))
        #print("len(predictions) = ", len(predictions))
        #time.sleep(2000)

        testDF = pd.DataFrame(test_price, index=idx_test)
        PredictionsDF = pd.DataFrame(predictions, index=idx_test)

        return [testDF, PredictionsDF, gpcparamList]

    def GPR_Walk(df, start, Kernel, rw):

        data_df_raw = df.reset_index()
        data_df_raw['t'] = data_df_raw.index
        data_df_raw['y'] = data_df_raw[df.name].fillna(0)
        data_df = data_df_raw[['t', 'y']]
        n = len(data_df_raw)

        x = data_df['t'].values.reshape(n, 1)
        y = data_df['y'].values.reshape(n, 1)
        n_train = round(start * n)
        x_train, x_test = x[:n_train], x[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        idx_train, idx_test = df.index[:n_train], df.index[n_train:]

        gpcparamList = []
        test_price = []
        predictions = []

        c = 0
        for t in tqdm(range(len(x_test))):

            if c > 0:
                subhistory_X = x_train[-rw:-1]
                subhistory_y = y_train[-rw:-1]
            else:
                subhistory_X = x_train  # [-5:-1]
                subhistory_y = y_train  # [-5:-1]

            ##########################################################################################
            newX = x_test[t].reshape(1, -1)

            with pm.Model() as model:
                # First seasonal component.
                ls_1 = pm.Gamma(name='ls_1', alpha=2.0, beta=1.0)
                period_1 = pm.Gamma(name='period_1', alpha=80, beta=2)
                gp1 = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_1, ls=ls_1))
                c_3 = pm.Normal(name='c_3', mu=np.mean(x_train), sigma=np.std(x_train))
                gp2 = pm.gp.Marginal(cov_func=pm.gp.cov.Linear(input_dim=1, c=c_3))
                gp = gp1 + gp2
                # Noise.
                sigma = pm.HalfNormal(name='sigma', sigma=np.std(x_train))
                # Likelihood.
                y_pred = gp.marginal_likelihood('y_pred', X=subhistory_X, y=subhistory_y.flatten(), noise=sigma)
                # Sample.
                trace = pm.sample(draws=100, chains=3, tune=100)

            with model:
                #x_train_conditional = gp.conditional('x_train_conditional', subhistory_X)
                #y_train_pred_samples = pm.sample_posterior_predictive(trace, vars=[x_train_conditional], samples=1)

                x_test_conditional = gp.conditional('x_test_conditional', newX)
                y_test_pred_samples = pm.sample_posterior_predictive(trace, vars=[x_test_conditional], samples=1)

            ##########################################################################################
            # Train
            #y_train_pred_samples_mean = y_train_pred_samples['x_train_conditional'].mean(axis=0)
            #y_train_pred_samples_std = y_train_pred_samples['x_train_conditional'].std(axis=0)
            #y_train_pred_samples_mean_plus = y_train_pred_samples_mean + 2 * y_train_pred_samples_std
            #y_train_pred_samples_mean_minus = y_train_pred_samples_mean - 2 * y_train_pred_samples_std
            # Test
            y_test_pred_samples_mean = y_test_pred_samples['x_test_conditional'].mean(axis=0)
            y_test_pred_samples_std = y_test_pred_samples['x_test_conditional'].std(axis=0)
            #y_test_pred_samples_mean_plus = y_test_pred_samples_mean + 2 * y_test_pred_samples_std
            #y_test_pred_samples_mean_minus = y_test_pred_samples_mean - 2 * y_test_pred_samples_std

            #print(y_test_pred_samples_mean)
            #print(len(y_test_pred_samples_mean))
            #print(y_test_pred_samples_std)
            #print(len(y_test_pred_samples_std))
            #time.sleep(500)
            ##########################################################################################
            predictions.append(y_test_pred_samples_mean[0])
            x_train = np.append(x_train, newX, axis=0)
            y_train = np.append(y_train, [y_test[t]], axis=0)

            test_price.append(y_test[t][0])

            ###
            gpcparamList.append(y_test_pred_samples_std)

            c += 1

        # print("len(X_test) = ", len(X_test), ", len(y_test) = ", len(y_test), ", len(idx_test) = ", len(idx_test))
        # print("len(test_price) = ", len(test_price))
        # print("len(predictions) = ", len(predictions))
        # time.sleep(2000)

        testDF = pd.DataFrame(test_price, index=idx_test)
        PredictionsDF = pd.DataFrame(predictions, index=idx_test)

        return [testDF, PredictionsDF, gpcparamList]

    'Arima Dataframe process'
    'Input: list of Dataframe, start: start/window, mode: rolling/expanding, opt: AIC, BIC, (p,d,q)'

    def ARIMA_process(datalist):
        Asset = datalist[0];
        start = datalist[1];
        mode = datalist[2];
        opt = datalist[3];
        orderList = datalist[4]
        AssetName = Asset.columns[0]

        X = Asset.to_numpy()
        predictions = []
        stderrList = []
        err = []
        confList = []
        for t in tqdm(range(start, len(X)), desc=AssetName):

            if mode == 'roll':
                history = X[t - start:t]
            elif mode == 'exp':
                history = X[0:t]
            try:
                if opt == 'AIC' or opt == 'BIC':
                    Orders = Slider.get_optModel(history, opt, orderList=orderList)
                else:
                    Orders = opt
                model = ARIMA(history, order=Orders)
                model_fit = model.fit(disp=0)
                # print(model_fit.resid)
                forecast, stderr, conf = model_fit.forecast()
                yhat = forecast
                c1 = conf[0][0]
                c2 = conf[0][1]
            except Exception as e:
                print(e)
                yhat = np.zeros(1) + X[t - 1]
                stderr = np.zeros(1)
                c1 = np.nan; c2 = np.nan

            predictions.append(yhat)
            stderrList.append(stderr)
            obs = X[t]
            err.append((yhat - obs)[0])
            confList.append((c1, c2))

        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderrList, index=Asset[start:].index, columns=[AssetName + '_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName + '_err'])
        confDF = pd.DataFrame(confList, index=Asset[start:].index, columns=[AssetName + '_conf1', AssetName + '_conf2'])

        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_Walk(df, start, orderIn, rw):
        X = df.values
        size = int(len(X) * start)
        #print("size =", size)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        arparamList = []
        predictions = [0] * len(history)
        c = 0
        for t in tqdm(range(len(test))):

            if c > 0:
                subhistory = history[-rw:]
            else:
                subhistory = history

            model = ARIMA(subhistory, order=orderIn)

            try:
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0][0]

                modelParamsDF = pd.DataFrame(model_fit.conf_int(), columns=['Lower', 'Upper'])
                modelParamsDF['pvalues'] = model_fit.pvalues
                modelParamsDF['params'] = model_fit.params

            except:
                yhat = np.nan

                nullSeries = pd.Series([np.nan] * (sum([x for x in orderIn]) + 1))
                modelParamsDF = pd.DataFrame()
                for col in ['Lower', 'Upper', 'pvalues', 'params']:
                    modelParamsDF[col] = nullSeries

            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            modelParamsDF['Dates'] = df.index[t + size]
            arparamList.append(modelParamsDF)

            ###

            c += 1

        testDF = pd.DataFrame(history, index=df.index)
        PredictionsDF = pd.DataFrame(predictions, index=df.index)

        return [testDF, PredictionsDF, arparamList]

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
            c1 = np.nan;
            c2 = np.nan

        obs = history[-1]
        err = (yhat - obs)
        predictions = yhat[0]
        PredictionsDF = pd.DataFrame(columns=[AssetName], index=Asset.index)
        stderrDF = pd.DataFrame(columns=[AssetName + '_stderr'], index=Asset.index)
        errDF = pd.DataFrame(columns=[AssetName + '_err'], index=Asset.index)
        confDF = pd.DataFrame(columns=[AssetName + '_conf1', AssetName + '_conf2'], index=Asset.index)

        PredictionsDF.loc[Asset.index[-1], AssetName] = predictions
        stderrDF.loc[Asset.index[-1], AssetName + '_stderr'] = stderr[0]
        errDF.loc[Asset.index[-1], AssetName + '_err'] = err[0]
        confDF.loc[Asset.index[-1], AssetName + '_conf1'] = c1;
        confDF.loc[Asset.index[-1], AssetName + '_conf2'] = c2

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
            if mode != 'static':
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
        history = historyOpt[0];
        opt = historyOpt[1];
        orderList = historyOpt[2]
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
            c1 = np.nan;
            c2 = np.nan
        obs = history[-1]
        err = (yhat - obs)
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
        # print(history)
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

        ## ML RELATED OPERATORS ##

        def overlappingPeriodSplitter(df, **kwargs):

            if 'sub_trainingSetIvl' in kwargs:
                sub_trainingSetIvl = kwargs['sub_trainingSetIvl']
            else:
                sub_trainingSetIvl = 250

            if 'sub_testSetInv' in kwargs:
                sub_testSetInv = kwargs['sub_testSetInv']
            else:
                sub_testSetInv = 250

            dfList = []
            i = 0
            while i < len(df):
                subProcessingHistory = df.iloc[i:i + sub_trainingSetIvl + sub_testSetInv]
                if len(subProcessingHistory) == sub_trainingSetIvl + sub_testSetInv:
                    #print(len(subProcessingHistory), ", from : ", subProcessingHistory.index[0], ", to : ",
                    #      subProcessingHistory.index[-1])
                    dfList.append(subProcessingHistory)
                else:
                    subProcessingHistory = df.iloc[i:]
                    print("end-Marginality-issue : --> ", len(subProcessingHistory), ", from : ",
                          subProcessingHistory.index[0], ", to : ", subProcessingHistory.index[-1])
                    dfList.append(subProcessingHistory)
                    break
                i += sub_testSetInv

            return dfList

        def gReshape(data, Features):

            samples = data.shape[0]
            if len(data.shape) == 1:
                TimeSteps = 1
            else:
                TimeSteps = data.shape[1]

            data = data.reshape((samples, TimeSteps, Features))

            return data

        ## MODELS ##

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

            if 'nD' not in kwargs:
                nD = 2
            else:
                nD = kwargs['nD']

            if 'sigma' not in kwargs:
                sigma = 'bgh'
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
                psi[col] = U[col] / U.iloc[:, 0]
                phi[col] = VT[col] * VT.iloc[:, 0]

            eigOut = psi.fillna(0)
            # eigOut = psi.iloc[:, 1:nD + 1].fillna(0)
            # eigOut = phi.iloc[:, 1:nD + 1].fillna(0)
            eigOut.columns = [str(x) for x in range(nD)]
            eigOut.index = df.index

            # print(df)
            # print(eigOut)
            # time.sleep(300)

            'Building Contractive Observer Data'
            aMat = []
            for z in df.index:
                aSubMat = []
                for eig in eigOut:
                    ajl = df.loc[z] * eigOut[eig]
                    aSubMat.append(ajl.sum())
                aMat.append(aSubMat)

            aMatDF = pd.DataFrame(aMat)
            # print("aMatDF = ", aMatDF)
            a_inv = pd.DataFrame(np.linalg.pinv(aMatDF.values))
            # print("a_inv = ", a_inv)
            lMat = pd.DataFrame(np.diag(s[:nD]))
            glA = gammaCO * pd.DataFrame(np.dot(a_inv.T, lMat))
            glA.index = df.index
            # for c in glA.columns:
            #    glA[c] /= glA[c].abs().sum()
            # print("df = ", df)

            # print("lMat = ", lMat)
            # print("glA = ", glA)
            # time.sleep(3000)
            """
            runCO = 0
            if runCO == 1:
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
                cObserver = (eq0 + eq1).fillna(0).iloc[-1]
            else:
                cObserver = 1
            return [eigOut, sigmaDMAPS, s[:nD], glA, cObserver]
            """

            return [glA, s[:nD], sigmaDMAPS]

        def pyDmapsRun(df, **kwargs):
            if 'nD' not in kwargs:
                nD = 2
            else:
                nD = kwargs['nD']

            X = df.values

            dmapObj = pydiffmap.diffusion_map.DiffusionMap.from_sklearn(n_evecs=nD, epsilon='bgh')

            dmapObj.construct_Lmat(X)
            sigmaDMAPS = dmapObj.epsilon_fitted
            TransitionMatrix = pd.DataFrame(dmapObj.kernel_matrix.toarray())
            U, s, VT = svd(TransitionMatrix)

            Udf = pd.DataFrame(U)

            try:
                dMapsProjectionOut = pd.DataFrame(dmapObj.fit_transform(X), columns=[str(x) for x in range(nD)],
                                                  index=df.index).fillna(0)
                eigFirst = Udf.iloc[:, :nD]
                eigFirst.columns = [str(x) for x in range(nD)]
                eigFirst.index = df.index

                eigLast = Udf.iloc[:, -nD:]
                eigLast.columns = [str(x) for x in range(nD)]
                eigLast.index = df.index
            except Exception as e:
                print(e)
                dMapsProjectionOut = pd.DataFrame(np.zeros((len(df.index), nD)), columns=[str(x) for x in range(nD)],
                                                  index=df.index).fillna(0)
                eigFirst = pd.DataFrame(np.zeros((len(df.index), nD)), columns=[str(x) for x in range(nD)],
                                        index=df.index).fillna(0)
                eigLast = pd.DataFrame(np.zeros((len(df.index), nD)), columns=[str(x) for x in range(nD)],
                                       index=df.index).fillna(0)

            return [dMapsProjectionOut, eigFirst, eigLast, s[:nD], sigmaDMAPS]

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

            Loadings_Target = [[] for j in range(len(eigsPC))]
            Loadings_First = [[] for j in range(len(eigsPC))]
            Loadings_Last = [[] for j in range(len(eigsPC))]
            lambdasList = []
            sigmaList = []
            for i in range(st, len(df0) + 1):

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
                elif Scaler == 'SimpleImputer':
                    x = SimpleImputer(missing_values=np.nan, strategy='constant').fit_transform(x)

                if manifoldIn == 'PCA':
                    pca = PCA(n_components=NumProjections)
                    X_pca = pca.fit_transform(x)
                    lambdasList.append(list(pca.singular_values_))
                    c = 0
                    for eig in eigsPC:
                        # print("c = ", c, ", eig = ", eig, ' : ', len(pca.components_[eig]), ',', len(pca.components_))
                        # print(list(pca.components_[eig]))
                        Loadings_Target[c].append(list(pca.components_[eig]))
                        c += 1

                elif manifoldIn == 'LLE':
                    lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections,
                                                          method=LLE_Method, n_jobs=-1)
                    X_lle = lle.fit_transform(x)  # ; print(X_lle.shape)
                    lambdasList.append(1)
                    c = 0
                    for eig in eigsPC:
                        Loadings_Target[c].append(list(X_lle[:, eig]))
                        c += 1

                elif manifoldIn == 'DMAP_gDmapsRun':
                    dMapsOut = Slider.AI.gDmaps(df, nD=NumProjections)
                    dmapsEigsOut = dMapsOut[0]
                    lambdasList.append(list(dMapsOut[1]))
                    sigmaList.append(dMapsOut[2])
                    c = 0
                    for eig in eigsPC:
                        Loadings_Target[c].append(dmapsEigsOut.iloc[:, eig])
                        c += 1

                elif manifoldIn == 'DMAP_pyDmapsRun':
                    dMapsOut = Slider.AI.pyDmapsRun(df, nD=NumProjections)
                    dMapsProjectionOut = dMapsOut[0]
                    eigFirst = dMapsOut[1]
                    eigLast = dMapsOut[2]
                    lambdasList.append(list(dMapsOut[3]))
                    sigmaList.append(dMapsOut[4])
                    c = 0
                    for eig in eigsPC:
                        Loadings_Target[c].append(dMapsProjectionOut.iloc[:, eig])
                        Loadings_First[c].append(eigFirst.iloc[:, eig])
                        Loadings_Last[c].append(eigLast.iloc[:, eig])
                        c += 1

            lambdasListDF = pd.DataFrame(lambdasList)
            lambdasDF = pd.concat(
                [pd.DataFrame(np.zeros((st - 1, lambdasListDF.shape[1])), columns=lambdasListDF.columns),
                 lambdasListDF],
                axis=0, ignore_index=True).fillna(0)
            lambdasDF.index = df0.index

            principalCompsDf_Target = [[] for j in range(len(Loadings_Target))]
            principalCompsDf_First = [[] for j in range(len(Loadings_First))]
            principalCompsDf_Last = [[] for j in range(len(Loadings_Last))]
            for k in range(len(Loadings_Target)):
                principalCompsDf_Target[k] = pd.concat(
                    [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                     pd.DataFrame(Loadings_Target[k], columns=df0.columns)], axis=0, ignore_index=True)
                principalCompsDf_Target[k].index = df0.index
                principalCompsDf_Target[k] = principalCompsDf_Target[k].ffill()
                ####
                try:
                    principalCompsDf_First[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Loadings_First[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf_First[k].index = df0.index
                    principalCompsDf_First[k] = principalCompsDf_First[k].ffill()
                    ####
                    principalCompsDf_Last[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Loadings_Last[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf_Last[k].index = df0.index
                    principalCompsDf_Last[k] = principalCompsDf_Last[k].ffill()
                except:
                    principalCompsDf_First[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Loadings_Target[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf_First[k].index = df0.index
                    principalCompsDf_First[k] = principalCompsDf_First[k].ffill()
                    ####
                    principalCompsDf_Last[k] = pd.concat(
                        [pd.DataFrame(np.zeros((st - 1, len(df0.columns))), columns=df0.columns),
                         pd.DataFrame(Loadings_Target[k], columns=df0.columns)], axis=0, ignore_index=True)
                    principalCompsDf_Last[k].index = df0.index
                    principalCompsDf_Last[k] = principalCompsDf_Last[k].ffill()

            if manifoldIn in ['PCA', 'LLE']:
                return [df0, principalCompsDf_Target, lambdasDF]
            elif manifoldIn in ['DMAP_gDmapsRun', 'DMAP_pyDmapsRun']:
                sigmaListDF = pd.DataFrame(sigmaList)
                sigmaDF = pd.concat(
                    [pd.DataFrame(np.zeros((st - 1, sigmaListDF.shape[1])), columns=sigmaListDF.columns), sigmaListDF],
                    axis=0, ignore_index=True).fillna(0)
                sigmaDF.index = df0.index

                return [df0, [principalCompsDf_Target, principalCompsDf_First, principalCompsDf_Last], lambdasDF,
                        sigmaDF]

        def gRollingManifoldPyErb(manifoldIn, df0, st, NumProjections, eigsPC, **kwargs):
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

            "CALCULATE ROLLING STATISTIC"
            if manifoldIn == 'CustomMetric':
                if 'CustomMetricStatistic' in kwargs:
                    CustomMetricStatistic = kwargs['CustomMetricStatistic']
                    metaDF_Rolling = Slider.rollStatistics(df0.copy(), CustomMetricStatistic)
                else:
                    CustomMetricStatistic = None
                    metaDF_Rolling = df0.copy()

                if 'CustomMetric' in kwargs:
                    CustomMetric = kwargs['CustomMetric']
                else:
                    CustomMetric = "euclidean"

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

                    customMetric = Slider.Metric(metaDF_Rolling, statistic=CustomMetricStatistic, metric=CustomMetric)
                    lambdasList.append(list(customMetric[0]))
                    sigmaList.append(list(customMetric[0]))
                    c = 0
                    for eig in eigsPC:
                        # print(eig, ', customMetric[1][eig] =', customMetric[1][eig]) # 0 , 100 , 5
                        Comps[c].append(list(customMetric[1][eig]))
                        c += 1

                elif manifoldIn == 'PCA':
                    pca = PCA(n_components=NumProjections)
                    X_pca = pca.fit_transform(x)
                    lambdasList.append(list(pca.singular_values_))
                    sigmaList.append(list(pca.explained_variance_ratio_))
                    c = 0
                    for eig in eigsPC:
                        # print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                        Comps[c].append(list(pca.components_[eig]))
                        c += 1

                elif manifoldIn == 'BetaRegressV':
                    BetaKernelDF = Slider.BetaKernel(df)

                    lambdasList.append(1)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        Comps[c].append(BetaKernelDF.iloc[:, eig].tolist())
                        c += 1

                elif manifoldIn == 'BetaRegressH':
                    BetaKernelDF = Slider.BetaKernel(df)

                    lambdasList.append(1)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        Comps[c].append(BetaKernelDF.iloc[eig, :].tolist())
                        c += 1

                elif manifoldIn == 'BetaRegressC':
                    BetaKernelDF = Slider.BetaKernel(df)

                    lambdasList.append(1)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        Comps[c].append((BetaKernelDF.iloc[eig, :] + BetaKernelDF.iloc[:, eig]).tolist())
                        c += 1

                elif manifoldIn == 'Beta':
                    BetaKernelDF = Slider.BetaKernel(df)
                    U, s, VT = svd(BetaKernelDF.values)

                    lambdasList.append(s)
                    sigmaList.append(1)
                    c = 0
                    for eig in eigsPC:
                        Comps[c].append(U[eig])
                        c += 1

                elif manifoldIn == 'DMAPS':
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

                    exPostProjections[k] = df0 * Slider.S(principalCompsDf[k])

                return [df0, principalCompsDf, exPostProjections, sigmaDF, lambdasDF]

            else:

                return [df0, pd.DataFrame(eigDf), pd.DataFrame(cObserverList), sigmaDF, lambdasDF, Comps, eigCoeffs]

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

        """
        def gRNN_Repo(dataset_all, params):

            history = History()

            ################################### Data Preprocessing ###################################################

            SingleSample = False
            if isinstance(dataset_all, pd.Series):
                SingleSample = True

            TrainEnd = int(params["TrainEndPct"] * len(dataset_all))
            dataVals = dataset_all.values

            #################################### Feature Scaling #####################################################
            # sc = StandardScaler()
            # sc = MinMaxScaler() #feature_range=(0, 1)
            if SingleSample:
                # dataVals = sc.fit_transform(dataVals.reshape(-1, 1))
                dataVals = dataVals.reshape(-1, 1)
                FeatSpaceDims = 1
                outNaming = [dataset_all.name]
                print(outNaming)
            else:
                # dataVals = sc.fit_transform(dataVals)
                FeatSpaceDims = len(dataset_all.columns)
                outNaming = dataset_all.columns
            #################### Creating a data structure with N timesteps and 1 output #############################
            X = []
            y = []
            for i in range(params["TrainWindow"], len(dataset_all)):
                X.append(dataVals[i - params["TrainWindow"]:i - params["HistLag"]])
                y.append(dataVals[i])
            X, y = np.array(X), np.array(y)
            yBinary = np.sign(y)
            yBinary[yBinary == -1] = 0
            idx = dataset_all.iloc[params["TrainWindow"]:].index

            ####################################### Reshaping ########################################################
            "Samples : One sequence is one sample. A batch is comprised of one or more samples."
            "Time Steps : One time step is one point of observation in the sample."
            "Features : One feature is one observation at a time step."
            X = np.reshape(X, (X.shape[0], X.shape[1], FeatSpaceDims))

            X_train, y_train, yBinary_train = X[:TrainEnd], y[:TrainEnd], yBinary[:TrainEnd]
            X_test, y_test, yBinary_test = X[TrainEnd:], y[TrainEnd:], yBinary[TrainEnd:]

            # df_real_price_train = pd.DataFrame(sc.inverse_transform(y_train), index=idx[:TrainEnd], columns=outNaming)
            # df_real_price_test = pd.DataFrame(sc.inverse_transform(y_test), index=idx[TrainEnd:], columns=outNaming)
            df_real_price_train = pd.DataFrame(y_train, index=idx[:TrainEnd], columns=outNaming)
            df_real_price_test = pd.DataFrame(y_test, index=idx[TrainEnd:], columns=outNaming)

            print("X.shape=", X.shape, ", TrainWindow=", params["TrainWindow"],
                  ", TrainEnd=", TrainEnd, ", X_train.shape=", X_train.shape, ", y_train.shape=", y_train.shape,
                  ", X_test.shape=", X_test.shape, ", y_test.shape=", y_test.shape)
            # print("y_train = ", y_train, ", yBinary_train = ", yBinary_train)

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
            regressor.add(Dense(units=yBinary_train.shape[1]))

            ######################################## Compiling the RNN ###############################################
            my_callbacks = [
                # tf.keras.callbacks.RemoteMonitor(root="http://localhost:9000", path="/publish/epoch/end/", field="data", headers=None, send_as_json=False, ),
                tf.keras.callbacks.EarlyStopping(patience=params["EarlyStopping_patience_Epochs"]),
                history,
                # tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
                # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            ]
            regressor.compile(optimizer=params["CompilerSettings"][0], loss=params["CompilerSettings"][1])
            # Fitting the RNN to the Training set
            regressor.fit(X_train, yBinary_train, epochs=params["epochsIn"], batch_size=params["batchSIzeIn"],
                          verbose=0,
                          callbacks=my_callbacks)

            ########################## Get Predictions for Static or Online Learning #################################
            # predicted_price_train = sc.inverse_transform(regressor.predict(X_train))
            predicted_price_train = regressor.predict(X_train)

            scoreList = []
            if params["LearningMode"] == 'static':
                # predicted_price_test = sc.inverse_transform(regressor.predict(X_test))
                predicted_price_test = regressor.predict(X_test)
                scoreDF = pd.DataFrame(history.history['loss'], columns=['loss'])

            elif params["LearningMode"] == 'static_MultiStep_Ahead':

                predicted_price_test = []  #
                # print(1, ", X_test[0]=", X_test[0],
                #      ", reShaped_X_test[0]=", np.reshape(X_test[0][0], (1, 1, 1)),
                #      ", UnConvertedPrediction=",
                #      regressor.predict(np.reshape(X_test[0], (1, X_test.shape[1], FeatSpaceDims))))

                predicted_price_test.append(
                    regressor.predict(np.reshape(X_test[0], (1, 1, 1))))
                # print("Predicting ", len(df_real_price_test) - 1, " steps ahead...")
                for obs in range(params["static_MultiStep_Ahead_Horizon"]):
                    predicted_price_test.append(
                        regressor.predict(np.reshape(predicted_price_test[-1], (1, 1, 1)))[0])
                # print(predicted_price_test)
                # predicted_price_test = sc.inverse_transform(predicted_price_test)
                # print("predicted_price_test : ", predicted_price_test)
                scoreDF = pd.DataFrame(history.history['loss'], columns=['loss'])

            elif params["LearningMode"] == 'onlineRepo':

                # X_test, y_test
                predicted_price_test = []
                for i in range(len(X_test)):
                    # X_test[i], y_test[i] = np.array(X_test[i]), np.array(y_test[i])

                    # print('Calculating: ' + str(round(i / len(X_test) * 100, 2)) + '%')
                    # print("X_test.shape=", X_test.shape, ", y_test.shape=", y_test.shape)
                    # print("X_test[i].shape=", X_test[i].shape, ", y_test[i].shape=", y_test[i].shape)
                    # print("X_test[i]=", X_test[i])
                    # print("y_test[i]=", y_test[i])

                    # print("(1, X_test[i].shape[0], len(dataset_all.columns)) = ",
                    #      (1, X_test[i].shape[0], FeatSpaceDims))
                    indXtest = np.reshape(X_test[i], (1, X_test[i].shape[0], FeatSpaceDims))
                    # print("(X_test[i].shape[0], 1, len(dataset_all.columns)) = ", (X_test[i].shape[0], 1, len(dataset_all.columns)))
                    # indXtest = np.reshape(X_test[i], (X_test[i].shape[0], 1, len(dataset_all.columns)))

                    # print("indXest.shape=", indXtest.shape)
                    # print("predicted_price_test=", sc.inverse_transform(regressor.predict(indXtest))[0])
                    # predicted_price_test.append(sc.inverse_transform(regressor.predict(indXtest))[0])
                    predicted_price_test.append(regressor.predict(indXtest)[0])

                    indYtest = np.reshape(y_test[i], (1, FeatSpaceDims))

                    try:
                        sc.partial_fit(dataVals[i + TrainEnd])
                    except Exception as e:
                        print(e)
                    regressor.train_on_batch(indXtest, indYtest)
                    scores = regressor.evaluate(indXtest, indYtest, verbose=0)
                    scoreList.append(scores)
                    # print(scores)

            elif params["LearningMode"] == "online":
                rw = params["rw"]
                predicted_price_test = []
                c = 0
                for t in tqdm(range(len(X_test))):

                    if c > 0:
                        subhistory_X = X_train[-rw:-1]
                        subhistory_y = yBinary_train[-rw:-1]
                    else:
                        subhistory_X = X_train  # [-5:-1]
                        subhistory_y = yBinary_train  # [-5:-1]

                    newX = np.reshape(X_test[t], (1, X_test[t].shape[0], FeatSpaceDims))
                    pred_test = regressor.predict(newX)
                    # predicted_price_test.append(sc.inverse_transform(pred_test)[0])
                    predicted_price_test.append(pred_test[0])

                    regressor.fit(subhistory_X, subhistory_y, epochs=params["epochsIn"],
                                  batch_size=params["batchSIzeIn"], verbose=0,
                                  callbacks=my_callbacks)

                    X_train = np.append(X_train, newX, axis=0)
                    yBinary_train = np.append(yBinary_train, [yBinary_test[t]], axis=0)

                    scores = regressor.evaluate(newX, yBinary_test[t], verbose=0)
                    scoreList.append(scores)

                    # print("X_train.shape = ", X_train.shape, ", y_train.shape = ", y_train.shape, ", scores = ", scores)

                    c += 1

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
        """

        def gClassification(dataset_all, params):

            ################################### Very First Data Preprocessing #########################################

            dataVals = dataset_all.values

            if isinstance(dataset_all, pd.Series): # Single Sample
                FeatSpaceDims = 1
                outNaming = [dataset_all.name]
                print(outNaming)
            else:
                FeatSpaceDims = len(dataset_all.columns)
                outNaming = dataset_all.columns

            #################################### Feature Scaling #####################################################
            if params['Scaler'] == "Standard":
                sc_X = StandardScaler()
                #sc_y = StandardScaler()
            elif params['Scaler'] == 'MinMax':
                sc_X = MinMaxScaler() #feature_range=(0, 1)
                #sc_y = MinMaxScaler() #feature_range=(0, 1)

            #################### Creating a data structure with N timesteps and 1 output #############################
            X = []
            y = []
            real_y = []
            for i in range(params["InputSequenceLength"], len(dataset_all)):
                X.append(dataVals[i - params["InputSequenceLength"]:i - params["HistLag"]])
                y.append(np.sign(dataVals[i]))
                real_y.append(dataVals[i])
            X, y, real_y = np.array(X), np.array(y), np.array(real_y)
            y[y==-1] = 2
            idx = dataset_all.iloc[params["InputSequenceLength"]:].index

            #print("X.shape=", X.shape, ", y.shape=", y.shape,", real_y.shape=", real_y.shape, ", FeatSpaceDims=", FeatSpaceDims,
            #      ", InputSequenceLength=", params["InputSequenceLength"],
            #      ", SubHistoryLength=", params["SubHistoryLength"],
            #      ", SubHistoryTrainingLength=", params["SubHistoryTrainingLength"], ", len(idx) = ", len(idx))

            stepper = params["SubHistoryLength"]-params["SubHistoryTrainingLength"]

            df_predicted_price_train_List = []
            df_real_price_class_train_List = []
            df_real_price_train_List = []
            df_predicted_price_test_List = []
            df_real_price_class_test_List = []
            df_real_price_test_List = []

            breakFlag = False
            megaCount = 0
            for i in tqdm(range(0, X.shape[0], stepper)):
                subProcessingHistory_X, subProcessingHistory_y, subProcessingHistory_real_y = X[i:i+params["SubHistoryLength"]], \
                                                                                              y[i:i+params["SubHistoryLength"]], \
                                                                                              real_y[i:i+params["SubHistoryLength"]]
                subIdx = idx[i:i+params["SubHistoryLength"]]
                if len(subProcessingHistory_X) < params["SubHistoryLength"]:
                    subProcessingHistory_X, subProcessingHistory_y, subProcessingHistory_real_y = X[i:], y[i:], real_y[i:]
                    subIdx = idx[i:]
                    breakFlag = True
                X_train, y_train, real_y_train = subProcessingHistory_X[:params["SubHistoryTrainingLength"]],\
                                           subProcessingHistory_y[:params["SubHistoryTrainingLength"]],\
                                           subProcessingHistory_real_y[:params["SubHistoryTrainingLength"]],
                subIdx_train = subIdx[:params["SubHistoryTrainingLength"]]
                X_test, y_test, real_y_test = subProcessingHistory_X[params["SubHistoryTrainingLength"]:], \
                                              subProcessingHistory_y[params["SubHistoryTrainingLength"]:],\
                                              subProcessingHistory_real_y[params["SubHistoryTrainingLength"]:]
                subIdx_test = subIdx[params["SubHistoryTrainingLength"]:]

                # Enable Scaling
                if params['Scaler'] is not None:
                    X_train = sc_X.fit_transform(X_train)
                    X_test = sc_X.transform(X_test)

                #print("Data subHistories Set : i = ", i, ", len(subProcessingHistory_X) = ", len(subProcessingHistory_X),
                #      ", len(subProcessingHistory_y) = ", len(subProcessingHistory_y),
                #      ", X_train.shape = ", X_train.shape, ", y_train = ", y_train.shape, ", X_test.shape = ", X_test.shape,
                #      ", y_test.shape = ", y_test.shape)

                ####################################### Reshaping ########################################################
                "Samples : One sequence is one sample. A batch is comprised of one or more samples."
                "Time Steps : One time step is one point of observation in the sample."
                "Features : One feature is one observation at a time step."

                ################################### Build the RNN (LSTM) #####################################
                #print("megaCount = ", megaCount)

                if params["model"] == "RNN":
                    X_train, X_test = Slider.AI.gReshape(X_train, FeatSpaceDims), \
                                      Slider.AI.gReshape(X_test,  FeatSpaceDims)
                    if megaCount == 0:
                        #print("After Reshaping : X_train.shape = ", X_train.shape,
                        #      ", y_train = ", y_train.shape, ", X_test.shape = ", X_test.shape,
                        #      ", y_test.shape = ", y_test.shape)

                        ########################################## RNN #############################################
                        print("Recurrent Neural Networks Classification...", outNaming)
                        model = Sequential()
                        # Adding the first LSTM layer and some Dropout regularisation
                        for layer in range(len(params["medSpecs"])):
                            if params["medSpecs"][layer]["units"] == 'xShape1':
                                unitsIn = X_train.shape[1]
                                if params["medSpecs"][layer]["LayerType"] == "LSTM":
                                    model.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                                       unit_forget_bias=True, bias_initializer='ones',
                                                       input_shape=(X_train.shape[1], FeatSpaceDims)))
                                elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                                    model.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                                            unit_forget_bias=True, bias_initializer='ones',
                                                            input_shape=(X_train.shape[1], FeatSpaceDims)))
                                model.add(Dropout(params["medSpecs"][layer]["Dropout"]))
                            else:
                                unitsIn = params["medSpecs"][layer]["units"]
                                if params["medSpecs"][layer]["LayerType"] == "LSTM":
                                    model.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                                elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                                    model.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                                model.add(Dropout(params["medSpecs"][layer]["Dropout"]))
                        # Adding the output layer
                        if len(y_train.shape) == 1:
                            model.add(Dense(units=1))
                        else:
                            model.add(Dense(units=y_train.shape[1]))
                        ####
                        my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=params["EarlyStopping_patience_Epochs"])]
                        model.compile(optimizer=params["CompilerSettings"][0], loss=params["CompilerSettings"][1])
                    # Fitting the RNN Model to the Training set
                    model.fit(X_train, y_train, epochs=params["epochsIn"], batch_size=params["batchSIzeIn"],
                              verbose=0, callbacks=my_callbacks)

                elif params["model"] == "GPC":
                    ########################################## GPC #############################################
                    # define model
                    if megaCount == 0:
                        print("Gaussian Process Classification...", outNaming)
                        model = GaussianProcessClassifier()
                        if params['Kernel'] == 'Optimize':
                            # define model evaluation method
                            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                            # define grid
                            grid = dict()
                            grid['kernel'] = [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(),
                                              1 * WhiteKernel()]
                            # define search
                            search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
                            # perform the search
                            results = search.fit(X_train, y_train)
                            # summarize best
                            print('Best Mean Accuracy: %.3f' % results.best_score_)
                            print('Best Config: %s' % results.best_params_)
                            # summarize all
                            gpc_means = results.cv_results_['mean_test_score']
                            gpc_params = results.cv_results_['params']
                            for gpc_mean, gpc_param in zip(gpc_means, gpc_params):
                                print(">%.3f with: %r" % (gpc_mean, gpc_param))
                                print(gpc_param['kernel'])
                            print("GPC Fitting using Kernel = ", results.best_params_['kernel'])
                            mainKernel = results.best_params_['kernel']

                        elif params['Kernel'] == '0':
                            mainKernel = 1**2 * Matern(length_scale=1, nu=0.5) + 1**2 * DotProduct(sigma_0=1) +\
                                                  1**2 * RationalQuadratic(alpha=1, length_scale=1) + 1**2 * ConstantKernel()
                        elif params['Kernel'] == '1':
                            # Add Noise
                            mainKernel = 1**2 * Matern(length_scale=1, nu=0.5) + 1**2 * DotProduct(sigma_0=1) +\
                                                  1**2 * RationalQuadratic(alpha=1, length_scale=1) + 1**2 * ConstantKernel()+\
                                         1**2 * WhiteKernel()
                        ##################### Running with Greedy Search Best Model ##################
                        model = GaussianProcessClassifier(kernel=mainKernel, random_state=0)
                    # Fitting the GPC Model to the Training set
                    model.fit(X_train, y_train)

                elif params["model"] == "GPR":
                    if megaCount == 0:
                        print("Gaussian Process Regression...", outNaming)
                        if params['Kernel'] == 'Optimize':
                            # define model evaluation method
                            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
                            # define grid
                            grid = dict()
                            grid['kernel'] = [1 * RBF(), 1 * DotProduct(), 1 * Matern(), 1 * RationalQuadratic(),
                                              1 * WhiteKernel()]
                            # define search
                            search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
                            # perform the search
                            results = search.fit(X_train, y_train)
                            # summarize best
                            print('Best Mean Accuracy: %.3f' % results.best_score_)
                            print('Best Config: %s' % results.best_params_)
                            # summarize all
                            gpc_means = results.cv_results_['mean_test_score']
                            gpc_params = results.cv_results_['params']
                            for gpc_mean, gpc_param in zip(gpc_means, gpc_params):
                                print(">%.3f with: %r" % (gpc_mean, gpc_param))
                                print(gpc_param['kernel'])
                            print("GPC Fitting using Kernel = ", results.best_params_['kernel'])
                            mainKernel = results.best_params_['kernel']

                        elif params['Kernel'] == '0':
                            mainKernel = 1 ** 2 * Matern(length_scale=1, nu=0.5) + 1 ** 2 * DotProduct(sigma_0=1) + \
                                         1 ** 2 * RationalQuadratic(alpha=1, length_scale=1) + 1 ** 2 * ConstantKernel()
                        elif params['Kernel'] == '1':
                            # Add Noise
                            mainKernel = 1**2 * Matern(length_scale=1, nu=0.5) + 1**2 * DotProduct(sigma_0=1) +\
                                                  1**2 * RationalQuadratic(alpha=1, length_scale=1) + 1**2 * ConstantKernel()+\
                                         1**2 * WhiteKernel()
                        model = GaussianProcessRegressor(kernel=mainKernel)

                    model.fit(X_train, y_train)
                    y_pred_tr, y_pred_tr_std = model.predict(X_train, return_std=True)
                    y_pred_te, y_pred_te_std = model.predict(X_test, return_std=True)

                ############################### TRAIN PREDICT #################################
                #if params['Scaler'] is None:
                predicted_price_train = model.predict(X_train)
                if params["model"] == "GPC":
                    predicted_price_proba_train = model.predict_proba(X_train)
                #else:
                #    print("model.predict(X_train) = ", model.predict(X_train))
                #    predicted_price_train = sc_y.inverse_transform(model.predict(X_train))
                #    print("predicted_price_train = ", predicted_price_train)
                #    time.sleep(3000)

                df_predicted_price_train = pd.DataFrame(predicted_price_train, index=subIdx_train,
                                                        columns=['Predicted_Train_'+str(x) for x in outNaming])
                if params["model"] == "GPC":
                    df_predicted_price_proba_train = pd.DataFrame(predicted_price_proba_train, index=subIdx_train,
                                                                  columns=['Predicted_Proba_Train_'+str(x) for x in model.classes_])
                    df_predicted_price_train = pd.concat([df_predicted_price_train, df_predicted_price_proba_train], axis=1)
                df_real_price_class_train = pd.DataFrame(y_train, index=subIdx_train,
                                                   columns=['Real_Train_Class_'+str(x) for x in outNaming])
                df_real_price_train = pd.DataFrame(real_y_train, index=subIdx_train,
                                                   columns=['Real_Train_'+str(x) for x in outNaming])
                ############################### TEST PREDICT ##################################
                if params["LearningMode"] == 'static':
                    #if params['Scaler'] is None:
                    predicted_price_test = model.predict(X_test)
                    if params["model"] == "GPC":
                        predicted_price_proba_test = model.predict_proba(X_test)
                    # else:
                    #    print("model.predict(X_test) = ", model.predict(X_test))
                    #    predicted_price_test = sc_y.inverse_transform(model.predict(X_test))
                    #    print("predicted_price_test = ", predicted_price_test)
                    #    time.sleep(3000)

                    df_predicted_price_test = pd.DataFrame(predicted_price_test, index=subIdx_test,
                                                           columns=['Predicted_Test_'+x for x in outNaming])
                    if params["model"] == "GPC":
                        df_predicted_price_proba_test = pd.DataFrame(predicted_price_proba_test, index=subIdx_test,
                                                                      columns=['Predicted_Proba_Test_' + str(x) for x in model.classes_])
                        df_predicted_price_test = pd.concat([df_predicted_price_test, df_predicted_price_proba_test], axis=1)
                    df_real_price_class_test = pd.DataFrame(y_test, index=subIdx_test,
                                                           columns=['Real_Test_Class_'+x for x in outNaming])
                    df_real_price_test = pd.DataFrame(real_y_test, index=subIdx_test,
                                                           columns=['Real_Test_'+x for x in outNaming])

                df_predicted_price_train_List.append(df_predicted_price_train)
                df_real_price_class_train_List.append(df_real_price_class_train)
                df_real_price_train_List.append(df_real_price_train)
                df_predicted_price_test_List.append(df_predicted_price_test)
                df_real_price_class_test_List.append(df_real_price_class_test)
                df_real_price_test_List.append(df_real_price_test)

                megaCount += 1

                if breakFlag:
                    break

            df_predicted_price_train_DF = pd.concat(df_predicted_price_train_List, axis=0)
            df_real_price_class_train_DF = pd.concat(df_real_price_class_train_List, axis=0)
            df_real_price_train_DF = pd.concat(df_real_price_train_List, axis=0)
            df_predicted_price_test_DF = pd.concat(df_predicted_price_test_List, axis=0)
            df_real_price_class_test_DF = pd.concat(df_real_price_class_test_List, axis=0)
            df_real_price_test_DF = pd.concat(df_real_price_test_List, axis=0)

            return [df_predicted_price_train_DF, df_real_price_class_train_DF, df_real_price_train_DF,
                     df_predicted_price_test_DF, df_real_price_class_test_DF, df_real_price_test_DF]

    class Strategies:

        def EMA_signal(df, **kwargs):
            if 'nIn' in kwargs:
                nIn = kwargs['nIn']
            else:
                nIn = 3

            sig = Slider.sign(Slider.ema(df, nperiods=nIn))
            pnl = df.mul(Slider.S(sig), axis=0)
            return pnl