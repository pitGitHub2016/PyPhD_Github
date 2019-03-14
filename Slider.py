import pandas as pd
import numpy as np

class Slider:

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

    def roller(df, func, n):
        ROLL = df.rolling(window=n, center=False).apply(lambda x: func(x))
        return ROLL

    def rollerVol(df, rvn):
        return Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def rollVol(df, rvn):
        return np.sqrt(len(df) / 14) * Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def rollSh(df):
        return np.sqrt(len(df) / 14) * Slider.roller(df, Slider.sharpe, 100)

    def sharpe(df):
        return df.mean() / df.std()

    def CorrMatrix(df):
        return df.corr()

    def bma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        MA = pd.DataFrame(df.rolling(nperiods, min_periods=nperiods).mean()).fillna(0)
        return MA

    def bema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = pd.DataFrame(df.ewm(span=nperiods, min_periods=nperiods).mean()).fillna(0)
        return EMA
