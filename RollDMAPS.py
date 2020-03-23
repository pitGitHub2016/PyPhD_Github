from Slider import Slider as sl
import numpy as np, pickle
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
import glob, investpy
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodDataDMAPS.db')

def DataHandler(mode):
    if mode == 'investingCom':
        dataAll = []
        for fx in ['USD/SEK', 'USD/NOK', 'USD/ZAR', 'USD/RUB', 'USD/CNH', 'USD/INR', 'USD/COP', 'USD/CAD', 'USD/EUR',
                   'USD/ILS',
                   'USD/CRC', 'USD/PLN', 'USD/CHF', 'USD/CZK', 'USD/IDR', 'USD/HUF', 'USD/KRW', 'USD/GBP', 'USD/MXN',
                   'USD/CLP', 'USD/DKK', 'USD/NZD', 'USD/JPY', 'USD/AUD']:
            print(fx)
            name = fx.replace('/', '')
            df = investpy.get_currency_cross_historical_data(currency_cross=fx, from_date='01/01/1990',
                                                             to_date='14/03/2020').reset_index().rename(
                columns={"Date": "Dates", "Close": name}).set_index('Dates')[name]
            dataAll.append(df)
        pd.concat(dataAll, axis=1).to_sql('FxDataInvestingCom', conn, if_exists='replace')

def shortTermInterestRatesSetup():
    # df = pd.read_csv('shortTermInterestRates.csv')
    df = pd.read_csv('shortTermInterestRates_14032020.csv')
    irList = []
    for item in set(list(df['LOCATION'])):
        df0 = df[df['LOCATION'] == item].reset_index()
        df1 = df0[['TIME', 'Value']].set_index('TIME')
        df1.columns = [item]
        irList.append(df1)
    pd.concat(irList, axis=1).to_sql('irData', conn, if_exists='replace')

    rawData = pd.read_sql('SELECT * FROM irData', conn)
    rawData['index'] = pd.to_datetime(rawData['index'])
    irData = rawData.set_index('index').fillna(method='ffill')
    # dailyReSample = irData.resample('D').mean().fillna(method='ffill')
    dailyReSample = irData.resample('D').interpolate(method='linear').fillna(method='ffill').fillna(0)
    dailyReSample.to_sql('irDataDaily', conn, if_exists='replace')

    dailyIR = (pd.read_sql('SELECT * FROM irDataDaily', conn).set_index('index') / 365) / 100
    IRD = dailyIR

    IRD['USDSEK'] = IRD['USA'] - IRD['SWE']
    IRD['USDNOK'] = IRD['USA'] - IRD['NOR']
    IRD['USDZAR'] = IRD['USA'] - IRD['ZAF']
    IRD['USDRUB'] = IRD['USA'] - IRD['RUS']
    IRD['USDCNH'] = IRD['USA'] - IRD['CHN']
    IRD['USDINR'] = IRD['USA'] - IRD['IND']
    IRD['USDCOP'] = IRD['USA'] - IRD['COL']
    IRD['USDCAD'] = IRD['USA'] - IRD['CAN']
    IRD['USDEUR'] = IRD['USA'] - IRD['EA19']
    IRD['USDILS'] = IRD['USA'] - IRD['ISR']
    IRD['USDCRC'] = IRD['USA'] - IRD['CRI']
    IRD['USDPLN'] = IRD['USA'] - IRD['POL']
    IRD['USDCHF'] = IRD['USA'] - IRD['CHE']
    IRD['USDCZK'] = IRD['USA'] - IRD['CZE']
    IRD['USDIDR'] = IRD['USA'] - IRD['IDN']
    IRD['USDHUF'] = IRD['USA'] - IRD['HUN']
    IRD['USDKRW'] = IRD['USA'] - IRD['KOR']
    IRD['USDGBP'] = IRD['USA'] - IRD['GBR']
    IRD['USDMXN'] = IRD['USA'] - IRD['MEX']
    IRD['USDCLP'] = IRD['USA'] - IRD['CHL']
    IRD['USDDKK'] = IRD['USA'] - IRD['DNK']
    IRD['USDNZD'] = IRD['USA'] - IRD['NZL']
    IRD['USDJPY'] = IRD['USA'] - IRD['JPN']
    IRD['USDAUD'] = IRD['USA'] - IRD['AUS']

    iRTimeSeries = IRD[['USA', 'SWE', 'NOR', 'ZAF', 'RUS', 'CHN', 'IND', 'COL', 'CAN', 'EA19',
                        'ISR', 'CRI', 'POL', 'CHE', 'CZE', 'IDN', 'HUN', 'KOR', 'GBR', 'MEX', 'CHL',
                        'DNK', 'NZL', 'JPN', 'AUS']]
    iRTimeSeries.astype(float).to_sql('iRTimeSeries', conn, if_exists='replace')

    IRD = IRD[['USDSEK', 'USDNOK', 'USDZAR', 'USDRUB', 'USDCNH', 'USDINR', 'USDCOP', 'USDCAD', 'USDEUR', 'USDILS',
               'USDCRC', 'USDPLN', 'USDCHF', 'USDCZK', 'USDIDR', 'USDHUF', 'USDKRW', 'USDGBP', 'USDMXN', 'USDCLP',
               'USDDKK',
               'USDNZD', 'USDJPY', 'USDAUD']]
    IRD.astype(float).to_sql('IRD', conn, if_exists='replace')

    IRD.plot()
    plt.show()

def DataSelect(mode):
    if mode == 'retsIRDsSetup':
        dfInvesting = pd.read_sql('SELECT * FROM FxDataInvestingCom', conn).set_index('Dates', drop=True)
        dfIRD = pd.read_sql('SELECT * FROM IRD', conn).rename(columns={"index": "Dates"}).set_index('Dates').loc[
                dfInvesting.index, :].ffill()

        fxRets = sl.dlog(dfInvesting)
        fxIRD = fxRets + dfIRD

        fxRets.fillna(0).to_sql('dataRaw', conn, if_exists='replace')
        fxIRD.fillna(0).to_sql('dataAdj', conn, if_exists='replace')

    elif mode == 'retsIRDs':
        dfRaw = pd.read_sql('SELECT * FROM dataRaw', conn).set_index('Dates', drop=True)
        dfAdj = pd.read_sql('SELECT * FROM dataAdj', conn).set_index('Dates', drop=True)
        print(dfAdj.columns)

        fig, axes = plt.subplots(nrows=2, ncols=1)

        sl.cs(dfAdj).plot(ax=axes[0], title='Data Adjusted with IRDs')
        sl.cs(dfRaw).plot(ax=axes[1], title='Data Raw')
        plt.show()

def RunRollManifoldOnFXPairs():
    dfAdj = pd.read_sql('SELECT * FROM dataAdj', conn).set_index('Dates', drop=True)  # .iloc[1000:1100,:]

    out = sl.AI.gRollingManifold('DMAPS', dfAdj, 50, 5, [0, 1, 2, 3, 4], contractiveObserver=1, DMAPS_sigma='bgh')

    out[0].to_sql('RollDMAPSdf0', conn, if_exists='replace')
    out[1].to_sql('RollDMAPSpsi', conn, if_exists='replace')
    out[2].to_sql('RollDMAPScObserverDF', conn, if_exists='replace')
    out[3].to_sql('RollDMAPSsigmaDF', conn, if_exists='replace')
    out[4].to_sql('RollDMAPSlambdasDF', conn, if_exists='replace')
    glAs = out[5]
    pd.DataFrame(glAs[0], index=out[1].index).to_sql('RollDMAPSComps0', conn, if_exists='replace')
    pd.DataFrame(glAs[1], index=out[1].index).to_sql('RollDMAPSComps1', conn, if_exists='replace')
    pd.DataFrame(glAs[2], index=out[1].index).to_sql('RollDMAPSComps2', conn, if_exists='replace')
    pd.DataFrame(glAs[3], index=out[1].index).to_sql('RollDMAPSComps3', conn, if_exists='replace')
    pd.DataFrame(glAs[4], index=out[1].index).to_sql('RollDMAPSComps4', conn, if_exists='replace')
    eigCoeffsDF = out[6]
    pd.DataFrame(eigCoeffsDF[0], index=out[1].index, columns=dfAdj.columns).to_sql('RollDMAPSeigCoeffsDF0', conn,
                                                                                   if_exists='replace')
    pd.DataFrame(eigCoeffsDF[1], index=out[1].index, columns=dfAdj.columns).to_sql('RollDMAPSeigCoeffsDF1', conn,
                                                                                   if_exists='replace')
    pd.DataFrame(eigCoeffsDF[2], index=out[1].index, columns=dfAdj.columns).to_sql('RollDMAPSeigCoeffsDF2', conn,
                                                                                   if_exists='replace')
    pd.DataFrame(eigCoeffsDF[3], index=out[1].index, columns=dfAdj.columns).to_sql('RollDMAPSeigCoeffsDF3', conn,
                                                                                   if_exists='replace')
    pd.DataFrame(eigCoeffsDF[4], index=out[1].index, columns=dfAdj.columns).to_sql('RollDMAPSeigCoeffsDF4', conn,
                                                                                   if_exists='replace')

def plotData(assetSel):
    if assetSel == 'raw':
        dataRaw = pd.read_sql('SELECT * FROM dataRaw', conn).set_index('Dates', drop=True)
        dataAdj = pd.read_sql('SELECT * FROM dataAdj', conn).set_index('Dates', drop=True)

        fig = plt.figure()
        sl.cs(dataRaw).plot(ax=fig.add_subplot(121), title='FX Pairs Cumulative Returns')
        sl.cs(dataAdj).plot(ax=fig.add_subplot(122), title='Carry Adjusted - FX Pairs Cumulative Returns')
        print(np.sqrt(252) * sl.sharpe(dataRaw).round(4))
        print(np.sqrt(252) * sl.sharpe(dataAdj).round(4))
        plt.show()

        sl.cs(sl.E(dataAdj)).plot(title='Cumulative Returns of the Buy-And-Hold Portfolio of FX Carry Adjusted Returns')
        print((np.sqrt(252) * sl.sharpe(sl.E(dataAdj))).round(4))
        plt.show()

    elif assetSel == 'IRD':
        iRTimeSeries = pd.read_sql('SELECT * FROM iRTimeSeries', conn).set_index('index', drop=True)
        IRD = pd.read_sql('SELECT * FROM IRD', conn).set_index('index', drop=True)

        fig = plt.figure()
        iRTimeSeries.plot(ax=fig.add_subplot(121), title='Interest Rates Daily Time Series')
        IRD.plot(ax=fig.add_subplot(122), title='Interest Rates Differentials Time Series')
        plt.show()

    elif assetSel == 'cObserver':
        df = pd.read_sql('SELECT * FROM dataAdj', conn).set_index('Dates', drop=True)
        psi = pd.read_sql('SELECT * FROM RollDMAPSpsi', conn).set_index('index', drop=True)
        lDF = (-1) * pd.read_sql('SELECT * FROM RollDMAPSlambdasDF', conn).set_index('Dates', drop=True).loc[df.index]
        # lDF.plot()
        # plt.show()
        sigmaF = pd.read_sql('SELECT * FROM RollDMAPSsigmaDF', conn).set_index('Dates', drop=True)
        rollCo = pd.read_sql('SELECT * FROM RollDMAPScObserverDF', conn).set_index('index', drop=True)
        # sl.cs(rollCo).plot(title='rollCo')
        # plt.show()

        pr = 4
        eigCoeffsDF = \
        pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF' + str(pr), conn).set_index('index', drop=True).loc[df.index]
        pnl = sl.S(eigCoeffsDF) * df

        # eigCoeffsDF.plot()
        fig = plt.figure()
        sl.cs(pnl).plot(ax=fig.add_subplot(121), title='Linear Regression EigCoeffsDF Portfolios')
        sl.cs(sl.E(pnl)).plot(ax=fig.add_subplot(122), title='Linear Regression EigCoeffsDF EWP')
        plt.show()


def CustomRun(runScript):
    if runScript == 0:
        kDM0 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio0', conn).set_index('index', drop=True)
        kDM1 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio1', conn).set_index('index', drop=True)
        kDM2 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio2', conn).set_index('index', drop=True)
        kDM3 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio3', conn).set_index('index', drop=True)
        kDM4 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio4', conn).set_index('index', drop=True)

        sHall = pd.concat([(np.sqrt(252) * sl.sharpe(kDM0)).round(4).T, (np.sqrt(252) * sl.sharpe(kDM1)).round(4).T,
                           (np.sqrt(252) * sl.sharpe(kDM2)).round(4).T,
                           (np.sqrt(252) * sl.sharpe(kDM3)).round(4).T, (np.sqrt(252) * sl.sharpe(kDM4)).round(4).T],
                          axis=1).reset_index()

        sHall.astype(str).agg(' & '.join, axis=1).to_csv('allKDmSharpes.csv', index=False)

        kDMall = pd.concat([sl.rs(kDM0), sl.rs(kDM1), sl.rs(kDM2), sl.rs(kDM3), sl.rs(kDM4)], axis=1, ignore_index=True)
        print((np.sqrt(252) * sl.sharpe(kDMall)).round(4))
        print((np.sqrt(252) * sl.sharpe(sl.rs(kDMall))).round(4))

    elif runScript == 1:
        pass


# DataHandler('investingCom')
# shortTermInterestRatesSetup()

# DataSelect('retsIRDsSetup')
# DataSelect('retsIRDs')

# RunRollManifoldOnFXPairs()
# plotData('raw')
# plotData('IRD')
plotData('cObserver')

#CustomRun()
