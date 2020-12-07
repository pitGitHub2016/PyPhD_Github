from Slider import Slider as sl
import numpy as np, investpy
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

conn = sqlite3.connect('FXeodData.db')

def DataHandler(mode):
    if mode == 'investingCom':
        dataAll = []
        fxPairsList = ['USD/EUR', 'USD/GBP', 'USD/AUD', 'USD/NZD', 'USD/JPY', 'USD/CAD','USD/CHF','USD/SEK','USD/NOK', 'USD/DKK',
                       'USD/ZAR', 'USD/RUB', 'USD/PLN', 'USD/MXN', 'USD/CNH', 'USD/KRW', 'USD/INR', 'USD/IDR', 'USD/HUF', 'USD/COP']
        namesDF = pd.DataFrame(fxPairsList, columns=["Names"])
        namesDF['Names'] = namesDF['Names'].str.replace('/', '')
        namesDF.to_sql('BasketGekko', conn, if_exists='replace')
        for fx in fxPairsList:
            print(fx)
            name = fx.replace('/', '')
            df = investpy.get_currency_cross_historical_data(currency_cross=fx, from_date='01/01/2000',
                                                             to_date='29/10/2020').reset_index().rename(
                columns={"Date": "Dates", "Close": name}).set_index('Dates')[name]
            dataAll.append(df)
        pd.concat(dataAll, axis=1).to_sql('UsdBasedPairs', conn, if_exists='replace')
    elif mode == 'investingCom_Invert':
        df = pd.read_sql('SELECT * FROM UsdBasedPairs', conn).set_index('Dates', drop=True)
        df.columns = [x.replace('USD', '') + 'USD' for x in df.columns]
        df = df.apply(lambda x : 1/x)
        df.to_sql('FxDataRaw', conn, if_exists='replace')
    else:
        pd.read_csv('BasketGekko.csv', delimiter=' ').to_sql('BasketGekko', conn, if_exists='replace')
        P = pd.read_csv('P.csv', header=None)
        P.columns = pd.read_csv('Basket.csv', delimiter=' ')['Names'].tolist()
        P['Dates'] = pd.read_csv('Dates.csv', header=None)
        P['Dates'] = pd.to_datetime(P['Dates'], infer_datetime_format=True)
        P = P.set_index('Dates', drop=True)
        P.to_sql('FxData', conn, if_exists='replace')

def shortTermInterestRatesSetup(mode):
    if mode == 'MainSetup':
        # df = pd.read_csv('shortTermInterestRates.csv')
        # https://www.oecd-ilibrary.org/finance-and-investment/short-term-interest-rates/indicator/english_2cc37d77-en?parentId=http%3A%2F%2Finstance.metastore.ingenta.com%2Fcontent%2Fthematicgrouping%2F86b91cb3-en
        df = pd.read_csv('shortTermInterestRates_29102020.csv')
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

        IRD['SEKUSD'] = IRD['SWE'] - IRD['USA']
        IRD['NOKUSD'] = IRD['NOR'] - IRD['USA']
        IRD['ZARUSD'] = IRD['ZAF'] - IRD['USA']
        IRD['RUBUSD'] = IRD['RUS'] - IRD['USA']
        IRD['CNHUSD'] = IRD['CHN'] - IRD['USA']
        IRD['INRUSD'] = IRD['IND'] - IRD['USA']
        IRD['COPUSD'] = IRD['COL'] - IRD['USA']
        IRD['CADUSD'] = IRD['CAN'] - IRD['USA']
        IRD['EURUSD'] = IRD['EA19'] - IRD['USA']
        IRD['PLNUSD'] = IRD['POL'] - IRD['USA']
        IRD['CHFUSD'] = IRD['CHE'] - IRD['USA']
        IRD['IDRUSD'] = IRD['IDN'] - IRD['USA']
        IRD['HUFUSD'] = IRD['HUN'] - IRD['USA']
        IRD['KRWUSD'] = IRD['KOR'] - IRD['USA']
        IRD['GBPUSD'] = IRD['GBR'] - IRD['USA']
        IRD['MXNUSD'] = IRD['MEX'] - IRD['USA']
        IRD['DKKUSD'] = IRD['DNK'] - IRD['USA']
        IRD['NZDUSD'] = IRD['NZL'] - IRD['USA']
        IRD['JPYUSD'] = IRD['JPN'] - IRD['USA']
        IRD['AUDUSD'] = IRD['AUS'] - IRD['USA']

        iRTimeSeries = IRD[['USA', 'SWE', 'NOR', 'ZAF', 'RUS', 'CHN', 'IND', 'COL', 'CAN', 'EA19',
                            'POL', 'CHE', 'IDN', 'HUN', 'KOR', 'GBR', 'MEX', 'DNK', 'NZL', 'JPN', 'AUS']]
        iRTimeSeries.astype(float).to_sql('iRTimeSeries', conn, if_exists='replace')

        IRD = IRD[['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'JPYUSD', 'CADUSD','CHFUSD','SEKUSD','NOKUSD', 'DKKUSD',
                       'ZARUSD', 'RUBUSD', 'PLNUSD', 'MXNUSD', 'CNHUSD', 'KRWUSD', 'INRUSD', 'IDRUSD', 'HUFUSD', 'COPUSD']]
        IRD = IRD.iloc[5389:,:]
        IRD.astype(float).to_sql('IRD', conn, if_exists='replace')

        fig, ax = plt.subplots()
        IRD.plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        mpl.pyplot.ylabel("Interest Rates Differentials")
        mpl.pyplot.xlabel("Dates")
        plt.show()
    elif mode == 'retsIRDsSetup':
        dfInvesting = pd.read_sql('SELECT * FROM FxDataRaw', conn).set_index('Dates', drop=True)
        dfIRD = pd.read_sql('SELECT * FROM IRD', conn).rename(columns={"index": "Dates"}).set_index('Dates').loc[
                dfInvesting.index, :].ffill()

        fxRets = sl.dlog(dfInvesting)
        fxIRD = fxRets + dfIRD

        fxRets.fillna(0).to_sql('FxDataRawRets', conn, if_exists='replace')
        fxIRD.fillna(0).to_sql('FxDataAdjRets', conn, if_exists='replace')
    elif mode == 'retsIRDs':
        dfRaw = pd.read_sql('SELECT * FROM FxDataRawRets', conn).set_index('Dates', drop=True)
        dfAdj = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        print(dfAdj.columns)

        fig, ax = plt.subplots()
        sl.cs(dfRaw).plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        #mpl.pyplot.ylabel("Unadjusted FX Cumulative Returns")
        mpl.pyplot.ylabel("Unadjsuted (Raw) FX Cumulative Returns")
        mpl.pyplot.xlabel("Dates")
        plt.show()

        fig, ax = plt.subplots()
        sl.cs(dfAdj).plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        # mpl.pyplot.ylabel("Unadjusted FX Cumulative Returns")
        mpl.pyplot.ylabel("Carry adjusted FX Cumulative Returns")
        mpl.pyplot.xlabel("Dates")
        plt.show()

def LongOnly():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    longOnlySharpes["Sharpe"] = "& " + longOnlySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    rsDf = pd.DataFrame(sl.rs(df))
    rsDf.to_sql('LongOnlyEWPrsDf', conn, if_exists='replace')
    print('LongOnly rsDf')
    print((np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    pnl3 = sl.S(sl.sign(sl.ema(rsDf, nperiods=3))) * rsDf
    pnl5 = sl.S(sl.sign(sl.ema(rsDf, nperiods=5))) * rsDf
    pnl25 = sl.S(sl.sign(sl.ema(rsDf, nperiods=25))) * rsDf
    pnl50 = sl.S(sl.sign(sl.ema(rsDf, nperiods=50))) * rsDf
    pnl250 = sl.S(sl.sign(sl.ema(rsDf, nperiods=250))) * rsDf
    pnlSharpes3 = np.sqrt(252) * sl.sharpe(pnl3).round(4)
    pnlSharpes5 = np.sqrt(252) * sl.sharpe(pnl5).round(4)
    pnlSharpes25 = np.sqrt(252) * sl.sharpe(pnl25).round(4)
    pnlSharpes50 = np.sqrt(252) * sl.sharpe(pnl50).round(4)
    pnlSharpes250 = np.sqrt(252) * sl.sharpe(pnl250).round(4)
    print(pnlSharpes3)
    print(pnlSharpes5)
    print(pnlSharpes25)
    print(pnlSharpes50)
    print(pnlSharpes250)

    fig, ax = plt.subplots()
    sl.cs(pnl3).plot(ax=fig.add_subplot(221), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(a)")
    sl.cs(pnl5).plot(ax=fig.add_subplot(222), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(b)")
    sl.cs(pnl25).plot(ax=fig.add_subplot(223), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(c)")
    sl.cs(pnl50).plot(ax=fig.add_subplot(224), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(d)")
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    csdf = sl.cs(df)
    csdf.index = [x.replace("00:00:00", "").strip() for x in csdf.index]
    fig, ax = plt.subplots()
    csdf.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14},  borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    csrsDf = sl.cs(rsDf)
    csrsDf.index = [x.replace("00:00:00", "").strip() for x in csrsDf.index]
    csrsDf.plot(ax=ax, legend=False) #title='Equally Weighted Portfolio'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    plt.show()

def RiskParity():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    SRollVol = np.sqrt(252) * sl.S(sl.rollerVol(df, 250)) * 100
    SRollVolToPlot = SRollVol.copy()
    SRollVolToPlot.index = [x.replace("00:00:00", "").strip() for x in SRollVolToPlot.index]
    SRollVol.to_sql('SRollVol', conn, if_exists='replace')

    df = (df / SRollVol).replace(np.inf, 0)
    df.to_sql('riskParityDF', conn, if_exists='replace')
    riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    riskParitySharpes["Sharpe"] = "& " + riskParitySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    print(riskParitySharpes)
    rsDf = pd.DataFrame(sl.rs(df))
    riskParitySharpes.to_sql('riskParitySharpeRatios', conn, if_exists='replace')
    rsDf.to_sql('RiskParityEWPrsDf', conn, if_exists='replace')
    print((np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    pnl3 = sl.S(sl.sign(sl.ema(rsDf, nperiods=3))) * rsDf
    pnl5 = sl.S(sl.sign(sl.ema(rsDf, nperiods=5))) * rsDf
    pnl25 = sl.S(sl.sign(sl.ema(rsDf, nperiods=25))) * rsDf
    pnl50 = sl.S(sl.sign(sl.ema(rsDf, nperiods=50))) * rsDf
    pnl250 = sl.S(sl.sign(sl.ema(rsDf, nperiods=250))) * rsDf
    pnlSharpes3 = np.sqrt(252) * sl.sharpe(pnl3).round(4)
    pnlSharpes5 = np.sqrt(252) * sl.sharpe(pnl5).round(4)
    pnlSharpes25 = np.sqrt(252) * sl.sharpe(pnl25).round(4)
    pnlSharpes50 = np.sqrt(252) * sl.sharpe(pnl50).round(4)
    pnlSharpes250 = np.sqrt(252) * sl.sharpe(pnl250).round(4)
    print(pnlSharpes3)
    print(pnlSharpes5)
    print(pnlSharpes25)
    print(pnlSharpes50)
    print(pnlSharpes250)

    csdf = sl.cs(df)
    csdf.index = [x.replace("00:00:00", "").strip() for x in csdf.index]
    csrsDf = sl.cs(rsDf)
    csrsDf.index = [x.replace("00:00:00", "").strip() for x in csrsDf.index]

    fig, ax = plt.subplots()
    SRollVolToPlot.iloc[51:,:].plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Annualised Rolling Volatilities (%)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    csdf.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    csrsDf.plot(ax=ax, legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.show()

def RunRollManifoldOnFXPairs(manifoldIn):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)

    if manifoldIn in ['PCA', 'LLE']:
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0, 1, 2, 3, 4], RollMode='ExpWindow', Scaler='Standard') #ExpWindow
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0, 1, 2, 3, 4], LLE_n_neighbors=5, ProjectionMode='Transpose',
                                         RollMode='ExpWindow', Scaler='Standard')

        out[0].to_sql('df', conn, if_exists='replace')
        principalCompsDfList = out[1]; exPostProjectionsList = out[2]
        out[3].to_sql(manifoldIn+'_sigmasDf', conn, if_exists='replace')
        out[4].to_sql(manifoldIn+'_lambdasDf', conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn+'_principalCompsDf_'+str(k), conn, if_exists='replace')
            exPostProjectionsList[k].to_sql(manifoldIn+'_exPostProjections_'+str(k), conn, if_exists='replace')

    elif manifoldIn in ['DMAP']:

        out = sl.AI.gRollingManifold('DMAPS', df, 25, 5, [0, 1, 2, 3, 4], contractiveObserver=1, DMAPS_sigma='bgh', RollMode='ExpWindow')

        out[0].to_sql('RollDMAPSdf', conn, if_exists='replace')
        out[1].to_sql('RollDMAPSpsi', conn, if_exists='replace')
        out[2].to_sql('RollDMAPScObserverDF', conn, if_exists='replace')
        out[3].to_sql('RollDMAPSsigmaDF', conn, if_exists='replace')
        out[4].to_sql('RollDMAPSlambdasDF', conn, if_exists='replace')
        glAs = out[5]
        pd.DataFrame(glAs[1], index=out[1].index).to_sql('RollDMAPSComps1', conn, if_exists='replace')
        pd.DataFrame(glAs[2], index=out[1].index).to_sql('RollDMAPSComps2', conn, if_exists='replace')
        pd.DataFrame(glAs[3], index=out[1].index).to_sql('RollDMAPSComps3', conn, if_exists='replace')
        pd.DataFrame(glAs[4], index=out[1].index).to_sql('RollDMAPSComps4', conn, if_exists='replace')
        eigCoeffsDF = out[6]
        for k in range(glAs):
            pd.DataFrame(glAs[k], index=out[1].index).to_sql('RollDMAPSComps'+str(k), conn, if_exists='replace')
            pd.DataFrame(eigCoeffsDF[0], index=out[1].index, columns=df.columns).to_sql(manifoldIn+'_principalCompsDf_'+str(k), conn,
                                                                                    if_exists='replace')
def CorrMatrix():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    mat = sl.CorrMatrix(df)
    mat.to_sql('CorrMatrix', conn, if_exists='replace')
    print(mat[mat!=1].abs().mean().mean())
    # sl.correlation_matrix_plot(df)

    dfRollingCorr = df['EURUSD'].rolling(25).corr(df, pairwise=True)
    dfRollingCorr = dfRollingCorr.abs()
    #dfRollingCorr.to_sql('RollingCorrelations25', conn, if_exists='replace')
    fig, ax = plt.subplots()
    dfRollingCorr.plot(ax=ax)  # title=manifoldIn+' Projections'
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    #mpl.pyplot.ylabel("Cumulative Return")
    plt.show()

def getProjections(mode):
    rng = [0, 1, 2, 3, 4]
    allProjectionsPCA = []
    #PCAls = pd.read_sql('SELECT * FROM PCA_lambdasDf', conn).set_index('Dates', drop=True)
    for pr in rng:
        # PCA
        PCArs = pd.DataFrame(
            sl.rs(pd.read_sql('SELECT * FROM PCA_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)))
        PCArs.columns = ['PCA' + str(pr)]
    #    medls = pd.DataFrame(PCAls.iloc[:, pr])
    #    medls.columns = ['PCA' + str(pr)]
        allProjectionsPCA.append(PCArs)
    PCAdf = pd.concat(allProjectionsPCA, axis=1)

    allProjectionsLLE = []
    for pr in rng:
        try:
            # LLE
            LLErs = pd.DataFrame(sl.rs(
                pd.read_sql('SELECT * FROM LLE_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)))
            # LLErs = pd.DataFrame(pd.read_sql('SELECT * FROM LLE_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True))
            LLErs.columns = ['LLE' + str(pr)]
            allProjectionsLLE.append(LLErs)
        except Exception as e:
            print(e)
    LLEdf = pd.concat(allProjectionsLLE, axis=1)

    allProjectionsDF = pd.concat([PCAdf, LLEdf], axis=1)

    allProjectionsDF['LongOnlyEWPrsDf'] = sl.rs(pd.read_sql('SELECT * FROM LongOnlyEWPrsDf', conn).set_index('Dates', drop=True))
    allProjectionsDF['RiskParityPortfolio'] = sl.rs(pd.read_sql('SELECT * FROM riskParityDF', conn).set_index('Dates', drop=True))

    if mode == 'RV':
        allProjectionsDF = sl.RV(allProjectionsDF)
    elif mode == 'RVPriceRatio':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="priceRatio")
    elif mode == 'RVExpCorr':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="ExpCorr")
    elif mode == 'RVRollHedgeRatio':
        allProjectionsDF = sl.RV(allProjectionsDF, mode="RollHedgeRatio")
    elif mode == 'Baskets':
        allProjectionsDF = sl.Baskets(allProjectionsDF)

    return allProjectionsDF

def semaOnProjections():
    allProjectionsDF = getProjections('')

    #allProjectionsDFPCA = allProjectionsDF[[x for x in allProjectionsDF.columns if 'PCA' in x]]
    #allProjectionsDFLLE = allProjectionsDF[[x for x in allProjectionsDF.columns if 'LLE' in x]]
    #allProjectionsDF = pd.concat([sl.rs(allProjectionsDFPCA), sl.rs(allProjectionsDFLLE)], axis=1)
    #allProjectionsDF.columns = ["PCA", "LLE"]

    allProjectionsDFSharpes = np.sqrt(252) * sl.sharpe(allProjectionsDF).round(4)
    allProjectionsDFSharpes.to_sql('allProjectionsDFSharpes', conn, if_exists='replace')

    print('semaPnLSharpe #############################################################')
    pnl3 = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=3))) * allProjectionsDF
    pnl3.to_sql('semapnl3', conn, if_exists='replace')
    pnl5 = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=5))) * allProjectionsDF
    pnl5.to_sql('semapnl5', conn, if_exists='replace')
    pnl25 = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=25))) * allProjectionsDF
    pnl25.to_sql('semapnl25', conn, if_exists='replace')
    pnl50 = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=50))) * allProjectionsDF
    pnl50.to_sql('semapnl50', conn, if_exists='replace')
    pnl250 = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=250))) * allProjectionsDF
    pnl250.to_sql('semapnl250', conn, if_exists='replace')
    pnlSharpes3 = np.sqrt(252) * sl.sharpe(pnl3).round(4)
    pnlSharpes5 = np.sqrt(252) * sl.sharpe(pnl5).round(4)
    pnlSharpes25 = np.sqrt(252) * sl.sharpe(pnl25).round(4)
    pnlSharpes50 = np.sqrt(252) * sl.sharpe(pnl50).round(4)
    pnlSharpes250 = np.sqrt(252) * sl.sharpe(pnl250).round(4)
    pnlSharpes3.to_sql('semapnlSharpes3', conn, if_exists='replace')
    pnlSharpes5.to_sql('semapnlSharpes5', conn, if_exists='replace')
    pnlSharpes25.to_sql('semapnlSharpes25', conn, if_exists='replace')
    pnlSharpes50.to_sql('semapnlSharpes50', conn, if_exists='replace')
    pnlSharpes250.to_sql('semapnlSharpes250', conn, if_exists='replace')
    print("pnlSharpes3_Best = ", pnlSharpes3.abs().max())
    print("pnlSharpes5_Best = ", pnlSharpes5.abs().max())
    print("pnlSharpes25_Best = ", pnlSharpes25.abs().max())
    print("pnlSharpes50_Best = ", pnlSharpes50.abs().max())
    print("pnlSharpes250_Best = ", pnlSharpes250.abs().max())

def StationarityOnProjections(manifoldIn, mode):
    allProjections = []
    for pr in range(5):
        exPostProjections = pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)
        allProjections.append(sl.rs(exPostProjections))
    allProjectionsDF = pd.concat(allProjections, axis=1, ignore_index=True)
    allProjectionsDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    if mode == 'build':

        out = sl.Stationarity(allProjectionsDF, 25, 'exp', multi=1)
        out[0].to_sql('ADF_Test_'+manifoldIn, conn, if_exists='replace')
        out[1].to_sql('Pval_Test_'+manifoldIn, conn, if_exists='replace')
        out[2].to_sql('critVal_Test_'+manifoldIn, conn, if_exists='replace')

    elif mode == 'filter':
        #adf = pd.read_sql('SELECT * FROM Pval_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf.columns = ['$y_{s'+manifoldIn+',(1,t)}$','$y_{s'+manifoldIn+',(2,t)}$','$y_{s'+manifoldIn+',(3,t)}$','$y_{s'+manifoldIn+',(4,t)}$','$y_{s'+manifoldIn+',(5,t)}$']
        #ylbl = "ADF Test p-Values"
        #adf = pd.read_sql('SELECT * FROM critVal_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf.columns = [x.replace('P0_', '$y_{'+manifoldIn+',(1,t)}$').replace('P1_', '$y_{'+manifoldIn+',(2,t)}$').replace('P2_', '$y_{'+manifoldIn+',(3,t)}$').replace('P3_', '$y_{'+manifoldIn+',(4,t)}$').replace('P4_', '$y_{'+manifoldIn+',(5,t)}$') for x in adf.columns]
        #ylbl = "Critical Values"
        adf = pd.read_sql('SELECT * FROM ADF_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        adf.columns = ['$y_{s'+manifoldIn+',(1,t)}$','$y_{s'+manifoldIn+',(2,t)}$','$y_{s'+manifoldIn+',(3,t)}$','$y_{s'+manifoldIn+',(4,t)}$','$y_{s'+manifoldIn+',(5,t)}$']
        ylbl = "ADF Test : $DF_T$"

        #stationaryDF = adf[adf < 0.05]
        #print("% of Stationary points relative to the entire dataset = ", (len(stationaryDF.dropna()) / len(adf)) * 100, " %")

        adf.index = [x.replace("00:00:00", "").strip() for x in adf.index]

        fig, ax = plt.subplots()
        #stationaryDF.plot(ax=ax)
        adf.iloc[500:].plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        mpl.pyplot.ylabel(ylbl)
        if ylbl == "Critical Values":
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 16}, frameon=False, borderaxespad=0.)
        else:
            plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
        plt.show()

def ProjectionsPlots(manifoldIn):
    list = []
    for c in range(5):
        list.append(sl.rs(pd.read_sql('SELECT * FROM '+manifoldIn+'_exPostProjections_'+str(c), conn).set_index('Dates', drop=True).fillna(0)))
    exPostProjections = pd.concat(list, axis=1, ignore_index=True)
    #exPostProjections.columns = ['$\Pi_{'+manifoldIn+',1,t}$','$\Pi_{'+manifoldIn+',2,t}$','$\Pi_{'+manifoldIn+',3,t}$','$\Pi_{'+manifoldIn+',4,t}$','$\Pi_{'+manifoldIn+',5,t}$',]
    exPostProjections.to_sql(manifoldIn + '_RsExPostProjections', conn, if_exists='replace')

    exPostProjections.index = [x.replace("00:00:00", "").strip() for x in exPostProjections.index]

    fig, ax = plt.subplots()
    sl.cs(exPostProjections).plot(ax=ax) #title=manifoldIn+' Projections'
    #sl.cs(rsDf).plot(title='Equally Weighted Portfolio')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    mpl.pyplot.ylabel("Cumulative Return")
    plt.show()

    rsProjection = sl.cs(sl.rs(exPostProjections))
    rsProjection.name = '$Y_(s'+manifoldIn+')(t)$'
    fig, ax = plt.subplots()
    rsProjection.plot(ax=ax) #title=manifoldIn+' Projections'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    mpl.pyplot.ylabel("Cumulative Return")
    plt.show()

def ARIMAonProjections(mode):

    allProjectionsDF = getProjections('') #100, 300, 500
    # allProjectionsDF = getProjections('RV')
    # allProjectionsDF = getProjections('RVExpCorr')
    #allProjectionsDF = getProjections('RVRollHedgeRatio')
    # allProjectionsDF = getProjections('Baskets')

    allProjectionsDFPCA = allProjectionsDF[[x for x in allProjectionsDF.columns if 'PCA' in x]]
    allProjectionsDFLLE = allProjectionsDF[[x for x in allProjectionsDF.columns if 'LLE' in x]]
    allProjectionsDF = pd.concat([sl.rs(allProjectionsDFPCA), sl.rs(allProjectionsDFLLE)], axis=1)
    allProjectionsDF.columns = ["PCA", "LLE"]

    orderIn = (5, 0, 0)
    if mode == "run":
        for selection in allProjectionsDF.columns:
            try:
                print(selection)
                Arima_Results = sl.ARIMA_Walk(allProjectionsDF[selection], 0.3, orderIn)

                Arima_Results[0].to_sql(selection + '_ARIMA_testDF_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')
                Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

                sig = sl.sign(Arima_Results[1])

                pnl = sig * Arima_Results[0]
                pnl.to_sql(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

                print("ARIMA ("+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+") Sharpe = ", np.sqrt(252) * sl.sharpe(pnl))
            except Exception as e:
                print("selection = ", selection, ", error : ", e)

    elif mode == "report":
        shList = []
        for selection in allProjectionsDF.columns:
            pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn).set_index('Dates', drop=True)
            medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
            shList.append([selection, medSh])
        shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
        print(shDF)
        shDF.to_sql('sh_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

def ContributionAnalysis():
    allProjectionsDF = getProjections('')

    allProjectionsDFPCA = allProjectionsDF[[x for x in allProjectionsDF.columns if 'PCA' in x]]
    allProjectionsDFLLE = allProjectionsDF[[x for x in allProjectionsDF.columns if 'LLE' in x]]
    allProjectionsDF = pd.concat([sl.rs(allProjectionsDFPCA), sl.rs(allProjectionsDFLLE)], axis=1)
    allProjectionsDF.columns = ["PCA", "LLE"]

    allProjectionsDF = sl.ExPostOpt(allProjectionsDF)[0]

    #mainPortfolio = sl.rs(pd.read_sql('SELECT * FROM LongOnlyEWPrsDf', conn).set_index('Dates', drop=True))
    #allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'LongOnlyEWPrsDf' not in x]]
    mainPortfolio = sl.rs(pd.read_sql('SELECT * FROM riskParityDF', conn).set_index('Dates', drop=True))
    allProjectionsDF = allProjectionsDF[[x for x in allProjectionsDF.columns if 'RiskParityPortfolio' not in x]]

    mainPortfolioSharpe = np.sqrt(252) * sl.sharpe(mainPortfolio).round(4)
    print("mainPortfolioSharpe = ", mainPortfolioSharpe)

    for subPortfolio in allProjectionsDF:
        try:
            ### RAW PROJECTIONS ###
            contribution = allProjectionsDF[subPortfolio]
            ### SEMA ###
            #contribution = pd.read_sql('SELECT * FROM semapnl3', conn).set_index('Dates', drop=True)
            #contribution = pd.read_sql('SELECT * FROM semapnl5', conn).set_index('Dates', drop=True)
            #contribution = pd.read_sql('SELECT * FROM semapnl25', conn).set_index('Dates', drop=True)
            #contribution = pd.read_sql('SELECT * FROM semapnl50', conn).set_index('Dates', drop=True)
            #contribution = pd.read_sql('SELECT * FROM semapnl250', conn).set_index('Dates', drop=True)
            #contribution = contribution[subPortfolio]
            #contribution = sl.ExPostOpt(contribution)[0][subPortfolio]

            ### ARIMA ###
            orderIn = (1, 0, 0)
            #contribution = pd.read_sql('SELECT * FROM ' + subPortfolio + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn).set_index('Dates', drop=True)
            ### RNN ###
            #contribution = pd.read_sql('SELECT * FROM pnl_RNN_'+subPortfolio+'0', conn).set_index('Dates', drop=True)

            totalPortfolioDF = pd.concat([mainPortfolio, contribution], axis=1)
            totalPortfolioDF.columns = ["mainPortfolio", "contribution"]
            totalPortfolioDF['stdRatio'] = 1
            #stdRolling = sl.S(sl.expanderVol(totalPortfolioDF, 25)).fillna(1)
            #totalPortfolioDF['stdRatio'] = stdRolling['mainPortfolio'] / stdRolling['contribution']
            #totalPortfolioDF['stdRatio'] = 1 / totalPortfolioDF['stdRatio']

            totalPortfolio = totalPortfolioDF['mainPortfolio'] + totalPortfolioDF['stdRatio'] * totalPortfolioDF['contribution']

            shrstotalPortfolio = np.sqrt(252) * sl.sharpe(totalPortfolio).round(4)
            print("subPortfolio = ", subPortfolio, ", shrstotalPortfolio = ", round(shrstotalPortfolio,4), " (", round(100*((shrstotalPortfolio-mainPortfolioSharpe)/mainPortfolioSharpe), 1), "\%) \\\\")
            #print("totalPortfolioDF['mainPortfolio'].std() = ", totalPortfolioDF['mainPortfolio'].std(), ", totalPortfolioDF['contribution'].std() = ", totalPortfolioDF['contribution'].std())
            #print("totalPortfolioDF['mainPortfolio'] * totalPortfolioDF['stdRatio'] ___ std =", (totalPortfolioDF['mainPortfolio'] * totalPortfolioDF['stdRatio']).std())
            #print(totalPortfolioDF['stdRatio'].iloc[-1])

        except Exception as e:
            print(e)

def FinalModelPlot():
    pnl = pd.concat([
        pd.read_sql('SELECT * FROM semapnl3', conn).set_index('Dates', drop=True)["PCA2"] * (-1),
        pd.read_sql('SELECT * FROM PCA2_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM PCA2_ARIMA_pnl_300', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM PCA2_ARIMA_pnl_500', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE4_ARIMA_pnl_300', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE_ARIMA_pnl_300', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE_ARIMA_pnl_500', conn).set_index('Dates', drop=True)
    ], axis=1).dropna()
    pnl.columns = ["${y}_{3,sPCA}$-EMA(3)", "${y}_{3,sPCA}$-AR(1)", "${y}_{3,sPCA}$-AR(3)",
   "${y}_{3,sPCA}$-AR(5)", "${y}_{5,sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(1)", "${Y}_{sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(5)"]

    #contribPnl = pnl
    #contribPnl = sl.rs(pnl[["${y}_{3,sPCA}$-AR(1)", "${y}_{5,sLLE}$-AR(3)"]])
    contribPnl = sl.rs(pnl[["${y}_{3,sPCA}$-AR(1)", "${y}_{5,sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(5)"]])
    #contribPnl = sl.RV(pnl[["${y}_{3,sPCA}$-AR(1)", "${y}_{5,sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(5)"]], mode='Baskets')
    #contribPnl = sl.rs(pnl)
    print("contribPnl = ", np.sqrt(252) * sl.sharpe(contribPnl).round(4))

    #fig, ax = plt.subplots()
    #sl.cs(pnl).plot(ax=ax)
    #sl.cs(contribPnl).plot(ax=ax)
    #mpl.pyplot.ylabel("Cumulative Returns")
    #mpl.pyplot.xlabel("Dates")
    #plt.show()

#DataHandler('investingCom')
#DataHandler('investingCom_Invert')
#shortTermInterestRatesSetup("MainSetup")
#shortTermInterestRatesSetup("retsIRDsSetup")
#shortTermInterestRatesSetup("retsIRDs")

#LongOnly()
#RiskParity()

#RunRollManifoldOnFXPairs('PCA')
#RunRollManifoldOnFXPairs('LLE')

#ProjectionsPlots('PCA')
#ProjectionsPlots('LLE')

#CorrMatrix()

#semaOnProjections()

#StationarityOnProjections('PCA', 'build')
#StationarityOnProjections('LLE', 'build')
#StationarityOnProjections('PCA', 'filter')
#StationarityOnProjections('LLE', 'filter')

#ARIMAonProjections("")
#ARIMAonProjections("run")
#ARIMAonProjections("report")

#ContributionAnalysis()

FinalModelPlot()
