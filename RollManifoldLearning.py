from Slider import Slider as sl
import numpy as np, investpy, time
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
#mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250]

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
        df.plot()
        plt.show()

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

        IRD.index = [x.replace("00:00:00", "").strip() for x in IRD.index]

        fig, ax = plt.subplots()
        IRD.plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        mpl.pyplot.ylabel("IRD")
        mpl.pyplot.xlabel("Dates")
        plt.show()
    elif mode == 'retsIRDsSetup':
        dfInvesting = pd.read_sql('SELECT * FROM FxDataRaw', conn).set_index('Dates', drop=True)
        dfIRD = pd.read_sql('SELECT * FROM IRD', conn).rename(columns={"index": "Dates"}).set_index('Dates').loc[
                dfInvesting.index, :].ffill()

        fxRets = sl.dlog(dfInvesting, fillna="no")
        fxIRD = fxRets + dfIRD

        fxRets.fillna(0).to_sql('FxDataRawRets', conn, if_exists='replace')
        fxIRD.fillna(0).to_sql('FxDataAdjRets', conn, if_exists='replace')
    elif mode == 'retsIRDs':
        dfRaw = pd.read_sql('SELECT * FROM FxDataRawRets', conn).set_index('Dates', drop=True)
        dfAdj = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        print(dfAdj.columns)

        csdfRaw = sl.cs(dfRaw)
        csdfRaw.index = [x.replace("00:00:00", "").strip() for x in csdfRaw.index]
        csdfAdj = sl.cs(dfAdj)
        csdfAdj.index = [x.replace("00:00:00", "").strip() for x in csdfAdj.index]

        fig, ax = plt.subplots()
        csdfRaw.plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        #mpl.pyplot.ylabel("Unadjusted FX Cumulative Returns")
        mpl.pyplot.ylabel("$r_t$")
        mpl.pyplot.xlabel("Dates")
        plt.show()

        fig, ax = plt.subplots()
        csdfAdj.plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, prop={'size': 14}, frameon=False, borderaxespad=0.)
        # mpl.pyplot.ylabel("Unadjusted FX Cumulative Returns")
        mpl.pyplot.ylabel("$x_t$")
        mpl.pyplot.xlabel("Dates")
        plt.show()

def LongOnly():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    longOnlySharpes.to_sql('LongOnlySharpes', conn, if_exists='replace')
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

def RiskParity(mode):
    if mode == 'run':
        df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        expVol = np.sqrt(252) * sl.S(sl.expanderVol(df, 25)) * 100
        twList.append('ExpWindow25')
        shList = []
        for tw in twList:
            print("Risk Parity tw = ", tw)
            if tw == 'ExpWindow25':
                riskParityVol = expVol
            else:
                riskParityVol = np.sqrt(252) * sl.S(sl.rollerVol(df, tw)) * 100
            riskParityVolToPlot = riskParityVol.copy()
            riskParityVolToPlot.index = [x.replace("00:00:00", "").strip() for x in riskParityVolToPlot.index]
            riskParityVol.to_sql('riskParityVol_tw_'+str(tw), conn, if_exists='replace')

            df = (df / riskParityVol).replace([np.inf, -np.inf], 0)
            df.to_sql('riskParityDF_tw_'+str(tw), conn, if_exists='replace')
            riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
            riskParitySharpes.to_sql('riskParitySharpeRatios_tw_'+str(tw), conn, if_exists='replace')
            rsDf = pd.DataFrame(sl.rs(df))
            rsDf.to_sql('RiskParityEWPrsDf_tw_'+str(tw), conn, if_exists='replace')
            shrsdfRP = (np.sqrt(252) * sl.sharpe(rsDf)).round(4)

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

            shList.append([tw, shrsdfRP.values[0], pnlSharpes3.values[0], pnlSharpes5.values[0], pnlSharpes25.values[0],
                           pnlSharpes50.values[0], pnlSharpes250.values[0]])

        riskParitySharpesDF = pd.DataFrame(shList, columns=["tw", "shrsdfRP", "pnlSharpes3", "pnlSharpes5", "pnlSharpes25",
                           "pnlSharpes50", "pnlSharpes250"])
        riskParitySharpesDF.to_sql('riskParitySharpesDF', conn, if_exists='replace')

    elif mode == 'plot':
        volToPlot0 = pd.read_sql('SELECT * FROM riskParityVol_tw_250', conn).set_index('Dates', drop=True)
        volToPlot0.index = [x.replace("00:00:00", "").strip() for x in volToPlot0.index]
        volToPlot1 = pd.read_sql('SELECT * FROM riskParityVol_tw_ExpWindow25', conn).set_index('Dates', drop=True)
        volToPlot1.index = [x.replace("00:00:00", "").strip() for x in volToPlot1.index]

        fig, ax = plt.subplots()
        volToPlot0.plot(ax=ax, legend=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        mpl.pyplot.ylabel("$\hat{\sigma}_{i,t}$")
        plt.show()

        fig, ax = plt.subplots()
        volToPlot1.plot(ax=ax, legend=False)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        mpl.pyplot.ylabel("$\hat{\sigma}_{i,t}$")
        plt.show()

def RunManifoldLearningOnFXPairs(manifoldIn, mode):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)

    if mode == 'Rolling':
        for tw in twList:
            print(manifoldIn + " tw = ", tw)
            if manifoldIn == 'PCA':
                out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], Scaler='Standard') #RollMode='ExpWindow'
            elif manifoldIn == 'LLE':
                out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], LLE_n_neighbors=5, ProjectionMode='Transpose') # RollMode='ExpWindow'

            out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')
                exPostProjectionsList[k].to_sql(manifoldIn + '_exPostProjections_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

    elif mode == 'Expanding':
        tw = 'ExpWindow25'
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], Scaler='Standard', RollMode='ExpWindow')
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], LLE_n_neighbors=5, ProjectionMode='Transpose', RollMode='ExpWindow')

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        exPostProjectionsList = out[2]
        out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')
            exPostProjectionsList[k].to_sql(manifoldIn + '_exPostProjections_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

def ProjectionsPlots(manifoldIn):
    rsProjectionList = []
    twList.append('ExpWindow25')
    for tw in twList:
        print(manifoldIn + " tw = ", tw)
        list = []
        for c in [0,1,2,3,4]:
            try:
                pr = sl.rs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_tw_'+str(tw) + "_" + str(c), conn).set_index('Dates', drop=True).fillna(0))
                list.append(pr)
            except:
                pass
        exPostProjections = pd.concat(list, axis=1, ignore_index=True)
        exPostProjections.columns = ['$\Pi_{'+manifoldIn+','+str(tw)+',1,t}$','$\Pi_{'+manifoldIn+','+str(tw)+',2,t}$',
                                     '$\Pi_{'+manifoldIn+','+str(tw)+',3,t}$','$\Pi_{'+manifoldIn+','+str(tw)+',4,t}$',
                                     '$\Pi_{'+manifoldIn+','+str(tw)+',5,t}$']

        exPostProjections.to_sql(manifoldIn + '_RsExPostProjections_tw_'+str(tw), conn, if_exists='replace')

        exPostProjections.index = [x.replace("00:00:00", "").strip() for x in exPostProjections.index]
        csExPostProjections = sl.cs(exPostProjections)

        rsProjection = sl.cs(sl.rs(exPostProjections))
        rsProjection.name = '$Y_(s'+manifoldIn+','+str(tw)+')(t)$'
        rsProjectionList.append(rsProjection)

        #fig2, ax2 = plt.subplots()
        #ax2 = plt.axes(projection='3d')
        #ax2.scatter3D(csExPostProjections.iloc[:,0], csExPostProjections.iloc[:,1], csExPostProjections.iloc[:,2], cmap='Greens')
        #ax2.set_xlabel(exPostProjections.columns[0])
        #ax2.set_ylabel(exPostProjections.columns[1])
        #ax2.set_zlabel(exPostProjections.columns[2])
        #fig2.savefig(GraphsFolder + manifoldIn+'_csExPostProjections3D_tw_' + str(tw) + '.png')

    rsProjectionDF = pd.concat(rsProjectionList, axis=1)

    fig, ax = plt.subplots()
    rsProjectionDF.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("Cumulative Return")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

def getProjections():
    twList.append('ExpWindow25')
    rng = [0, 1, 2, 3, 4]
    allProjectionsList = []
    for tw in twList:
        print("getProjections - tw = ", tw)

        allProjectionsPCA = []
        allProjectionsLLE = []
        for pr in rng:
            # PCA
            PCArs = pd.DataFrame(
                sl.rs(pd.read_sql('SELECT * FROM PCA_exPostProjections_tw_' +str(tw) + "_" + str(pr), conn).set_index('Dates', drop=True)))
            PCArs.columns = ['PCA_' +str(tw) + "_" + str(pr)]
            allProjectionsPCA.append(PCArs)

            # LLE
            LLErs = pd.DataFrame(
                sl.rs(pd.read_sql('SELECT * FROM LLE_exPostProjections_tw_' + str(tw) + "_" + str(pr), conn).set_index('Dates', drop=True)))
            LLErs.columns = ['LLE_' +str(tw) + "_" + str(pr)]
            allProjectionsLLE.append(LLErs)

        PCAdf = pd.concat(allProjectionsPCA, axis=1)
        PCAdf['PCA_'+str(tw)] = sl.rs(PCAdf)
        LLEdf = pd.concat(allProjectionsLLE, axis=1)
        LLEdf['LLE_'+str(tw)] = sl.rs(LLEdf)

        medProjectionsDF = pd.concat([LLEdf, PCAdf], axis=1)
        allProjectionsList.append(medProjectionsDF)

    allProjectionsDF = pd.concat(allProjectionsList, axis=1)
    allProjectionsDF.to_sql('allProjectionsDF', conn, if_exists='replace')

def semaOnProjections():
    allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

    allProjectionsDFSharpes = np.sqrt(252) * sl.sharpe(allProjectionsDF).round(4)
    allProjectionsDFSharpes.to_sql('allProjectionsDFSharpes', conn, if_exists='replace')

    print("Sema on Projections")
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

def StationarityOnProjections(manifoldIn, mode):
    allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

    if mode == 'build':

        out = sl.Stationarity(allProjectionsDF, 25, 'exp', multi=1)
        out[0].to_sql('ADF_Test_'+manifoldIn, conn, if_exists='replace')
        out[1].to_sql('Pval_Test_'+manifoldIn, conn, if_exists='replace')
        out[2].to_sql('critVal_Test_'+manifoldIn, conn, if_exists='replace')

    elif mode == 'filter':
        #adf = pd.read_sql('SELECT * FROM Pval_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf.columns = ['$y_{s'+manifoldIn+',(1,t)}$','$y_{s'+manifoldIn+',(2,t)}$','$y_{s'+manifoldIn+',(3,t)}$','$y_{s'+manifoldIn+',(4,t)}$','$y_{s'+manifoldIn+',(5,t)}$']
        #ylbl = "ADF Test p-Values"
        adf = pd.read_sql('SELECT * FROM critVal_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        adf.columns = [x.replace('P0_', '$y_{'+manifoldIn+',(1,t)}$').replace('P1_', '$y_{'+manifoldIn+',(2,t)}$').replace('P2_', '$y_{'+manifoldIn+',(3,t)}$').replace('P3_', '$y_{'+manifoldIn+',(4,t)}$').replace('P4_', '$y_{'+manifoldIn+',(5,t)}$') for x in adf.columns]
        ylbl = "Critical Values"
        #adf = pd.read_sql('SELECT * FROM ADF_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf.columns = ['$y_{s'+manifoldIn+',(1,t)}$','$y_{s'+manifoldIn+',(2,t)}$','$y_{s'+manifoldIn+',(3,t)}$','$y_{s'+manifoldIn+',(4,t)}$','$y_{s'+manifoldIn+',(5,t)}$']
        #ylbl = "ADF Test : $DF_T$"

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

def ARIMAonProjections(scanMode, mode):
    allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    #pd.DataFrame(allProjectionsDF.columns).to_csv("AllProjections.csv")

    if scanMode == 'Main':

        notProcessed = []
        for OrderP in [3,5]:#:[1, 3, 5]:
            orderIn = (OrderP, 0, 0)
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
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                        shList.append([selection, medSh])
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]))
                shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
                shDF.to_sql('sh_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')
                notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
                notProcessedDF.to_sql('notProcessedDF', conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        notProcessedDF = pd.read_sql('SELECT * FROM notProcessedDF', conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
            selection = splitInfo[0]
            orderStr = str(splitInfo[1])
            orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
            try:
                print(selection)
                Arima_Results = sl.ARIMA_Walk(allProjectionsDF[selection], 0.3, orderIn)

                Arima_Results[0].to_sql(
                    selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn,
                    if_exists='replace')
                Arima_Results[1].to_sql(
                    selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn,
                    if_exists='replace')

                sig = sl.sign(Arima_Results[1])

                pnl = sig * Arima_Results[0]
                pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn,
                           if_exists='replace')

                print("ARIMA (" + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + ") Sharpe = ",
                      np.sqrt(252) * sl.sharpe(pnl))
            except Exception as e:
                print("selection = ", selection, ", error : ", e)

def plotARIMASharpes(manifoldIn):
    dfList = []
    for OrderP in [1, 3, 5]:
        medDF = pd.read_sql('SELECT * FROM sh_ARIMA_pnl_'+str(OrderP)+'00', conn).set_index('selection', drop=True)
        medDF['order'] = str(OrderP)+'00'
        dfList.append(medDF)

    df = pd.concat(dfList, axis=0).reset_index()
    print(df[df['sharpe'] > 0.8].set_index('selection', drop=True))
    df.set_index(['selection', 'order'], inplace=True)
    df.sort_index(inplace=True)
    dfUnstack = df.unstack(level=0)
    print(dfUnstack)
    PCAlist = [x for x in dfUnstack.columns if manifoldIn in x[1]]
    dfToplot = dfUnstack.loc[:, PCAlist].abs()
    dfToplot.columns = [x[1] for x in dfToplot.columns]

    fig, ax = plt.subplots()
    dfToplot.plot(ax=ax, kind='bar')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Sharpe Ratio")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 11}, borderaxespad=0.)
    plt.show()

def ContributionAnalysis():
    allProjectionsDF = getProjections('')  # 100, 300, 500

    allProjectionsDF['PCA'] = sl.rs(allProjectionsDF[[x for x in allProjectionsDF.columns if 'PCA' in x]])
    allProjectionsDF['LLE'] = sl.rs(allProjectionsDF[[x for x in allProjectionsDF.columns if 'LLE' in x]])
    allProjectionsDF['PCA_And_LLE'] = allProjectionsDF['PCA'] + allProjectionsDF['LLE']

    allProjectionsDF = sl.ExPostOpt(allProjectionsDF)[0]

    #mainPortfolio = sl.rs(pd.read_sql('SELECT * FROM LongOnlyEWPrsDf', conn).set_index('Dates', drop=True))
    mainPortfolio = sl.rs(pd.read_sql('SELECT * FROM riskParityDF', conn).set_index('Dates', drop=True))

    mainPortfolioSharpe = np.sqrt(252) * sl.sharpe(mainPortfolio).round(4)
    print("mainPortfolioSharpe = ", mainPortfolioSharpe)

    for subPortfolio in allProjectionsDF:
        try:
            ### RAW PROJECTIONS ###
            contribution = allProjectionsDF[subPortfolio]
            ### ARIMA ###
            orderIn = (1, 0, 0)
            #contribution = pd.read_sql('SELECT * FROM ' + subPortfolio + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn).set_index('Dates', drop=True)
            ### RNN ###
            #contribution = pd.read_sql('SELECT * FROM pnl_RNN_'+subPortfolio+'0', conn).set_index('Dates', drop=True)

            totalPortfolioDF = pd.concat([mainPortfolio, contribution], axis=1)
            totalPortfolioDF.columns = ["mainPortfolio", "contribution"]
            totalPortfolioDF['stdRatio'] = 1
            #totalPortfolioDF['stdRatio'] = totalPortfolioDF['mainPortfolio'].std() / totalPortfolioDF['contribution'].std()
            #stdRolling = sl.S(sl.rollerVol(totalPortfolioDF, 25)).fillna(1)
            #totalPortfolioDF['stdRatio'] = stdRolling['mainPortfolio'] / stdRolling['contribution']
            #totalPortfolioDF['stdRatio'] = 1 / totalPortfolioDF['stdRatio']

            totalPortfolio = totalPortfolioDF['mainPortfolio'] + totalPortfolioDF['stdRatio'] * totalPortfolioDF['contribution']

            shrstotalPortfolio = np.sqrt(252) * sl.sharpe(totalPortfolio).round(4)
            print("subPortfolio = ", subPortfolio, ", shrstotalPortfolio = ", round(shrstotalPortfolio,4), " (", round(100*((shrstotalPortfolio-mainPortfolioSharpe)/mainPortfolioSharpe), 1), "\%) \\\\")

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
    pnl = pnl.iloc[round(0.3*len(pnl)):]
    pnl.columns = ["${y}_{3,sPCA}$-EMA(3)", "${y}_{3,sPCA}$-AR(1)", "${y}_{3,sPCA}$-AR(3)",
   "${y}_{3,sPCA}$-AR(5)", "${y}_{5,sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(1)", "${Y}_{sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(5)"]

    targetStratsPnL = pnl[["${y}_{3,sPCA}$-AR(1)", "${y}_{5,sLLE}$-AR(3)", "${Y}_{sLLE}$-AR(3)"]]
    contribPnl = sl.RV(targetStratsPnL, mode='Baskets')
    contribPnl["${y}_{3,sPCA}$-AR(1)_${y}_{5,sLLE}$-AR(3)_${Y}_{sLLE}$-AR(3)"] = sl.rs(targetStratsPnL)
    print("contribPnl = ", np.sqrt(252) * sl.sharpe(contribPnl).round(4))

    fig, ax = plt.subplots()
    #sl.cs(pnl).plot(ax=ax)
    sl.cs(contribPnl).plot(ax=ax)
    #mpl.pyplot.ylabel("Cumulative Returns")
    #mpl.pyplot.xlabel("Dates")
    plt.show()

#DataHandler('investingCom')
#DataHandler('investingCom_Invert')
#shortTermInterestRatesSetup("MainSetup")
#shortTermInterestRatesSetup("retsIRDsSetup")
#shortTermInterestRatesSetup("retsIRDs")

#LongOnly()
#RiskParity('run')
#RiskParity('plot')

#RunManifoldLearningOnFXPairs('PCA', 'Rolling')
#RunManifoldLearningOnFXPairs('PCA', 'Expanding')
#RunManifoldLearningOnFXPairs('LLE', 'Rolling')
#RunManifoldLearningOnFXPairs('LLE', 'Expanding')

#ProjectionsPlots('PCA')
#ProjectionsPlots('LLE')

#getProjections()

#semaOnProjections()

#StationarityOnProjections('PCA', 'build')
#StationarityOnProjections('LLE', 'build')
#StationarityOnProjections('PCA', 'filter')
#StationarityOnProjections('LLE', 'filter')

#ARIMAonProjections('', "")
ARIMAonProjections('Main', "run")
#ARIMAonProjections('Main', "report")
#ARIMAonProjections('ScanNotProcessed', "")
#plotARIMASharpes("PCA")
#plotARIMASharpes("LLE")

#ContributionAnalysis()

#FinalModelPlot()
