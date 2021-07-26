from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData_FxData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

#twList = ['ExpWindow25']
twList = [25, 100, 150, 250, 'ExpWindow25']

def DataHandler(mode):

    if mode == 'investingCom':
        dataAll = []
        fxPairsList = ['USD/EUR', 'USD/GBP', 'USD/AUD', 'USD/NZD', 'USD/JPY', 'USD/CAD','USD/CHF','USD/SEK','USD/NOK', 'USD/DKK',
                       'USD/ZAR', 'USD/RUB', 'USD/PLN', 'USD/MXN', 'USD/CNY', 'USD/KRW', 'USD/INR', 'USD/IDR', 'USD/HUF', 'USD/COP']
        namesDF = pd.DataFrame(fxPairsList, columns=["Names"])
        namesDF['Names'] = namesDF['Names'].str.replace('/', '')
        namesDF.to_sql('BasketGekko', conn, if_exists='replace')
        for fx in fxPairsList:
            print(fx)
            name = fx.replace('/', '')
            df = investpy.get_currency_cross_historical_data(currency_cross=fx, from_date='01/09/2001',
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
        IRD['CNYUSD'] = IRD['CHN'] - IRD['USA']
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
                       'ZARUSD', 'RUBUSD', 'PLNUSD', 'MXNUSD', 'CNYUSD', 'KRWUSD', 'INRUSD', 'IDRUSD', 'HUFUSD', 'COPUSD']]
        IRD = IRD.iloc[5389:,:]
        IRD.astype(float).to_sql('IRD', conn, if_exists='replace')

        IRD.index = [x.replace("00:00:00", "").strip() for x in IRD.index]

        fig, ax = plt.subplots()
        mpl.pyplot.locator_params(axis='x', nbins=35)
        IRD.plot(ax=ax)
        for label in ax.get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(45)
        ax.set_xlim(xmin=0.0, xmax=len(IRD) + 1)
        mpl.pyplot.ylabel("IRD", fontsize=32)
        plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.grid()
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
        dfRaw.index = [x.replace("00:00:00", "").strip() for x in dfRaw.index]
        dfAdj.index = [x.replace("00:00:00", "").strip() for x in dfAdj.index]

        csdfRaw = sl.ecs(dfRaw)
        csdfRaw.index = [x.replace("00:00:00", "").strip() for x in csdfRaw.index]
        csdfAdj = sl.ecs(dfAdj)
        csdfAdj.index = [x.replace("00:00:00", "").strip() for x in csdfAdj.index]

        labelList = ['$x_{i,t}$', '$r_{i,t}$']
        c = 0
        for df in [dfAdj, dfRaw]:
            df -= 1
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            df.plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(csdfRaw) + 1)
            mpl.pyplot.ylabel(labelList[c], fontsize=32)
            plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()
            c += 1

        labelList = ['$\Pi x_{i,t}$', '$\Pi r_{i,t}$']
        c = 0
        for df in [csdfAdj, csdfRaw]:
            df -= 1
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            df.plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(csdfRaw) + 1)
            mpl.pyplot.ylabel(labelList[c], fontsize=32)
            plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()
            c+=1

def LongOnly():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df), columns=["Sharpe"])
    longOnlySharpes_active = pd.DataFrame(sl.sharpe(df, mode='processNA', annualiseFlag='yes'))
    df_Mean = 100 * 252 * df.mean()
    tConfdf = sl.tConfDF(pd.DataFrame(df).fillna(0), scalingFactor=252 * 100).set_index("index", drop=True)
    stddf = 100 * np.sqrt(252) * df.std()
    longOnlySharpes.to_sql('LongOnlySharpes', conn, if_exists='replace')
    longOnlySharpes_active.to_sql('LongOnlySharpesActive', conn, if_exists='replace')

    randomWalkPnl_df = sl.S(sl.sign(df)) * df
    randomWalkPnl_dfSharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(randomWalkPnl_df), columns=["Sharpe"])
    randomWalkPnl_df_Mean = 100 * 252 * randomWalkPnl_df.mean()
    tConfrandomWalkPnl_df = sl.tConfDF(pd.DataFrame(randomWalkPnl_df).fillna(0), scalingFactor=252 * 100).set_index("index", drop=True)
    stdrandomWalkPnl_df = 100 * np.sqrt(252) * randomWalkPnl_df.std()
    AssetsStatistics = pd.concat([longOnlySharpes, df_Mean, tConfdf.astype(str), stddf, randomWalkPnl_dfSharpes, randomWalkPnl_df_Mean, tConfrandomWalkPnl_df.astype(str), stdrandomWalkPnl_df], axis=1)
    AssetsStatistics.columns = ["longOnlySharpes", "df_Mean", "tConfdf", "stddf", "randomWalkPnl_dfSharpes", "randomWalkPnl_df_Mean", "tConfrandomWalkPnl_df", "stdrandomWalkPnl_df"]
    AssetsStatistics.round(2).to_sql('AssetsStatistics', conn, if_exists='replace')

    Edf = sl.ew(df)
    print("Edf Sharpe = ", np.sqrt(252) * sl.sharpe(Edf).round(2))
    randomWalkPnl_Edf = sl.S(sl.sign(Edf)) * Edf
    print("Random Walk Edf : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_Edf).round(2))

    Edf_classic = sl.E(df).fillna(0)
    Edf_classic.to_sql('Edf_classic', conn, if_exists='replace')
    meanEdf_classic = 100 * 252 * Edf_classic.mean()
    tConfEWP = sl.tConfDF(pd.DataFrame(Edf_classic).fillna(0), scalingFactor=252 * 100).set_index("index", drop=True)

    stdEWP = 100 * np.sqrt(252) * Edf_classic.std()
    print("Edf_classic Sharpe = ", np.sqrt(252) * sl.sharpe(Edf_classic).round(2), ", Mean = ", meanEdf_classic, ", tConf = ", tConfEWP, ", stdEWP = ", stdEWP)
    randomWalkPnl_Edf_classic = sl.S(sl.sign(Edf_classic)) * Edf_classic
    meanrwEdf_classic = 100 * 252 * randomWalkPnl_Edf_classic.mean()
    tConfrwEWP = sl.tConfDF(pd.DataFrame(randomWalkPnl_Edf_classic).fillna(0), scalingFactor=252 * 100).set_index("index", drop=True)
    stdrwEWP = 100 * np.sqrt(252) * randomWalkPnl_Edf_classic.std()
    print("Edf_classic Sharpe = ", np.sqrt(252) * sl.sharpe(Edf_classic).round(2), ", Mean = ", meanEdf_classic, ", tConf = ", tConfEWP, ", stdEWP = ", stdEWP)
    print("Random Walk Edf_classic : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_Edf_classic).round(2), ", Mean = ", meanrwEdf_classic, ", tConf = ", tConfrwEWP, ", stdEWP = ", stdrwEWP)

    csEDf = sl.ecs(Edf)
    csEDf_classic = sl.ecs(Edf_classic)

    approxRetsDiff = Edf - Edf_classic
    cs_approxRetsDiff = sl.cs(approxRetsDiff)
    years = (pd.to_datetime(cs_approxRetsDiff.index[-1]) - pd.to_datetime(cs_approxRetsDiff.index[0])) / np.timedelta64(1, 'Y')
    #print("Avg LogReturns : ", Edf.mean() * 100, " (%)")
    #print("Avg Approximated Returns : ", Edf_classic.mean() * 100, " (%)")
    #print("Avg Annual LogReturns = ", (csEDf.iloc[-1] / years) * 100, " (%)")
    #print("Avg Annual Approximated Returns = ", (csEDf_classic.iloc[-1] / years) * 100, " (%)")
    #print("Average Log vs Approximated Returns Difference : ", approxRetsDiff.mean() * 100, " (%)")
    #print("years = ", years)
    #print("Total Log vs Approximated Returns Difference = ", cs_approxRetsDiff.iloc[-1] * 100, " (%)")
    #print("Avg Annual Log vs Approximated Returns Difference = ", (cs_approxRetsDiff.iloc[-1] / years) * 100, " (%)")

    LOcsplot = pd.concat([csEDf, csEDf_classic], axis=1)
    LOcsplot.columns = ["$\\tilde{y}_{t, (LO)}$", "$y_{t, (LO)}$"]
    fig, ax = plt.subplots()
    LOcsplot.index = [x.replace("00:00:00", "").strip() for x in csEDf.index]
    LOcsplot.plot()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("")
    plt.show()

    fig, ax = plt.subplots()
    cs_approxRetsDiff.index = [x.replace("00:00:00", "").strip() for x in cs_approxRetsDiff.index]
    cs_approxRetsDiff.plot(ax=ax, legend=False)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("$Err_{yLO, t}$")
    plt.show()

    Edf = pd.DataFrame(Edf)
    Edf.to_sql('LongOnlyEWPEDf', conn, if_exists='replace')
    csEDf.to_sql('LongOnlyEWPcsEdf', conn, if_exists='replace')
    pnlSharpes0 = (np.sqrt(252) * sl.sharpe(Edf)).round(4)
    print("pnlSharpes0 = ", pnlSharpes0)

    pnlList = []
    for n in [3,5,25,50,250]:
        subSemaPnL = sl.S(sl.sign(sl.ema(Edf, nperiods=n))) * Edf
        meanSemaEdf_classic = 100 * 252 * subSemaPnL.mean()
        tConfSemaEWP = sl.tConfDF(pd.DataFrame(subSemaPnL).fillna(0), scalingFactor=252 * 100).set_index("index", drop=True)
        stdSemaEWP = 100 * np.sqrt(252) * subSemaPnL.std()
        print("subSemaPnL Edf_classic : n = ", n, ", Mean = ",
              meanSemaEdf_classic, ", tConf = ", tConfSemaEWP, ", stdEWP = ", stdSemaEWP)
        subSemaPnL.columns = ["semaE_"+str(n)]
        pnlList.append(subSemaPnL)
    pnlDF = pd.concat(pnlList, axis=1)
    pnlDF.to_sql('LongOnlypnlDF', conn, if_exists='replace')
    pnlSharpes = np.sqrt(252) * sl.sharpe(pnlDF).round(4)

    shDF = pd.concat([pnlSharpes0, pnlSharpes])
    shDF.to_sql('LongOnlyEWPEDfSemaSharpes', conn, if_exists='replace')

def RiskParity(mode):
    if mode == 'run':
        df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        expVol = np.sqrt(252) * sl.S(sl.expanderVol(df, 25)) * 100
        shList = []
        semaShList = []
        for tw in [250]:
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
            riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df), columns=["Sharpe_"+str(tw)])
            shList.append(riskParitySharpes)

            rsDf = pd.DataFrame(sl.rs(df))

            rsDf.to_sql('RiskParityEWPrsDf_tw_'+str(tw), conn, if_exists='replace')
            shrsdfRP = (np.sqrt(252) * sl.sharpe(rsDf)).round(4)
            print("Risk Parity Sharpe Ratio : ", shrsdfRP, ", Mean : ", 100 * 252 * rsDf.mean().round(4), ", TConf : ", sl.tConfDF(rsDf, scalingFactor=252 * 100).set_index("index", drop=True),
                  ", Std : ", 100 * np.sqrt(252) * rsDf.std().round(4))

            randomWalkPnl_rsDf = (sl.S(sl.sign(rsDf)) * rsDf).fillna(0)
            rsDf.to_sql('RiskParityEWPrsDf_randomWalkPnl_tw_'+str(tw), conn, if_exists='replace')
            shRWrp = np.sqrt(252) * sl.sharpe(randomWalkPnl_rsDf).round(4)
            meanRWrp = 100 * 252 * randomWalkPnl_rsDf.mean().round(4)
            tConfRWrp = sl.tConfDF(randomWalkPnl_rsDf, scalingFactor=252 * 100).set_index("index", drop=True)
            stdRWrp = 100 * np.sqrt(252) * randomWalkPnl_rsDf.std().round(4)
            print("Random Walk rsDf : tw = ", tw, ", Sharpe : ", shRWrp, ", Mean = ", meanRWrp, ", tConf = ", tConfRWrp, ", stdRWrp = ", stdRWrp)

            # print("Done ....")
            # time.sleep(3000)

            subPnlList = []
            for n in [3, 5, 25, 50, 250]:
                subSemaPnL = (sl.S(sl.sign(sl.ema(rsDf, nperiods=n))) * rsDf).fillna(0)
                meanSemarp = (100 * 252 * subSemaPnL.mean()).round(2)
                tConfSemarp = sl.tConfDF(subSemaPnL, scalingFactor=252 * 100).set_index("index", drop=True)
                stdSemarp = (100 * np.sqrt(252) * subSemaPnL.std()).round(2)
                print("Sema rsDf : n = ", n, ", tw ", tw, ", Sharpe = ", (np.sqrt(252) * sl.sharpe(subSemaPnL)).round(2), ", Mean = ", meanSemarp, ", tConf = ", tConfSemarp, ", stdRWrp = ", stdSemarp)
                subSemaPnL.columns = ["semaRs_" + str(n)]
                subPnlList.append(subSemaPnL)
            subPnlDF = pd.concat(subPnlList, axis=1)
            pnlSharpes = np.sqrt(252) * sl.sharpe(subPnlDF).round(2)
            pnlSharpes['semaRs_0'] = shrsdfRP.values[0]
            pnlSharpes['tw'] = tw

            semaShList.append(pnlSharpes)

        riskParitySharpesDF = pd.concat(shList, axis=1).round(4)
        riskParitySharpesDF = sl.Paperize(riskParitySharpesDF)
        riskParitySharpesDF.to_sql('riskParitySharpesDF', conn, if_exists='replace')
        riskParitySemaSharpesDF = pd.concat(semaShList, axis=1).round(4)
        riskParitySemaSharpesDF.to_sql('riskParitySemaSharpesDF', conn, if_exists='replace')

    elif mode == 'plots':

        # VOLATILITIES
        volToPlot0 = pd.read_sql('SELECT * FROM riskParityVol_tw_250', conn).set_index('Dates', drop=True)
        volToPlot0.index = [x.replace("00:00:00", "").strip() for x in volToPlot0.index]

        for volDF in [volToPlot0]:
            fig0, ax0 = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=40)
            volDF.plot(ax=ax0)
            for label in ax0.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax0.set_xlim(xmin=0.0, xmax=len(volToPlot0) + 1)
            mpl.pyplot.ylabel("$\hat{\sigma}_{i,t}$", fontsize=32)
            plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()

def RunManifold(argList):
    localConn = sqlite3.connect('FXeodData_principalCompsDf.db')
    df = argList[0]
    manifoldIn = argList[1]
    tw = argList[2]

    print([manifoldIn, tw])

    if tw != 'ExpWindow25':
        print(manifoldIn + " tw = ", tw)
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 20, range(len(df.columns)), Scaler='Standard') #RollMode='ExpWindow'
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], LLE_n_neighbors=5, ProjectionMode='Temporal') # RollMode='ExpWindow', ProjectionMode='Transpose'

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), localConn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), localConn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), localConn, if_exists='replace')

    else:
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 20, range(len(df.columns)), Scaler='Standard', RollMode='ExpWindow')
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], LLE_n_neighbors=5, RollMode='ExpWindow', ProjectionMode='Temporal') # RollMode='ExpWindow', ProjectionMode='Transpose'

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), localConn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), localConn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), localConn, if_exists='replace')

def RunManifoldLearningOnFXPairs():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', sqlite3.connect('FXeodData_FxData.db')).set_index('Dates', drop=True)
    processList = []
    for manifoldIn in ['LLE']: #'PCA'
        for tw in twList:
            print(manifoldIn, ",", tw)
            processList.append([df, manifoldIn, tw])

    print("Total Processes = ", len(processList))

    p = mp.Pool(mp.cpu_count())
    result = p.map(RunManifold, tqdm(processList))
    p.close()
    p.join()

def getProjections():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)

    allProjectionsList = []
    for manifoldIn in ["PCA"]: #"LLE"

        projections_subgroup_List = []
        for tw in twList:
            print(manifoldIn + " tw = ", tw)
            prlist = []
            for c in range(len(df.columns)):
                try:
                    medDf = df * sl.S(pd.read_sql(
                        'SELECT * FROM ' + manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(c),
                        conn).set_index('Dates', drop=True))
                    pr = sl.rs(medDf.fillna(0))
                    prlist.append(pr)
                except:
                    pass
            exPostProjections = pd.concat(prlist, axis=1, ignore_index=True)
            if manifoldIn == 'PCA':
                exPostProjections.columns = [manifoldIn + '_' + str(tw) + '_' + str(x) for x in
                                             range(len(df.columns))]
            elif manifoldIn == 'LLE':
                exPostProjections.columns = [manifoldIn + '_' + str(tw) + '_' + str(x) for x in
                                             range(len(df.columns) - 1)]

            exPostProjections.to_sql(manifoldIn + '_RsExPostProjections_tw_' + str(tw), conn, if_exists='replace')
            allProjectionsList.append(exPostProjections)

            ### Global Projections ###

            for h in range(2, 6):
                for subgroup in ['Head', 'Tail']:
                    print("h = ", h, ", subgroup = ", subgroup)
                    if subgroup == 'Head':
                        projections_subgroup = sl.rs(exPostProjections.iloc[:, :h])
                    else:
                        projections_subgroup = sl.rs(exPostProjections.iloc[:, -h:])
                    projections_subgroup = pd.DataFrame(projections_subgroup)
                    projections_subgroup.columns = [manifoldIn + "_" + str(tw) + "_" + str(h) + "_" + str(subgroup)]
                    projections_subgroup_List.append(projections_subgroup)

        globalProjectionsDF = pd.concat(projections_subgroup_List, axis=1)
        globalProjectionsDF.to_sql('globalProjectionsDF_' + manifoldIn, conn, if_exists='replace')

    allProjectionsDF = pd.concat(allProjectionsList, axis=1)
    allProjectionsDF.to_sql('allProjectionsDF', conn, if_exists='replace')
    allProjectionsDFSharpes = np.sqrt(252) * sl.sharpe(allProjectionsDF)
    allProjectionsDFSharpes.to_sql('allProjectionsDFSharpes', conn, if_exists='replace')

def get_LLE_Temporal():
    localConn = sqlite3.connect('FXeodData_principalCompsDf.db')

    lleTemporalList = []
    for tw in twList:
        subDF = pd.read_sql('SELECT * FROM LLE_principalCompsDf_tw_' + str(tw) + "_0", localConn)
        subDF = subDF.rename(columns={"index": "Dates"}).set_index('Dates', drop=True)
        subDF.columns = ["LLE_Temporal_"+str(tw)+"_"+str(x.split("_")[1]) for x in subDF.columns]
        lleTemporalList.append(subDF)
    allProjectionsDF = pd.concat(lleTemporalList, axis=1)
    allProjectionsDF.to_sql('LLE_Temporal_allProjectionsDF', localConn, if_exists='replace')

def Trade_LLE_Temporal():
    localConn = sqlite3.connect('FXeodData_FxData.db')
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', localConn).set_index('Dates', drop=True)
    df_rp = sl.rp(df)
    LO_DF = pd.read_sql('SELECT * FROM Edf_classic', localConn).set_index('Dates', drop=True)
    LO_DF.columns = ["LO"]
    RP_DF = pd.DataFrame(sl.rs(pd.read_sql('SELECT * FROM riskParityDF_tw_250', localConn).set_index('Dates', drop=True).fillna(0)), columns=["RP"])
    benchDF = pd.concat([LO_DF, RP_DF], axis=1)

    sigList = []
    for strategy in tqdm(["Raw", "EMA", "ARIMA", "GPR", "RNN_R"]):
        if strategy == "Raw":
            print("Raw Signal setup ... ")
            raw_sig = pd.read_sql('SELECT * FROM LLE_Temporal_allProjectionsDF', sqlite3.connect('FXeodData_principalCompsDf.db')).set_index('Dates', drop=True)
            raw_sig.columns = [strategy+"_"+str(x) for x in raw_sig.columns]
            sigList.append(raw_sig)
        elif strategy == "EMA":
            print("EMA Signal setup ... ")
            EMA_sig_conn = sqlite3.connect('FXeodData_sema.db')
            subsigList = []
            for Lag in [2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250]:
                subsig = pd.read_sql('SELECT * FROM sema_sig_'+str(Lag), EMA_sig_conn).set_index('Dates',drop=True)
                subsig.columns = [strategy+"_"+str(Lag)+"_"+str(x) for x in subsig.columns]
                subsigList.append(subsig)
            ema_sig = pd.concat(subsigList, axis=1)
            sigList.append(ema_sig)
        elif strategy == "ARIMA":
            print("ARIMA Signal setup ... ")
            ARIMA_sig_conn = sqlite3.connect('FXeodDataARIMA.db')
            subsigList = []
            for tw in twList:
                for pr in [0,1,2,3,4]:
                    for orderIn in ["100", "200", "300"]:
                        subsig = pd.read_sql('SELECT * FROM LLE_Temporal_' +str(tw) + "_" + str(pr) + "_ARIMA_PredictionsDF_" + orderIn + "_250", ARIMA_sig_conn).set_index('Dates', drop=True)
                        subsig.columns = [strategy+"_"+str(tw)+"_"+str(pr)+"_"+orderIn+"_"+str(x) for x in subsig.columns]
                        subsigList.append(subsig)
            arima_sig = pd.concat(subsigList, axis=1)
            sigList.append(arima_sig)
        elif strategy == "GPR":
            print("GPR Signal setup ... ")
            subsigList = []
            for tw in tqdm(twList):
                for pr in [0, 1, 2, 3, 4]:
                    for sys in [0, 1, 2, 3, 4]:
                        outRead = pickle.load(open("Repo/ClassifiersData/GPR_LLE_Temporal_"+str(tw)+"_"+str(pr)+"_"+str(sys)+".p", "rb"))
                        subsig = pd.DataFrame(outRead[3].iloc[:,0])
                        subsig.columns = [strategy + "_" + str(tw) + "_" + str(pr) + "_" + str(sys)]
                        subsigList.append(subsig)
            gpr_sig = pd.concat(subsigList, axis=1)
            sigList.append(gpr_sig)
        elif strategy == "RNN_R":
            sig_Table = "0"
            sig_conn = sqlite3.connect('FXeodData_RNN_R.db')

    df_pnlList = []
    df_rp_pnlList = []
    benchDF_pnlList = []

    for sig in sigList:
        for c in sig.columns:
            sigDF = sl.S(sl.sign(sig[c]))
            sub_df_pnl = df.mul(sigDF, axis=0)
            sub_df_pnl.columns = [c+"_"+str(x) for x in sub_df_pnl.columns]
            sub_df_pnl[c+"_"+"E_PnL"] = sl.E(sub_df_pnl)
            df_pnlList.append(sub_df_pnl)

            sub_df_rp_pnl = df_rp.mul(sigDF, axis=0)
            sub_df_rp_pnl.columns = [c+"_"+str(x) for x in sub_df_rp_pnl.columns]
            sub_df_rp_pnl[c+"_"+"E_PnL"] = sl.E(sub_df_rp_pnl)
            df_rp_pnlList.append(sub_df_rp_pnl)

            sub_benchDF_pnl = benchDF.mul(sigDF, axis=0)
            sub_benchDF_pnl.columns = [c + "_" + str(x) for x in sub_benchDF_pnl.columns]
            benchDF_pnlList.append(sub_benchDF_pnl)

    df_pnl_DF = pd.concat(df_pnlList, axis=1)
    df_rp_pnl_DF = pd.concat(df_rp_pnlList, axis=1)
    benchDF_pnl_DF = pd.concat(benchDF_pnlList, axis=1)

    df_pnl_DF_sh = pd.DataFrame(sl.sharpe(df_pnl_DF, mode='processNA', annualiseFlag='yes'))
    df_rp_pnl_DF_sh = pd.DataFrame(sl.sharpe(df_rp_pnl_DF, mode='processNA', annualiseFlag='yes'))
    benchDF_pnl_DF_sh = pd.DataFrame(sl.sharpe(benchDF_pnl_DF, mode='processNA', annualiseFlag='yes'))
    df_pnl_DF_sh.to_sql('LLE_Temporal_df_pnl_DF_sh', localConn, if_exists='replace')
    df_rp_pnl_DF_sh.to_sql('LLE_Temporal_df_rp_pnl_DF_sh', localConn, if_exists='replace')
    benchDF_pnl_DF_sh.to_sql('LLE_Temporal_benchDF_pnl_DF_sh', localConn, if_exists='replace')

    pickle.dump(df_pnl_DF, open("df_pnl_DF.p", "wb"))
    pickle.dump(df_rp_pnl_DF, open("df_rp_pnl_DF.p", "wb"))
    pickle.dump(benchDF_pnl_DF, open("benchDF_pnl_DF.p", "wb"))

def StationarityOnProjections(manifoldIn, mode):
    allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

    # rollMode = 'exp'
    rollMode = 'roll'
    rollModeTW = 250

    if mode == 'build':

        out = sl.Stationarity(allProjectionsDF[[manifoldIn+"_25",manifoldIn+"_100",manifoldIn+"_150",manifoldIn+"_250",manifoldIn+"_ExpWindow25"]], rollModeTW, rollMode, multi=1)
        out[0].to_sql('ADF_Test_'+manifoldIn+"_"+rollMode+"_"+str(rollModeTW), conn, if_exists='replace')
        out[1].to_sql('Pval_Test_'+manifoldIn+"_"+rollMode+"_"+str(rollModeTW), conn, if_exists='replace')
        out[2].to_sql('critVal_Test_'+manifoldIn+"_"+rollMode+"_"+str(rollModeTW), conn, if_exists='replace')

    elif mode == 'plot':
        adf = pd.read_sql('SELECT * FROM Pval_Test_' + manifoldIn+"_"+rollMode+"_"+str(rollModeTW), conn).set_index('Dates', drop=True)
        #adf.columns = ['$Y_{PCA,t}$ : '+manifoldIn+' RW(25)','$Y_{PCA,t}$ : '+manifoldIn+' RW(100)','$Y_{PCA,t}$ : '+manifoldIn+' RW(150)','$Y_{PCA,t}$ : '+manifoldIn+' RW(250)','$Y_{PCA,t}$ : '+manifoldIn+' EW']
        adf.columns = ['RW(25)','RW(100)','RW(150)','RW(250)','EW']
        #adf = pd.read_sql('SELECT * FROM critVal_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf = pd.read_sql('SELECT * FROM ADF_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #stationaryDF = adf[adf < 0.05]
        #print("% of Stationary points relative to the entire dataset = ", (len(stationaryDF.dropna()) / len(adf)) * 100, " %")

        adf.index = [x.replace("00:00:00", "").strip() for x in adf.index]

        pltMode = 'single'

        if pltMode == 'single':
            adf = adf.iloc[500:]
            #adf.columns = ['(a)', '(b)', '(c)', '(d)', '(e)']
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=40)
            adf.plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(adf) + 1)
            mpl.pyplot.ylabel("ADF Test on $Y_{"+manifoldIn+",t}$ : p-values", fontsize=32)
            plt.legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'size': 24})
            # plt.subplots_adjust(top=0.95, bottom=0.2, right=0.99, left=0.08, hspace=0, wspace=0)
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.12, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()
        elif pltMode == 'multi':
            dfList = [adf[x].iloc[500:] for x in adf.columns]  # [2:4]
            fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
            mpl.pyplot.locator_params(axis='x', nbins=35)
            titleList = ['(a)', '(b)', '(c)', '(d)', '(e)']
            c = 0
            for df in dfList:
                df.plot(ax=ax[c], legend=None, title=titleList[c])
                for label in ax[c].get_xticklabels():
                    label.set_fontsize(25)
                    label.set_ha("right")
                    label.set_rotation(40)
                ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
                ax[c].set_ylabel("p-value", fontsize=16)
                #ax[c].legend(loc=1, fancybox=True, frameon=True, shadow=True, prop={'weight': 'bold', 'size': 24})
                ax[c].grid()
                c += 1
            #plt.subplots_adjust(top=0.95, bottom=0.2, right=0.99, left=0.08, hspace=0.4, wspace=0)
            plt.subplots_adjust(top=0.95, bottom=0.15, right=0.85, left=0.12, hspace=0.4, wspace=0)
            plt.show()

def ContributionAnalysis():
    ProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

    allPortfoliosList = []
    for tw in twList:
        subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_' + str(tw), conn).set_index('Dates', drop=True)
        subDF.columns = ["RP_" + str(tw)]
        allPortfoliosList.append(subDF)
    LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
    LOportfolio.columns = ["LO"]
    allPortfoliosList.append(LOportfolio)
    allmainPortfoliosDF = pd.concat(allPortfoliosList, axis=1)

    allProjectionsDF = pd.concat([ProjectionsDF, allmainPortfoliosDF], axis=1)

    pnlBaskets = sl.RV(allProjectionsDF, mode='Baskets')
    shBaskets = np.sqrt(252) * sl.sharpe(pnlBaskets).round(4)
    shBaskets.to_sql('shBaskets', conn, if_exists='replace')

    shBaskets = shBaskets.reset_index()
    LObased = shBaskets[shBaskets['index'].str.contains("LO")]
    RPbased = shBaskets[shBaskets['index'].str.contains("RP")]

    print("LO")
    df = LObased.copy()
    for idx, row in df.iterrows():
        benchmark = row['index'][-2:]
        projection = row['index'][:-3].split("_")
        if len(projection) == 3:
            df.loc[idx, ['index']] = "\\xi_{"+benchmark+", "+projection[0]+", "+projection[1]+", "+str(int(projection[2])+1)+", t}"
        else:
            df.loc[idx, ['index']] = "\\Xi_{" + benchmark + ", " + projection[0] + ", " + projection[1] + ", t}"
    df = df.set_index('index', drop=True).reset_index()
    df.columns = ["index", "sharpe"]
    print(df.sort_values(by='sharpe', ascending=False).set_index('index', drop=True).reset_index().loc[:5])

    print("RP")
    df = RPbased.copy()
    for idx, row in df.iterrows():
        infoSplit = row['index'].split("_")
        if len(infoSplit) == 5:
            df.loc[idx, ['index']] = "\\xi_{"+infoSplit[3]+", "+infoSplit[0]+", "+infoSplit[-1]+", "+infoSplit[1]+", "+str(int(infoSplit[2])+1)+", t}"
        elif len(infoSplit) == 4:
            df.loc[idx, ['index']] = "\\Xi_{" + infoSplit[3]+", "+infoSplit[0]+", "+infoSplit[-1]+", "+infoSplit[1]+", t}"
    df = df.set_index('index', drop=True).reset_index()
    df.columns = ["index", "sharpe"]
    print(df.sort_values(by='sharpe', ascending=False).set_index('index', drop=True).reset_index().loc[:5])

def FinalModelPlot(mode):
    if mode == 'PnL':
        #pnl = pd.concat([
        #    pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_GaussianProcess_250_pnl', conn).set_index('Dates', drop=True),
        #    pd.read_sql('SELECT * FROM LLE_ExpWindow25_GaussianProcess_250_pnl', conn).set_index('Dates', drop=True),
        #], axis=1).dropna()
        pnl = pd.concat([
            pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_ARIMA_pnl_100_250', conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM LLE_ExpWindow25_ARIMA_pnl_100_250', conn).set_index('Dates', drop=True),
        ], axis=1).dropna()
        pnl = pnl.iloc[round(0.3*len(pnl)):]
        #pnl.columns = ["GPR on $y_{PCA,3,t}$", "GPR on $Y_{LLE,t}$"]
        pnl.columns = ["AR(1) on $y_{PCA,3,t}$", "AR(1) on $Y_{LLE,t}$"]
        print(np.sqrt(252) * sl.sharpe(pnl))
        comboPnL = sl.RV(pnl, mode='HedgeRatioBasket')
        print(np.sqrt(252) * sl.sharpe(comboPnL))

        BestPerformingStrategy_Coordinates = sl.ecs(pnl[pnl.columns[0]])
        BestPerformingStrategy_Coordinates.index = [x.replace("00:00:00", "").strip() for x in BestPerformingStrategy_Coordinates.index]
        BestPerformingStrategy_Global = sl.ecs(pnl[pnl.columns[1]])
        BestPerformingStrategy_Global.index = [x.replace("00:00:00", "").strip() for x in BestPerformingStrategy_Global.index]

        dfList = [BestPerformingStrategy_Coordinates, BestPerformingStrategy_Global]
        fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
        mpl.pyplot.locator_params(axis='x', nbins=35)
        titleList = ['(a)', '(b)']
        c = 0
        for df in dfList:
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
            df -= 1
            mpl.pyplot.locator_params(axis='x', nbins=35)
            (df * 100).plot(ax=ax[c], legend=None)
            for label in ax[c].get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(40)
            ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
            ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
            ax[c].set_ylabel(pnl.columns[c], fontsize=22)
            ax[c].grid()
            c += 1
        plt.subplots_adjust(top=0.95, bottom=0.2)#, right=0.8, left=0.08) #, hspace=0.5, wspace=0
        plt.show()

    elif mode == 'modelParams':
        gprDFList = []
        for modelPickle in ['LLE_ExpWindow25_ARIMA_arparamList_100_250.p', 'PCA_ExpWindow25_2_ARIMA_arparamList_100_250.p']:
            if modelPickle.split("_")[0] == "LLE":
                gpr_modelparamList = pickle.load(open("paramsRepo/LLE_ExpWindow25_GaussianProcess_250_paramList.p", "rb"))
            else:
                gpr_modelparamList = pickle.load(open("paramsRepo/PCA_ExpWindow25_2_GaussianProcess_250_paramList.p", "rb"))
            gprDF = pd.DataFrame(gpr_modelparamList)
            print(modelPickle)
            modelparamList = pickle.load(open(modelPickle, "rb"))
            paramSpace = [[] for j in range(len(modelparamList[0]))]
            for elemDF in modelparamList:
                c = 0
                for idx, row in elemDF.iterrows():
                    paramSpace[c].append(list(row))
                    c += 1

            gprDF.index = pd.DataFrame(paramSpace[0], columns=['Lower', 'Upper', 'pvalues', 'params', 'Dates']).set_index('Dates', drop=True).index
            gprDFList.append(gprDF)

            for paramData in paramSpace:
                paramDF = pd.DataFrame(paramData, columns=['Lower', 'Upper', 'pvalues', 'params', 'Dates']).set_index('Dates', drop=True)

                dfList = [paramDF[['params', 'Lower', 'Upper']], paramDF['pvalues']]
                fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
                mpl.pyplot.locator_params(axis='x', nbins=35)
                titleList = ['(a)', '(b)']
                yLabelList = ['Confidence Intervals', 'p-values']
                c = 0
                for df in dfList:
                    df.index = [x.replace("00:00:00", "").strip() for x in df.index]
                    mpl.pyplot.locator_params(axis='x', nbins=35)
                    if c == 0:
                        df.columns = ['Value', 'Lower', 'Upper']
                        df.plot(ax=ax[c], style=['-', 'y--', 'g--'])
                    else:
                        df.plot(ax=ax[c], legend=None)
                    for label in ax[c].get_xticklabels():
                        label.set_fontsize(25)
                        label.set_ha("right")
                        label.set_rotation(40)
                    ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
                    ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
                    #ax[c].set_ylabel(yLabelList[c], fontsize=22)
                    ax[c].grid()
                    c += 1
                plt.subplots_adjust(top=0.95, bottom=0.2)  # , right=0.8, left=0.08) #, hspace=0.5, wspace=0
                plt.show()

        fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
        mpl.pyplot.locator_params(axis='x', nbins=35)
        titleList = ['(a)', '(b)']
        c = 0
        for df in gprDFList:
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
            mpl.pyplot.locator_params(axis='x', nbins=35)
            df.plot(ax=ax[c], legend=None)
            for label in ax[c].get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(40)
            ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
            ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
            ax[c].grid()
            c += 1
        plt.subplots_adjust(top=0.95, bottom=0.2)  # , right=0.8, left=0.08) #, hspace=0.5, wspace=0
        plt.show()

    elif mode == 'residuals':
        for model in ['LLE_ExpWindow25', 'PCA_ExpWindow25_2']:
            print(model)
            #testPrice = pd.read_sql('SELECT * FROM '+model+'_ARIMA_testDF_100_250', conn).set_index('Dates', drop=True)
            #predPrice = pd.read_sql('SELECT * FROM '+model+'_ARIMA_PredictionsDF_100_250', conn).set_index('Dates', drop=True)

            testPrice = pd.read_sql('SELECT * FROM '+model+'_GaussianProcess_250_testDF', conn).set_index('Dates', drop=True)
            predPrice = pd.read_sql('SELECT * FROM '+model+'_GaussianProcess_250_PredictionsDF', conn).set_index('Dates', drop=True)

            residualsDF = predPrice - testPrice
            print(residualsDF.describe())

            import statsmodels.api as sm
            ljungBox = sm.stats.acorr_ljungbox(residualsDF.values, lags=[1000], boxpierce=True)
            print(ljungBox)

            residualsDF.index = [x.replace("00:00:00", "").strip() for x in residualsDF.index]

            from pandas.plotting import autocorrelation_plot

            fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1)
            mpl.pyplot.locator_params(axis='x', nbins=35)
            titleList = ['(a)', '(b)']
            residualsDF.plot(ax=ax[0], legend=None)
            ax[0].text(.5, .9, titleList[0], horizontalalignment='center', transform=ax[0].transAxes, fontsize=30)
            ax[0].grid()
            autocorrelation_plot(residualsDF, ax=ax[1])
            for label in ax[1].get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(40)
            ax[1].set_xlim(xmin=0.0, xmax=len(residualsDF) + 1)
            ax[1].text(.5, .9, titleList[1], horizontalalignment='center', transform=ax[1].transAxes, fontsize=30)
            ax[1].grid()
            plt.subplots_adjust(top=0.95, bottom=0.15, right=0.85, left=0.12, hspace=0.1, wspace=0)
            plt.grid()
            plt.show()

def RollingStatistics(mode, tvMode, prop):

    if mode == 'Assets':
        rpList = []
        for tw in twList:
            subRP = pd.read_sql('SELECT * FROM riskParityDF_tw_'+str(tw), conn).set_index('Dates', drop=True)
            subRP.columns = [x + '_'+str(tw) for x in subRP.columns]
            rpList.append(subRP)
        selPnls = pd.concat([pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True),
                             pd.concat(rpList, axis=1)], axis=1)

    elif mode == 'Benchmark':
        rpShList = []
        for tw in twList:
            df = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True)
            df.columns = ['Risk Parity : '+str(tw)]
            #df['RandomWalk_RP_'+str(tw)] = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_randomWalkPnl_tw_'+str(tw), conn).set_index('Dates', drop=True)
            rpShList.append(df)
        selPnls = pd.concat(rpShList, axis=1)
        selPnls['LO'] = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)

    elif mode == 'FinalModels_InitialTS':
        selPnls = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
                      ['PCA_ExpWindow25_2', 'LLE_ExpWindow25']]

        fig0, ax0 = plt.subplots()
        mpl.pyplot.locator_params(axis='x', nbins=40)
        selPnls.index = [x.replace("00:00:00", "").strip() for x in selPnls.index]
        sl.ecs(selPnls).plot(ax=ax0)
        for label in ax0.get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(45)
        ax0.set_xlim(xmin=0.0, xmax=len(selPnls) + 1)
        mpl.pyplot.ylabel(tvMode + " Window Sharpe", fontsize=25)
        plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.70, left=0.08, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.grid()
        plt.show()

    elif mode == 'FinalModels':
        selPnls = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_2', 'LLE_ExpWindow25']]
        selPnls['ARIMA_PCA2'] = pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_ARIMA_pnl_300_250', conn).set_index('Dates', drop=True)
        selPnls['ARIMA_LLEglobal'] = pd.read_sql('SELECT * FROM LLE_ExpWindow25_ARIMA_pnl_100_250', conn).set_index('Dates', drop=True)
        selPnls['GPR_PCA2'] = pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_GaussianProcess_250_pnl', conn).set_index('Dates', drop=True)
        selPnls['RandomWalk_PCA2'] = pd.read_sql('SELECT * FROM PCA_randomWalkPnlRSprojections_tw_ExpWindow25', conn).set_index('Dates', drop=True).iloc[:,2]
        selPnls['GPR_LLEglobal'] = pd.read_sql('SELECT * FROM LLE_ExpWindow25_GaussianProcess_250_pnl', conn).set_index('Dates', drop=True)
        rwLLEglobal = pd.read_sql('SELECT * FROM LLE_randomWalkPnlRSprojections', conn).set_index('index', drop=True)
        rwLLEglobal.index = selPnls['GPR_LLEglobal'].index
        selPnls['RandomWalk_LLEglobal'] = rwLLEglobal.iloc[:,4]

    selPnls = sl.ExPostOpt(selPnls)[0]
    selPnls = selPnls.loc["2006-04-05 00:00:00":,:].fillna(0)

    ##################################################################################################################
    from scipy import stats
    tPnLs = selPnls.copy()
    ttestList = []
    for c0 in tPnLs.columns:
        for c1 in tPnLs.columns:
            ttest = stats.ttest_ind(tPnLs[c0].values, tPnLs[c1].values, equal_var=True)
            ttestList.append([c0, c1, tPnLs[c0].mean()*100, tPnLs[c1].mean()*100, tPnLs[c0].std()*100, tPnLs[c1].std()*100, ttest.statistic, ttest.pvalue])

    Static_ttests = pd.DataFrame(ttestList, columns=['Portfolio1_Name','Portfolio2_Name','Portfolio1_Mean','Portfolio2_Mean','Portfolio1_std','Portfolio2_std','ttest_statistic','ttest_pvalue']).round(4)
    Static_ttests.to_sql(mode + '_Static_ttests', conn, if_exists='replace')

    # SHARPES
    StaticSharpes =np.sqrt(252) * sl.sharpe(selPnls).round(4)
    StaticSharpes.to_sql(mode + '_StaticSharpes', conn, if_exists='replace')

    # Counts
    StaticWinLoseRatio = pd.Series([(selPnls[c][selPnls[c]>0].dropna().count() / len(selPnls)) for c in selPnls.columns], index=selPnls.columns)
    StaticWinLoseRatio.to_sql(mode + '_StaticWinLoseRatio', conn, if_exists='replace')

    # Calmars
    StaticCalmars = sl.Calmar(selPnls)
    StaticCalmars.to_sql(mode + '_StaticCalmars', conn, if_exists='replace')

    # MAX DD
    StaticMaximumDDs = sl.MaximumDD(selPnls)
    StaticMaximumDDs.to_sql(mode + '_StaticMaximumDDs', conn, if_exists='replace')

    # SORTINO
    StaticSortinos = sl.sortino(selPnls)
    StaticSortinos.to_sql(mode + '_StaticSortinos', conn, if_exists='replace')

    # HURST
    StaticHurst = sl.Hurst(selPnls, conf='yes').round(4)
    StaticHurst.to_sql(mode + '_StaticHurst', conn, if_exists='replace')

    # ADF
    ADFlist = []
    for it in range(len(selPnls.columns)):
        ADFobj = sl.Stationarity_test(selPnls.iloc[:,it])
        ADFlist.append([selPnls.iloc[:,it].name, ADFobj[0], ADFobj[1], ADFobj[2][0]])
    ADF = pd.DataFrame(ADFlist, columns=['Portfolio', 'ADF', 'p-value', 'critVal']).round(4)
    ADF.to_sql(mode + '_ADF', conn, if_exists='replace')

    shdf = sl.rollStatistics(selPnls, prop, mode=tvMode)

    # ROLLING SHARPES
    fig0, ax0 = plt.subplots()
    mpl.pyplot.locator_params(axis='x', nbins=40)
    shdf.index = [x.replace("00:00:00", "").strip() for x in shdf.index]
    (np.sqrt(252) * shdf).plot(ax=ax0)
    for label in ax0.get_xticklabels():
        label.set_fontsize(25)
        label.set_ha("right")
        label.set_rotation(45)
    ax0.set_xlim(xmin=0.0, xmax=len(shdf) + 1)
    mpl.pyplot.ylabel(tvMode + " Window " + prop, fontsize=25)
    plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.70, left=0.08, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.grid()
    #plt.show()

    # PLOT CUMULATIVE RETURNS
    fig1, ax1 = plt.subplots()
    mpl.pyplot.locator_params(axis='x', nbins=40)
    shdf.index = [x.replace("00:00:00", "").strip() for x in shdf.index]
    sl.cs(selPnls[[x for x in selPnls.columns if 'PCA' in x]]).plot(ax=ax1)
    for label in ax1.get_xticklabels():
        label.set_fontsize(25)
        label.set_ha("right")
        label.set_rotation(45)
    ax1.set_xlim(xmin=0.0, xmax=len(shdf) + 1)
    mpl.pyplot.ylabel(tvMode + " Window " + prop, fontsize=25)
    plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.70, left=0.08, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.grid()
    plt.show()

############### CrossValidateEmbeddings #############
def CrossValidateEmbeddings(manifoldIn, tw, mode):
    if mode == 'run':
        df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        out = sl.AI.gRollingManifold(manifoldIn, df, tw, 20, range(len(df.columns)), Scaler='Standard')

        out[0].to_sql(manifoldIn + 'df_Test_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_Test_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_Test_tw_' + str(tw) + "_" + str(k), conn,
                                           if_exists='replace')

    elif mode == 'Test0':
        df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
        princ = pd.read_sql(
            'SELECT * FROM ' + manifoldIn + '_principalCompsDf_Test_tw_' + str(tw) + "_" + str(19),
            conn).set_index('Dates', drop=True)
        proj = df * sl.S(princ)
        #sl.cs(proj).plot()
        sl.cs(sl.rs(proj)).plot()
        plt.show()

def Test():
    selection = 'PCA_ExpWindow25_2'
    trainLength = 0.3
    tw = 250
    df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
    rwDF = pd.read_sql('SELECT * FROM PCA_randomWalkPnlRSprojections_tw_ExpWindow25', conn).set_index('Dates',
                                                                                                      drop=True).iloc[
           round(0.3 * len(df)):, 2]
    medSh = (np.sqrt(252) * sl.sharpe(rwDF)).round(4)
    print("Random Walk Sharpe : ", medSh)
    # GaussianProcess_Results = sl.GPC_Walk(df, trainLength, tw)
    magicNum = 1
    params = {
        "TrainWindow": 5,
        "LearningMode": 'static',
        "Kernel": "DotProduct",
        "modelNum": magicNum,
        "TrainEndPct": 0.3,
        "writeLearnStructure": 0
    }
    out = sl.AI.gGPC(df, params)

    out[0].to_sql('df_real_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    out[1].to_sql('df_predicted_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    out[2].to_sql('df_predicted_proba_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                  if_exists='replace')
    df_real_price = out[0]
    df_predicted_price = out[1]
    df_predicted_price.columns = df_real_price.columns
    # Returns Prediction
    sig = sl.sign(df_predicted_price)
    pnl = sig * df_real_price
    pnl.to_sql('pnl_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn, if_exists='replace')
    print("pnl_GPC_TEST_sharpe = ", np.sqrt(252) * sl.sharpe(pnl))
    sl.cs(pnl).plot()
    print(out[2].tail(10))
    out[2].plot()
    plt.show()

#####################################################

if __name__ == '__main__':
    #DataHandler('investingCom')
    #DataHandler('investingCom_Invert')
    #shortTermInterestRatesSetup("MainSetup")
    #shortTermInterestRatesSetup("retsIRDsSetup")
    #shortTermInterestRatesSetup("retsIRDs")

    #LongOnly()
    RiskParity('run')
    #RiskParity('plots')

    #RunManifoldLearningOnFXPairs()
    #CrossValidateEmbeddings("PCA", 250, "run")
    #CrossValidateEmbeddings("PCA", 250, "Test0")

    #getProjections()
    #get_LLE_Temporal()
    #Trade_LLE_Temporal()

    #StationarityOnProjections('PCA', 'build')
    #StationarityOnProjections('LLE', 'build')
    #StationarityOnProjections('PCA', 'plot')
    #StationarityOnProjections('LLE', 'plot')

    #Test()
    #ContributionAnalysis()

    #FinalModelPlot('PnL')
    #FinalModelPlot('modelParams')
    #FinalModelPlot('residuals')

    #RollingStatistics('Assets', 'Exp', 'Sharpe')
    #RollingStatistics('Assets', 'Exp', 'Hurst')
    #RollingStatistics('FinalModels', 'Exp', 'Hurst')
    #RollingStatistics('FinalModels_InitialTS', 'Exp', 'Hurst').
    #RollingStatistics('Benchmark', 'Roll', 'Sharpe')
    #RollingStatistics('Benchmark', 'Exp', 'Sharpe')
    #RollingStatistics('Benchmark', 'Exp', 'Hurst')
    #RollingStatistics('FinalModels', 'Roll', 'Sharpe')
    #RollingStatistics('FinalModels', 'Exp', 'Sharpe')
    #RollingStatistics('FinalModels', 'Roll', 'Sharpe')
    #RollingStatistics('FinalModels_InitialTS', 'Exp', 'Sharpe')

