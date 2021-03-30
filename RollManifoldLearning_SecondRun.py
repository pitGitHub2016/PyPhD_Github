from Slider import Slider as sl
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
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

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']
lagList = [2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250]

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
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    longOnlySharpes.to_sql('LongOnlySharpes', conn, if_exists='replace')
    longOnlySharpes["Sharpe"] = "& " + longOnlySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    print("longOnlySharpes = ", longOnlySharpes)

    randomWalkPnl_df = sl.S(sl.sign(df)) * df
    print("Random Walk df : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_df).round(4))

    Edf = sl.ew(df)
    print("Edf Sharpe = ", np.sqrt(252) * sl.sharpe(Edf).round(4))
    randomWalkPnl_Edf = sl.S(sl.sign(Edf)) * Edf
    print("Random Walk Edf : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_Edf).round(4))

    Edf_classic = sl.E(df)
    print("Edf_classic Sharpe = ", np.sqrt(252) * sl.sharpe(Edf_classic).round(4))
    randomWalkPnl_Edf_classic = sl.S(sl.sign(Edf_classic)) * Edf_classic
    print("Random Walk Edf_classic : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_Edf_classic).round(4))

    csEDf = sl.ecs(Edf)
    csEDf_classic = sl.ecs(Edf_classic)

    approxRetsDiff = Edf - Edf_classic
    cs_approxRetsDiff = sl.cs(approxRetsDiff)
    years = (pd.to_datetime(cs_approxRetsDiff.index[-1]) - pd.to_datetime(cs_approxRetsDiff.index[0])) / np.timedelta64(1, 'Y')
    print("Avg LogReturns : ", Edf.mean() * 100, " (%)")
    print("Avg Approximated Returns : ", Edf_classic.mean() * 100, " (%)")
    print("Avg Annual LogReturns = ", (csEDf.iloc[-1] / years) * 100, " (%)")
    print("Avg Annual Approximated Returns = ", (csEDf_classic.iloc[-1] / years) * 100, " (%)")
    print("Average Log vs Approximated Returns Difference : ", approxRetsDiff.mean() * 100, " (%)")
    print("years = ", years)
    print("Total Log vs Approximated Returns Difference = ", cs_approxRetsDiff.iloc[-1] * 100, " (%)")
    print("Avg Annual Log vs Approximated Returns Difference = ", (cs_approxRetsDiff.iloc[-1] / years) * 100, " (%)")

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
            riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe_"+str(tw)])
            shList.append(riskParitySharpes)

            rsDf = pd.DataFrame(sl.rs(df))

            rsDf.to_sql('RiskParityEWPrsDf_tw_'+str(tw), conn, if_exists='replace')
            shrsdfRP = (np.sqrt(252) * sl.sharpe(rsDf)).round(4)

            randomWalkPnl_rsDf = sl.S(sl.sign(rsDf)) * rsDf
            rsDf.to_sql('RiskParityEWPrsDf_randomWalkPnl_tw_'+str(tw), conn, if_exists='replace')
            print("Random Walk rsDf : tw = ", tw, ", Sharpe : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_rsDf).round(4))

            subPnlList = []
            for n in [3, 5, 25, 50, 250]:
                subSemaPnL = sl.S(sl.sign(sl.ema(rsDf, nperiods=n))) * rsDf
                subSemaPnL.columns = ["semaRs_" + str(n)]
                subPnlList.append(subSemaPnL)
            subPnlDF = pd.concat(subPnlList, axis=1)
            pnlSharpes = np.sqrt(252) * sl.sharpe(subPnlDF).round(4)
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
        volToPlot0 = pd.read_sql('SELECT * FROM riskParityVol_tw_25', conn).set_index('Dates', drop=True)
        volToPlot0.index = [x.replace("00:00:00", "").strip() for x in volToPlot0.index]
        volToPlot1 = pd.read_sql('SELECT * FROM riskParityVol_tw_ExpWindow25', conn).set_index('Dates', drop=True)
        volToPlot1.index = [x.replace("00:00:00", "").strip() for x in volToPlot1.index]

        for volDF in [volToPlot0, volToPlot1]:
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

        # Aggregated Risk Parities
        rpRsDF = pd.concat([pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True) for tw in twList], axis=1)
        rpRsDF.index = [x.replace("00:00:00", "").strip() for x in rpRsDF.index]
        rpRsDF.columns = [str(x)+"_Days" for x in twList]

def RunManifold(argList):
    df = argList[0]#.iloc[-300:,:]
    manifoldIn = argList[1]
    tw = argList[2]

    print([df, manifoldIn, tw])

    if tw != 'ExpWindow25':
        print(manifoldIn + " tw = ", tw)
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 20, range(len(df.columns)), Scaler='Standard') #RollMode='ExpWindow'
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 19, range(len(df.columns)-1), LLE_n_neighbors=5, ProjectionMode='Transpose') # RollMode='ExpWindow'

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

    else:
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 20, range(len(df.columns)), Scaler='Standard', RollMode='ExpWindow')
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 19, range(len(df.columns)-1), LLE_n_neighbors=5, ProjectionMode='Transpose', RollMode='ExpWindow')

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

def RunManifoldLearningOnFXPairs():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    processList = []
    for manifoldIn in ['PCA', 'LLE']:
        for tw in twList:
            print(manifoldIn, ",", tw)
            processList.append([df, manifoldIn, tw])

    print("Total Processes = ", len(processList))

    p = mp.Pool(mp.cpu_count())
    result = p.map(RunManifold, tqdm(processList))
    p.close()
    p.join()

def ProjectionsPlots(manifoldIn, mode):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)

    if mode == 'build':

        projections_subgroup_List = []
        for tw in twList:
            print(manifoldIn + " tw = ", tw)
            list = []
            for c in range(len(df.columns)):
                try:
                    medDf = df * sl.S(pd.read_sql('SELECT * FROM ' + manifoldIn + '_principalCompsDf_tw_'+str(tw) + "_" + str(c), conn).set_index('Dates', drop=True))
                    pr = sl.rs(medDf.fillna(0))
                    list.append(pr)
                except:
                    pass
            exPostProjections = pd.concat(list, axis=1, ignore_index=True)
            if manifoldIn == 'PCA':
                exPostProjections.columns = [manifoldIn+'_'+str(tw)+'_'+str(x) for x in range(len(df.columns))]
            elif manifoldIn == 'LLE':
                exPostProjections.columns = [manifoldIn+'_'+str(tw)+'_'+str(x) for x in range(len(df.columns)-1)]

            exPostProjections.to_sql(manifoldIn + '_RsExPostProjections_tw_'+str(tw), conn, if_exists='replace')

            exPostProjections.index = [x.replace("00:00:00", "").strip() for x in exPostProjections.index]

            ### Global Projections ###

            for h in range(1,6):
                for subgroup in ['Head', 'Tail']:
                    print("h = ", h, ", subgroup = ", subgroup)
                    if subgroup == 'Head':
                        projections_subgroup = sl.rs(exPostProjections.iloc[:, :h])
                    else:
                        projections_subgroup = sl.rs(exPostProjections.iloc[:, -h:])
                    projections_subgroup = pd.DataFrame(projections_subgroup)
                    projections_subgroup.columns = [manifoldIn+"_"+str(tw)+"_"+str(h)+"_"+str(subgroup)]
                    projections_subgroup_List.append(projections_subgroup)

        globalProjectionsDF = pd.concat(projections_subgroup_List, axis=1)
        globalProjectionsDF.to_sql('globalProjectionsDF_' + manifoldIn, conn,if_exists='replace')

    elif mode == 'plot':
        sh_randomWalkPnlRSprojections_subgroups = pd.read_sql('SELECT * FROM sh_randomWalkPnlRSprojections_subgroups_' + manifoldIn, conn).set_index('index', drop=True)
        sh_randomWalkPnlRSprojections_subgroups.columns = ['manifold', 'tw', 'subgroup', 'HT', 'sh']

        sh_headMegaDF = sh_randomWalkPnlRSprojections_subgroups[sh_randomWalkPnlRSprojections_subgroups['HT']=="Head"]
        sh_tailMegaDF = sh_randomWalkPnlRSprojections_subgroups[sh_randomWalkPnlRSprojections_subgroups['HT']=="Tail"]

        sh_head_list = []
        sh_tail_list = []
        for tw in sh_headMegaDF['tw'].unique():
            medHeadDF = sh_headMegaDF[sh_headMegaDF['tw'] == tw][['subgroup', 'sh']].set_index('subgroup', drop=True)
            medHeadDF.columns = [tw]
            sh_head_list.append(medHeadDF)
            ###
            medTailDF = sh_tailMegaDF[sh_tailMegaDF['tw'] == tw][['subgroup', 'sh']].set_index('subgroup', drop=True)
            medTailDF.columns = [tw]
            sh_tail_list.append(medTailDF)
        sh_head_DF = pd.concat(sh_head_list, axis=1)
        sh_tail_DF = pd.concat(sh_tail_list, axis=1)

        sl.PaperSinglePlot(sh_head_DF, positions=[0.95, 0.2, 0.85, 0.08, 0, 0])
        sl.PaperSinglePlot(sh_tail_DF, positions=[0.95, 0.2, 0.85, 0.08, 0, 0])

def getProjections(mode):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    if mode == 'build':
        rng = range(len(df.columns))
        allProjectionsList = []
        for tw in twList:
            print("getProjections - tw = ", tw)

            allProjectionsPCA = []
            allProjectionsLLE = []
            for pr in rng:
                # PCA
                PCArs = pd.DataFrame(
                    sl.rs(df * sl.S(pd.read_sql('SELECT * FROM PCA_principalCompsDf_tw_' +str(tw) + "_" + str(pr), conn).set_index('Dates', drop=True))))
                PCArs.columns = ['PCA_' +str(tw) + "_" + str(pr)]
                allProjectionsPCA.append(PCArs)

                # LLE
                try:
                    LLErs = pd.DataFrame(
                        sl.rs(df * sl.S(pd.read_sql('SELECT * FROM LLE_principalCompsDf_tw_' + str(tw) + "_" + str(pr), conn).set_index('Dates', drop=True))))
                    LLErs.columns = ['LLE_' +str(tw) + "_" + str(pr)]
                    allProjectionsLLE.append(LLErs)
                except:
                    pass

            PCAdf = pd.concat(allProjectionsPCA, axis=1)
            PCAdf['PCA_'+str(tw)] = sl.rs(PCAdf)
            LLEdf = pd.concat(allProjectionsLLE, axis=1)
            LLEdf['LLE_'+str(tw)] = sl.rs(LLEdf)

            medProjectionsDF = pd.concat([LLEdf, PCAdf], axis=1)
            allProjectionsList.append(medProjectionsDF)

        allProjectionsDF = pd.concat(allProjectionsList, axis=1)
        allProjectionsDF.to_sql('allProjectionsDF', conn, if_exists='replace')

    elif mode == 'plot':
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        sl.ecs(df).plot()
        plt.show()

def PaperizeProjections(pnlSharpes):
    pnlSharpes['paperText'] = ""
    for idx, row in pnlSharpes.iterrows():
        infoSplit = row['index'].split("_")
        if len(infoSplit) < 3:
            pnlSharpes['paperText'] = row["gamma"] + " & " + "Y_{s" + infoSplit[0] + "}"
        else:
            pnlSharpes['paperText'] = row["gamma"] + " & " + "y_{s" + infoSplit[0] + "}"

def semaOnProjections(space, mode):
    if space == "":
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        allProjectionsDF = allProjectionsDF.iloc[round(0.3 * len(allProjectionsDF)):]
    elif space == 'global':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            sub_globalProjectionsDF = pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index(
                'Dates', drop=True)
            globalProjectionsList.append(sub_globalProjectionsDF.iloc[round(0.3 * len(sub_globalProjectionsDF)):])
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)

    if mode == 'Direct':

        print("Sema on Projections")
        shSema = []
        for Lag in lagList:
            print(Lag)

            allProjectionsDF_pnlSharpes = np.sqrt(252) * sl.sharpe(allProjectionsDF).round(4)

            rw_pnl = sl.S(sl.sign(allProjectionsDF)) * allProjectionsDF
            rw_pnlSharpes = np.sqrt(252) * sl.sharpe(rw_pnl).round(4)

            pnl = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=Lag))) * allProjectionsDF
            pnl.to_sql('semapnl' + str(Lag), conn, if_exists='replace')

            pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl).round(4)).reset_index()
            pnlSharpes['Lag'] = Lag

            tConfDf = sl.tConfDF(allProjectionsDF).set_index("index", drop=True)
            tConfDf_rw = sl.tConfDF(rw_pnl.fillna(0)).set_index("index", drop=True)
            tConfDf_sema = sl.tConfDF(pnl.fillna(0)).set_index("index", drop=True)

            pnlSharpes = pnlSharpes.set_index("index", drop=True)
            pnlSharpes = pd.concat([allProjectionsDF_pnlSharpes, allProjectionsDF.mean() * 100, tConfDf.astype(str),
                                    allProjectionsDF.std() * 100,
                                    rw_pnlSharpes, rw_pnl.mean() * 100, tConfDf_rw.astype(str), rw_pnl.std() * 100,
                                    pnlSharpes, pnl.mean() * 100, tConfDf_sema.astype(str), pnl.std() * 100], axis=1)
            pnlSharpes.columns = ["allProjectionsDF_pnlSharpes", "allProjectionsDF_mean", "tConfDf",
                                  "allProjectionsDF_std",
                                  "rw_pnlSharpes", "rw_pnl_mean", "tConfDf_rw", "rw_pnl_std",
                                  "pnlSharpes", "Lag", "pnl_mean", "tConfDf_sema", "pnl_std"]
            shSema.append(pnlSharpes)

        shSemaDF = pd.concat(shSema).round(4)
        shSemaDF.to_sql('semapnlSharpes_' + space, conn, if_exists='replace')

    elif mode == 'BasketsCombos':
        from itertools import combinations
        shList = []
        for combos in [2, 3]:
            print(combos)
            cc = list(combinations(allProjectionsDF.columns, combos))
            for c in tqdm(cc):
                BasketDF = allProjectionsDF[c[0]] + allProjectionsDF[c[1]]
                shBasket = np.sqrt(252) * sl.sharpe(BasketDF).round(4)
                for Lag in lagList:
                    semaBasketDF = sl.S(sl.sign(sl.ema(BasketDF, nperiods=Lag))) * BasketDF
                    shSemaBasket = np.sqrt(252) * sl.sharpe(semaBasketDF).round(4)
                    shList.append([c[0] + "_" + c[1], shBasket, Lag, shSemaBasket])

        sh_Baskets = pd.DataFrame(shList, columns=['Basket', 'ShBasket', 'Lag', 'ShSemaBasket'])
        sh_Baskets.to_sql('sh_Baskets_' + str(combos), conn, if_exists='replace')

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

def GPRlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    kernelIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", kernelIn, ", ", rw)
    try:
        GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)

        GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
        GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

        pickle.dump(GPR_Results[2], open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) +".p", "wb"))

        sig = sl.sign(GPR_Results[1])

        pnl = sig * GPR_Results[0]
        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

    except Exception as e:
        print(e)

def GPRonPortfolios(Portfolios, scanMode, mode):
    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'ClassicPortfolios':
        allPortfoliosList = []
        for tw in twList:
            subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True)
            subDF.columns = ["RP_"+str(tw)]
            allPortfoliosList.append(subDF)
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        allPortfoliosList.append(LOportfolio)
        allProjectionsDF = pd.concat(allPortfoliosList, axis=1)
    elif Portfolios == 'Finalists':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_2', 'LLE_ExpWindow25']]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for kernelIn in ["RBF_DotProduct","RBF_Matern","RBF_RationalQuadratic", "RBF_WhiteKernel"]:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, kernelIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(GPRlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for kernelIn in ["RBF_DotProduct","RBF_Matern","RBF_RationalQuadratic", "RBF_WhiteKernel"]:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                        print(kernelIn, "_", selection, "_", medSh)
                        shList.append([selection, medSh, kernelIn])
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe', 'kernel']).set_index("selection", drop=True).abs()
            shDF.to_sql(Portfolios+'_sh_GPR_pnl_' + str(rw), conn, if_exists='replace')
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_GPR_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        rw = 250
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_GPR_' + str(rw), conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_GPR_pnl_")
            selection = splitInfo[0]
            kernelIn = str(splitInfo[1]).split("_")[0] + "_" + str(splitInfo[1]).split("_")[1]

            try:
                print(selection)
                GPR_Results = sl.GPR_Walk(allProjectionsDF[selection], 0.3, kernelIn, rw)

                GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
                GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn,
                                      if_exists='replace')

                pickle.dump(GPR_Results[2],
                            open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) + ".p", "wb"))

                sig = sl.sign(GPR_Results[1])

                pnl = sig * GPR_Results[0]
                pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
            except Exception as e:
                print("selection = ", selection, ", error : ", e)

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    orderIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", orderIn, ", ", rw)

    Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn, rw)

    Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')
    Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')

    pickle.dump(Arima_Results[2], open(selection + '_ARIMA_arparamList_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) +".p", "wb"))

    sig = sl.sign(Arima_Results[1])

    pnl = sig * Arima_Results[0]
    pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn, if_exists='replace')

def ARIMAonPortfolios(Portfolios, scanMode, mode):
    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        orderList = [1,3,5]
    elif Portfolios == 'ClassicPortfolios':
        allPortfoliosList = []
        for tw in twList:
            subDF = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True)
            subDF.columns = ["RP_"+str(tw)]
            allPortfoliosList.append(subDF)
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        allPortfoliosList.append(LOportfolio)
        allProjectionsDF = pd.concat(allPortfoliosList, axis=1)
        orderList = [1, 3, 5]
    elif Portfolios == 'Finalists':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_2', 'LLE_ExpWindow25']]
        orderList = [1]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for OrderP in orderList:
                orderIn = (OrderP, 0, 0)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, orderIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for OrderP in orderList:
                orderIn = (OrderP, 0, 0)
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                        shList.append([selection, medSh, orderIn[0]])
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw))
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe', 'order']).set_index("selection", drop=True).abs()
            shDF.to_sql(Portfolios+'_sh_ARIMA_pnl_' + str(rw), conn, if_exists='replace')
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF'+ '_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        for rw in [250]:
            notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'notProcessedDF'+ '_' + str(rw), conn).set_index('index', drop=True)
            for idx, row in notProcessedDF.iterrows():
                splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
                selection = splitInfo[0]
                orderStr = str(splitInfo[1])
                orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
                try:
                    print(selection)
                    Arima_Results = sl.ARIMA_Walk(allProjectionsDF[selection], 0.3, orderIn, rw)

                    Arima_Results[0].to_sql(
                        selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2])+ '_' + str(rw), conn,
                        if_exists='replace')
                    Arima_Results[1].to_sql(
                        selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2])+ '_' + str(rw), conn,
                        if_exists='replace')

                    sig = sl.sign(Arima_Results[1])

                    pnl = sig * Arima_Results[0]
                    pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2])+ '_' + str(rw), conn,
                               if_exists='replace')

                    print("ARIMA (" + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + ") Sharpe = ",
                          np.sqrt(252) * sl.sharpe(pnl))
                except Exception as e:
                    print("selection = ", selection, ", error : ", e)

def plotARIMASharpes(Portfolios, manifoldIn):
    dfList = []
    for OrderP in [1, 3, 5]:
        medDF = pd.read_sql('SELECT * FROM '+Portfolios+'_sh_ARIMA_pnl_'+str(OrderP)+'00', conn).set_index('selection', drop=True)
        medDF['order'] = str(OrderP)+'00'
        dfList.append(medDF)

    df = pd.concat(dfList, axis=0).reset_index()

    if Portfolios == "ClassicPortfolios":

        dfLO = df[df['selection'].str.contains("LO")][["order", "sharpe"]]
        print("LO")
        print(dfLO)
        dfRP = df[df['selection'].str.contains("RP")]
        dfRP.set_index(['selection', 'order'], inplace=True)
        dfRP.sort_index(inplace=True)
        dfRPUnstack = dfRP.unstack(level=0)
        print("RP")
        print(dfRPUnstack)

    elif Portfolios == "Projections":

        df['manifold'] = None
        df['prType'] = None
        for idx, row in df.iterrows():
            infoSplit = row['selection'].split('_')
            if len(infoSplit) == 3:
                df.loc[idx, ['selection']] = "$y_{" + str(int(infoSplit[2]) + 1) + ",s" + str(infoSplit[0]) + ",t}^{" + str(
                    infoSplit[1]) + "}$"
                df.loc[idx, ['prType']] = "coordinate"
            elif len(infoSplit) == 2:
                df.loc[idx, ['selection']] = "$Y_{s" + str(infoSplit[0]) + "," + str(infoSplit[1]) + ",t}$"
                df.loc[idx, ['prType']] = "global"
            df.loc[idx, ['manifold']] = str(infoSplit[0])

        dfCoordinates = df[df["prType"] == "coordinate"][["selection", "sharpe", "order"]]
        top_dfCoordinates = dfCoordinates.copy()
        top_dfCoordinates['sharpe'] = top_dfCoordinates['sharpe'].abs().round(4)
        top_dfCoordinates = top_dfCoordinates[top_dfCoordinates['selection'].str.contains(manifoldIn)].set_index("selection",
                                                                                                             drop=True)
        top_dfCoordinates = sl.Paperize(top_dfCoordinates.sort_values(by="sharpe", ascending=False).iloc[:5])
        print("top_dfCoordinates")
        print(top_dfCoordinates["PaperText"])
        dfGlobal = df[df["prType"] == "global"][["selection", "sharpe", "order"]]
        top_Global = dfGlobal.copy()
        top_Global['sharpe'] = top_Global['sharpe'].abs().round(4)
        top_Global = top_Global[top_Global['selection'].str.contains(manifoldIn)].set_index("selection", drop=True)
        top_Global = sl.Paperize(top_Global.sort_values(by="sharpe", ascending=False).iloc[:5])
        print("top_Global")
        print(top_Global["PaperText"])

        ##################################################
        dfCoordinates.set_index(['selection', 'order'], inplace=True)
        dfCoordinates.sort_index(inplace=True)
        dfUnstackCoordinates = dfCoordinates.unstack(level=0)
        manifoldlist = [x for x in dfUnstackCoordinates.columns if manifoldIn in x[1]]
        dfToplotCoordinates = dfUnstackCoordinates.loc[:, manifoldlist]
        dfToplotCoordinates.columns = [x[1] for x in dfToplotCoordinates.columns]

        fig, ax = plt.subplots()
        dfToplotCoordinates.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.tick_params(axis='x', labelrotation=0)
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, ncol=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
        plt.show()
        ##################################################
        dfGlobal.set_index(['selection', 'order'], inplace=True)
        dfGlobal.sort_index(inplace=True)
        dfUnstackGlobal = dfGlobal.unstack(level=0)
        manifoldlist = [x for x in dfUnstackGlobal.columns if manifoldIn in x[1]]
        dfToplotGlobal = dfUnstackGlobal.loc[:, manifoldlist]
        dfToplotGlobal.columns = [x[1] for x in dfToplotGlobal.columns]

        fig, ax = plt.subplots()
        dfToplotGlobal.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.tick_params(axis='x', labelrotation=0)
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 18}, borderaxespad=0.)
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
    shBaskets = np.sqrt(252) * sl.sharpe(pnlBaskets).round(4).abs()
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
    tPnLs = selPnls.copy()
    ttestList = []
    for c0 in tPnLs.columns:
        for c1 in tPnLs.columns:
            ttest = st.ttest_ind(tPnLs[c0].values, tPnLs[c1].values, equal_var=True)
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
        out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0, 1, 2, 3, 4], LLE_n_neighbors=5, ProjectionMode='Transpose',
                                     RollMode='ExpWindow')

        out[0].to_sql(manifoldIn + 'df_Test_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        exPostProjectionsList = out[2]
        out[3].to_sql(manifoldIn + '_lambdasDf_Test_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_Test_tw_' + str(tw) + "_" + str(k), conn,
                                           if_exists='replace')
            exPostProjectionsList[k].to_sql(manifoldIn + '_exPostProjections_Test_tw_' + str(tw) + "_" + str(k), conn,
                                            if_exists='replace')
    else:
        list = []
        for c in [0,1,2,3,4]:
            try:
                pr = sl.rs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_Test_tw_'+str(tw) + "_" + str(c), conn).set_index('Dates', drop=True).fillna(0))
                list.append(pr)
            except:
                pass
        exPostProjections = pd.concat(list, axis=1, ignore_index=True)
        rsProjection = sl.ecs(sl.rs(exPostProjections))
        rsProjection.name = '$Y_(s' + manifoldIn + ',' + str(tw) + ')(t)$'

        fig, ax = plt.subplots()
        rsProjection.plot(ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        mpl.pyplot.ylabel("Cumulative Return")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
        plt.show()

def Test(mode):
    if mode == 'GPC':
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

    elif mode == 'GPR':
        selection = 'PCA_ExpWindow25_2'
        trainLength = 0.9
        kernelIn = "RBF_Matern"
        rw = 10
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
        GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)
        GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
        GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn,
                              if_exists='replace')

        pickle.dump(GPR_Results[2],
                    open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) + ".p", "wb"))

        sig = sl.sign(GPR_Results[1])

        pnl = sig * GPR_Results[0]
        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

#####################################################

#DataHandler('investingCom')
#DataHandler('investingCom_Invert')
#shortTermInterestRatesSetup("MainSetup")
#shortTermInterestRatesSetup("retsIRDsSetup")
#shortTermInterestRatesSetup("retsIRDs")

#LongOnly()
#RiskParity('run')
#RiskParity('plots')

#RunManifoldLearningOnFXPairs()
#CrossValidateEmbeddings("LLE", "ExpWindow25", "run")
#CrossValidateEmbeddings("LLE", "ExpWindow25", "")

#ProjectionsPlots('PCA', 'build')
#ProjectionsPlots('LLE', 'build')
#ProjectionsPlots('PCA', 'plot')
#ProjectionsPlots('LLE', 'plot')
#ProjectionsPlots('PCA', 'mergeManifolds')

#getProjections("build")
#semaOnProjections()

#StationarityOnProjections('PCA', 'build')
#StationarityOnProjections('LLE', 'build')
#StationarityOnProjections('PCA', 'plot')
#StationarityOnProjections('LLE', 'plot')

#ARIMAonPortfolios("ClassicPortfolios", 'Main', "run")
#ARIMAonPortfolios("ClassicPortfolios", 'Main', "report")
#ARIMAonPortfolios("Projections", 'Main', "run")
#ARIMAonPortfolios("Projections", 'Main', "report")
#ARIMAonPortfolios("Finalists", 'Main', "run")
#ARIMAonPortfolios('ScanNotProcessed', "")
#plotARIMASharpes("ClassicPortfolios", "")
#plotARIMASharpes("Projections", "PCA")
#plotARIMASharpes("Projections", "LLE")

#GPRonPortfolios("ClassicPortfolios", 'Main', "run")
#GPRonPortfolios("ClassicPortfolios", 'Main', "report")
GPRonPortfolios("Projections", 'Main', "run")
GPRonPortfolios("Projections", 'Main', "report")
#GPRonPortfolios("Projections", "ScanNotProcessed", "")
#GPRonPortfolios("Finalists", 'Main', "run")

#Test('GPR')
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

