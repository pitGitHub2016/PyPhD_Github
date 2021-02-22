from Slider import Slider as sl
import numpy as np, investpy, time
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

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']

def tildeN(twIn):
    if twIn == 25:
        out = '(1)'
    elif twIn == 100:
        out = '(2)'
    elif twIn == 150:
        out = '(3)'
    elif twIn == 250:
        out = '(4)'
    elif twIn == 'ExpWindow25':
        out = '(5)'
    return out

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

        fig0, ax0 = plt.subplots()
        mpl.pyplot.locator_params(axis='x', nbins=35)
        (IRD*100).plot(ax=ax0)
        for label in ax0.get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(45)
        ax0.set_xlim(xmin=0.0, xmax=len(IRD) + 1)
        mpl.pyplot.ylabel("IRDs (%)")
        plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
        plt.margins(0, 0)
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

        csdfRaw = sl.ecs(dfRaw)
        csdfRaw.index = [x.replace("00:00:00", "").strip() for x in csdfRaw.index]
        csdfAdj = sl.ecs(dfAdj)
        csdfAdj.index = [x.replace("00:00:00", "").strip() for x in csdfAdj.index]

        labelList = ['$x_{i,t}$ (%)', '$r_{i,t}$ (%)']
        c = 0
        for df in [csdfAdj, csdfRaw]:
            df -= 1
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            (df * 100).plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(csdfRaw) + 1)
            mpl.pyplot.ylabel(labelList[c])
            plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()
            c+=1

def LongOnly():
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    longOnlySharpes.to_sql('LongOnlySharpes', conn, if_exists='replace')
    longOnlySharpes["Sharpe"] = "& " + longOnlySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    print("longOnlySharpes = ", longOnlySharpes)

    Edf = sl.EW(df)
    print("Edf Sharpe = ", np.sqrt(252) * sl.sharpe(Edf).round(4))
    Edf_classic = sl.EW(df, mode='classic')
    print("Edf_classic Sharpe = ", np.sqrt(252) * sl.sharpe(Edf_classic).round(4))
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
            mpl.pyplot.ylabel("$\hat{\sigma}_{i,t}$ (%)")
            plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 20})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.85, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.show()

        # Aggregated Risk Parities
        rpRsDF = pd.concat([pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_'+str(tw), conn).set_index('Dates', drop=True) for tw in twList], axis=1)
        rpRsDF.index = [x.replace("00:00:00", "").strip() for x in rpRsDF.index]
        rpRsDF.columns = [str(x)+"_Days" for x in twList]

        #rpRsDFRollToplot = sl.ecs(rpRsDF[['25_Days','100_Days','150_Days','250_Days',"ExpWindow25_Days"]])
        #rpRsDFRollToplot.columns = ['Rolling Window 25 Days','Rolling Window 100 Days', 'Rolling Window 150 Days', 'Rolling Window 250 Days', 'Expanding Window']
        #sl.plotCumulativeReturns([rpRsDFRollToplot], [""])

def RunManifold(argList):
    df = argList[0]
    manifoldIn = argList[1]
    tw = argList[2]

    print([df, manifoldIn, tw])

    if tw != 'ExpWindow25':
        print(manifoldIn + " tw = ", tw)
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], Scaler='Standard') #RollMode='ExpWindow'
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], LLE_n_neighbors=5, ProjectionMode='Transpose') # RollMode='ExpWindow'

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

    else:
        if manifoldIn == 'PCA':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], Scaler='Standard', RollMode='ExpWindow')
        elif manifoldIn == 'LLE':
            out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], LLE_n_neighbors=5, ProjectionMode='Transpose', RollMode='ExpWindow')

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
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

def ProjectionsPlots(manifoldIn):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    rsProjectionList = []
    for tw in twList:
        print(manifoldIn + " tw = ", tw)
        list = []
        for c in [0,1,2,3,4]:
            try:
                medDf = df * sl.S(pd.read_sql('SELECT * FROM ' + manifoldIn + '_principalCompsDf_tw_'+str(tw) + "_" + str(c), conn).set_index('Dates', drop=True))
                pr = sl.rs(medDf.fillna(0))
                list.append(pr)
            except:
                pass
        exPostProjections = pd.concat(list, axis=1, ignore_index=True)
        exPostProjections.columns = ['$\Pi_{'+manifoldIn+','+str(tw)+',1,t}$','$\Pi_{'+manifoldIn+','+str(tw)+',2,t}$',
                                     '$\Pi_{'+manifoldIn+','+str(tw)+',3,t}$','$\Pi_{'+manifoldIn+','+str(tw)+',4,t}$',
                                     '$\Pi_{'+manifoldIn+','+str(tw)+',5,t}$']

        exPostProjections.to_sql(manifoldIn + '_RsExPostProjections_tw_'+str(tw), conn, if_exists='replace')

        exPostProjections.index = [x.replace("00:00:00", "").strip() for x in exPostProjections.index]

        rsProjection = sl.ecs(sl.rs(exPostProjections))
        rsProjection.name = '$\Pi Y_{s'+manifoldIn+','+str(tw)+',t}$'
        rsProjectionList.append(rsProjection)

    csExPostProjections250 = sl.ecs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_RsExPostProjections_tw_250', conn).set_index('Dates', drop=True))
    #csExPostProjections250.columns = ["$\Pi y_{"+manifoldIn+", 1, t}^{(4)}$","$\Pi y_{"+manifoldIn+", 2, t}^{(4)}$","$\Pi y_{"+manifoldIn+", 3, t}^{(4)}$","$\Pi y_{"+manifoldIn+", 4, t}^{(4)}$","$\Pi y_{"+manifoldIn+", 5, t}^{(4)}$"]
    csExPostProjections250.columns = ["1st Coordinate","2nd Coordinate","3rd Coordinate","4th Coordinate","5th Coordinate"]
    csExPostProjections250.index = [x.replace("00:00:00", "").strip() for x in csExPostProjections250.index]
    csExPostProjectionsExpWindow25 = sl.ecs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_RsExPostProjections_tw_ExpWindow25', conn).set_index('Dates', drop=True))
    #csExPostProjectionsExpWindow25.columns = ["$\Pi y_{"+manifoldIn+", 1, t}^{(5)}$","$\Pi y_{"+manifoldIn+", 2, t}^{(5)}$","$\Pi y_{"+manifoldIn+", 3, t}^{(5)}$","$\Pi y_{"+manifoldIn+", 4, t}^{(5)}$","$\Pi y_{"+manifoldIn+", 5, t}^{(5)}$"]
    csExPostProjectionsExpWindow25.columns = ["1st Coordinate","2nd Coordinate","3rd Coordinate","4th Coordinate","5th Coordinate"]
    csExPostProjectionsExpWindow25.index = [x.replace("00:00:00", "").strip() for x in csExPostProjectionsExpWindow25.index]

    rsProjectionDF = pd.concat(rsProjectionList, axis=1)
    #rsProjectionDF.columns = ["$\Pi Y_{"+manifoldIn+",t}^{(1)}$","$\Pi Y_{"+manifoldIn+",t}^{(2)}$","$\Pi Y_{"+manifoldIn+",t}^{(3)}$","$\Pi Y_{"+manifoldIn+",t}^{(4)}$","$\Pi Y_{"+manifoldIn+",t}^{(5)}$"]
    rsProjectionDF.columns = ["Rolling Window (25)","Rolling Window (100)","Rolling Window (150)","Rolling Window (250)","Expanding Window"]
    rsProjectionDF.index = [x.replace("00:00:00", "").strip() for x in rsProjectionDF.index]

    sl.plotCumulativeReturns([csExPostProjections250], ["$\Pi y_{"+manifoldIn+", j, t}$, $j=1,\dots,5$, (%)"])
    sl.plotCumulativeReturns([csExPostProjectionsExpWindow25], ["$\Pi y_{"+manifoldIn+", j, t}$, $j=1,\dots,5$, (%)"])
    sl.plotCumulativeReturns([rsProjectionDF], ["$\Pi Y_{"+manifoldIn+",t}$ (%)"])

def getProjections(mode):
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True)
    if mode == 'build':
        rng = [0, 1, 2, 3, 4]
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
                LLErs = pd.DataFrame(
                    sl.rs(df * sl.S(pd.read_sql('SELECT * FROM LLE_principalCompsDf_tw_' + str(tw) + "_" + str(pr), conn).set_index('Dates', drop=True))))
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

def semaOnProjections(mode, manifoldIn):

    if mode == 'build':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)

        allProjectionsDFSharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(allProjectionsDF).round(4), columns=['Sharpe'])

        df = allProjectionsDFSharpes.copy().reset_index()
        df['manifold'] = None
        df['prType'] = None
        for idx, row in df.iterrows():
            infoSplit = row['index'].split('_')
            if len(infoSplit) == 3:
                df.loc[idx, ['index']] = "$y_{" + str(int(infoSplit[2]) + 1) + ",s" + str(infoSplit[0]) + ",t}^{" + str(
                    infoSplit[1]) + "}$"
                df.loc[idx, ['prType']] = "coordinate"
            elif len(infoSplit) == 2:
                df.loc[idx, ['index']] = "$Y_{s" + str(infoSplit[0]) + "," + str(infoSplit[1]) + ",t}$"
                df.loc[idx, ['prType']] = "global"
            df.loc[idx, ['manifold']] = str(infoSplit[0])
        df.to_sql('allProjectionsDFSharpes', conn, if_exists='replace')

        for manifoldToPlot in ["PCA", "LLE"]:
            for toPlot in ["coordinate", "global"]:
                dfToplot = df[df['prType']==toPlot][['index', 'Sharpe', 'manifold']]
                top_df = dfToplot[dfToplot['index'].str.contains(manifoldToPlot)].set_index("index", drop=True)
                top_df['Sharpe'] = top_df['Sharpe'].round(4).abs()
                top_df = sl.Paperize(top_df.sort_values(by="Sharpe", ascending=False).iloc[:5])
                print("top_df")
                print(top_df["PaperText"])

                dfToplot = dfToplot[dfToplot['manifold']==manifoldToPlot]
                dfToplot = dfToplot.set_index(['index', 'manifold'])

                dfToplot.sort_index(inplace=True)
                dfToplotData = dfToplot.unstack(level=0)
                dfToplotData.columns = [x[1] for x in dfToplotData.columns]

                fig, ax = plt.subplots()
                dfToplotData.plot(ax=ax, kind='bar')
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax.tick_params(axis='x', labelrotation=0)
                mpl.pyplot.ylabel("Sharpe Ratio")
                plt.legend(bbox_to_anchor=(1.01, 1), loc=2, ncol=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
                plt.show()

        print("Sema on Projections")
        for Lag in [3, 5, 25, 50, 250]:
            print(Lag)
            pnl = sl.S(sl.sign(sl.ema(allProjectionsDF, nperiods=Lag))) * allProjectionsDF
            pnl.to_sql('semapnl'+str(Lag), conn, if_exists='replace')

            pnlSharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(pnl).round(4), columns=['Sharpe'])
            pnlSharpes["$\gamma$"] = Lag
            pnlSharpes.to_sql('semapnlSharpes'+str(Lag), conn, if_exists='replace')
            pnlSharpes.abs().to_sql('semapnlAbsSharpes'+str(Lag), conn, if_exists='replace')

    elif mode == 'plot':
        dfList = []
        for Lag in [3, 5, 25, 50, 250]:
            medDF = pd.read_sql('SELECT * FROM semapnlSharpes' + str(Lag), conn).set_index('index', drop=True)
            dfList.append(medDF)

        df = pd.concat(dfList, axis=0).reset_index()

        df['prType'] = None
        for idx, row in df.iterrows():
            infoSplit = row['index'].split('_')
            if len(infoSplit) == 3:
                df.loc[idx, ['index']] = "$y_{"+str(int(infoSplit[2])+1)+",s"+str(infoSplit[0])+",t}^{"+str(infoSplit[1])+"}$"
                df.loc[idx, ['prType']] = "coordinate"
            elif len(infoSplit) == 2:
                df.loc[idx, ['index']] = "$Y_{s"+str(infoSplit[0])+","+str(infoSplit[1])+",t}$"
                df.loc[idx, ['prType']] = "global"

        dfCoordinates = df[df["prType"] == "coordinate"][["index", "Sharpe", "$\gamma$"]]
        top_dfCoordinates = dfCoordinates.copy()
        top_dfCoordinates['Sharpe'] = top_dfCoordinates['Sharpe'].abs().round(4)
        top_dfCoordinates = top_dfCoordinates[top_dfCoordinates['index'].str.contains(manifoldIn)].set_index("index", drop=True)
        top_dfCoordinates = sl.Paperize(top_dfCoordinates.sort_values(by="Sharpe", ascending=False).iloc[:5])
        print("top_dfCoordinates")
        print(top_dfCoordinates["PaperText"])
        dfGlobal = df[df["prType"] == "global"][["index", "Sharpe", "$\gamma$"]]
        top_Global = dfGlobal.copy()
        top_Global['Sharpe'] = top_Global['Sharpe'].abs().round(4)
        top_Global = top_Global[top_Global['index'].str.contains(manifoldIn)].set_index("index", drop=True)
        top_Global = sl.Paperize(top_Global.sort_values(by="Sharpe", ascending=False).iloc[:5])
        print("top_Global")
        print(top_Global["PaperText"])

        ##################################################
        dfCoordinates.set_index(['index', '$\gamma$'], inplace=True)
        dfCoordinates.sort_index(inplace=True)
        dfUnstackCoordinates = dfCoordinates.unstack(level=0)
        manifoldlist = [x for x in dfUnstackCoordinates.columns if manifoldIn in x[1]]
        dfToplotCoordinates = dfUnstackCoordinates.loc[:, manifoldlist]
        dfToplotCoordinates.columns = [x[1] for x in dfToplotCoordinates.columns]

        fig, ax = plt.subplots()
        dfToplotCoordinates.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, ncol=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
        plt.show()
        ##################################################
        dfGlobal.set_index(['index', '$\gamma$'], inplace=True)
        dfGlobal.sort_index(inplace=True)
        dfUnstackGlobal = dfGlobal.unstack(level=0)
        manifoldlist = [x for x in dfUnstackGlobal.columns if manifoldIn in x[1]]
        dfToplotGlobal = dfUnstackGlobal.loc[:, manifoldlist]
        dfToplotGlobal.columns = [x[1] for x in dfToplotGlobal.columns]

        fig, ax = plt.subplots()
        dfToplotGlobal.plot(ax=ax, kind='bar')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        mpl.pyplot.ylabel("Sharpe Ratio")
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 18}, borderaxespad=0.)
        plt.show()

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

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    orderIn = argList[3]
    print(selection, ",", trainLength, ",", orderIn)

    Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn)

    Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn,
                            if_exists='replace')
    Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn,
                            if_exists='replace')

    sig = sl.sign(Arima_Results[1])

    pnl = sig * Arima_Results[0]
    pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]), conn, if_exists='replace')

def ARIMAonPortfolios(Portfolios, scanMode, mode):
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

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            for OrderP in [1, 3, 5]:
                orderIn = (OrderP, 0, 0)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, orderIn])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            for OrderP in [1, 3, 5]:
                orderIn = (OrderP, 0, 0)
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
                shDF.to_sql(Portfolios+'_sh_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')
                notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
                notProcessedDF.to_sql(Portfolios+'_notProcessedDF', conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'notProcessedDF', conn).set_index('index', drop=True)
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

def FinalModelPlot():
    pnl = pd.concat([
        pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE_ExpWindow25_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
    ], axis=1).dropna()
    pnl = pnl.iloc[round(0.3*len(pnl)):]
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
    c = 0
    for df in dfList:
        df.index = [x.replace("00:00:00", "").strip() for x in df.index]
        df -= 1
        (df * 100).plot(ax=ax[c], legend=None)
        for label in ax[c].get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(40)
        ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
        ax[c].set_ylabel(pnl.columns[c], fontsize=18)
        c += 1
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.99, left=0.08, hspace=0, wspace=0)
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

#ProjectionsPlots('PCA')
#ProjectionsPlots('LLE')

#getProjections("build")
#getProjections("plot")

#semaOnProjections("build", "")
#semaOnProjections("plot", "PCA")
#semaOnProjections("plot", "LLE")

#StationarityOnProjections('PCA', 'build')
#StationarityOnProjections('LLE', 'build')
#StationarityOnProjections('PCA', 'filter')
#StationarityOnProjections('LLE', 'filter')

#ARIMAonPortfolios("ClassicPortfolios", 'Main', "run")
#ARIMAonPortfolios("ClassicPortfolios", 'Main', "report")
#ARIMAonPortfolios("Projections", 'Main', "run")
#ARIMAonPortfolios("Projections", 'Main', "report")
#ARIMAonPortfolios('ScanNotProcessed', "")
#plotARIMASharpes("ClassicPortfolios", "")
#plotARIMASharpes("Projections", "PCA")
#plotARIMASharpes("Projections", "LLE")

#ContributionAnalysis()

FinalModelPlot()
