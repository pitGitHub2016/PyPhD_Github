from Slider import Slider as sl
import numpy as np, investpy, time
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
#mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('RollingDMAPs.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs_RollingDMAPs/'

twList = [250, 'ExpWindow25']

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

def RunManifold(argList):
    df = argList[0]
    manifoldIn = argList[1]
    tw = argList[2]

    print([df, manifoldIn, tw])

    if tw != 'ExpWindow25':
        print(manifoldIn + " tw = ", tw)
        out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0,1,2,3,4], Scaler='Standard') #RollMode='ExpWindow'

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

    else:
        out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0,1,2,3,4], Scaler='Standard', RollMode='ExpWindow')

        out[0].to_sql(manifoldIn + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[3].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

def RunManifoldLearningOnFXPairs(mode):
    df = sl.fd(pd.read_sql('SELECT * FROM FxDataAdjRets', conn).set_index('Dates', drop=True).fillna(0))#.iloc[-300:]

    if mode == 'test':
        manifoldIn = "DMAP_gDmapsRun"
        tw = 250
        out = sl.AI.gRollingManifold(manifoldIn, df, tw, 5, [0, 1, 2, 3, 4], Scaler='Standard', ProjectionMode='Transpose')  # RollMode='ExpWindow'

        out[0].to_sql("DMAP_gDmapsRun" + 'df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList = out[1]
        out[2].to_sql(manifoldIn + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        out[3].to_sql(manifoldIn + '_sigmasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldIn + '_principalCompsDf_tw_' + str(tw) + "_" + str(k), conn,
                                           if_exists='replace')
    elif mode == 'runOfficial':
        processList = []
        for manifoldIn in ['DMAP_gDmapsRun']:
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

        rsProjection = sl.cs(sl.rs(exPostProjections))
        rsProjection.name = '$\Pi Y_{s'+manifoldIn+','+str(tw)+',t}$'
        rsProjectionList.append(rsProjection)

    csExPostProjections250 = sl.cs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_RsExPostProjections_tw_250', conn).set_index('Dates', drop=True))

    fig, ax = plt.subplots()
    csExPostProjections250.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel(manifoldIn + " Projections (Rolling Window of 250 Days)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()
    ########
    csExPostProjectionsExpWindow25 = sl.cs(pd.read_sql('SELECT * FROM ' + manifoldIn + '_RsExPostProjections_tw_ExpWindow25', conn).set_index('Dates', drop=True))

    fig, ax = plt.subplots()
    csExPostProjectionsExpWindow25.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel(manifoldIn + " Projections (Expanding Window)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    rsProjectionDF = pd.concat(rsProjectionList, axis=1)

    fig, ax = plt.subplots()
    rsProjectionDF.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("Cumulative Return")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

def getProjections(mode):
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

    elif mode == 'plot':
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
        sl.cs(df).plot()
        plt.show()

def PaperizeProjections(pnlSharpes):
    pnlSharpes['paperText'] = ""
    for idx, row in pnlSharpes.iterrows():
        infoSplit = row['index'].split("_")
        if len(infoSplit) < 3:
            pnlSharpes['paperText'] = row["gamma"] + " & " + "Y_{s" + infoSplit[0] + "}"
        else:
            pnlSharpes['paperText'] = row["gamma"] + " & " + "y_{s" + infoSplit[0] + "}"

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
            for OrderP in [3,5]: #[1, 3, 5]:
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

def FinalModelPlot():
    pnl = pd.concat([
        pd.read_sql('SELECT * FROM PCA_ExpWindow25_2_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
        pd.read_sql('SELECT * FROM LLE_ExpWindow25_ARIMA_pnl_100', conn).set_index('Dates', drop=True),
    ], axis=1).dropna()
    pnl = pnl.iloc[round(0.3*len(pnl)):]
    pnl.columns = ["$y_{3,sPCA,t}^{ExpWindow25}$", "$Y_{sLLE,ExpWindow25,t}$"]
    print(np.sqrt(252) * sl.sharpe(pnl))
    comboPnL = sl.RV(pnl, mode='HedgeRatioBasket')
    print(np.sqrt(252) * sl.sharpe(comboPnL))

    fig, ax = plt.subplots()
    sl.cs(pnl["$y_{3,sPCA,t}^{ExpWindow25}$"]).plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    sl.cs(pnl["$Y_{sLLE,ExpWindow25,t}$"]).plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    sl.cs(comboPnL).plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 16}, borderaxespad=0.)
    plt.show()

#####################################################

#DataHandler('investingCom')
#DataHandler('investingCom_Invert')
#shortTermInterestRatesSetup("MainSetup")
#shortTermInterestRatesSetup("retsIRDsSetup")
#shortTermInterestRatesSetup("retsIRDs")

RunManifoldLearningOnFXPairs("test")

#ProjectionsPlots('PCA')
#ProjectionsPlots('LLE')

#getProjections("build")
#getProjections("plot")

#ARIMAonPortfolios("ClassicPortfolios", 'Main', "run")
#ARIMAonPortfolios("ClassicPortfolios", 'Main', "report")
#ARIMAonPortfolios("Projections", 'Main', "run")
#ARIMAonPortfolios("Projections", 'Main', "report")
#ARIMAonPortfolios('ScanNotProcessed', "")
#plotARIMASharpes("ClassicPortfolios", "")
#plotARIMASharpes("Projections", "PCA")
#plotARIMASharpes("Projections", "LLE")

#FinalModelPlot()
