import pandas as pd, numpy as np, matplotlib.pyplot as plt, pdblp, sqlite3, os, time, pickle
from tqdm import tqdm
from datetime import datetime, timedelta
from pyerb import pyerb as pe
from pyerbML import ML, ManSee
import quantstats as qs
from itertools import combinations, permutations
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

"MULTIPROCESSING LOCAL RUNNERS"
def ManifoldLearnerRunner(argDict):
    try:
        if argDict["ManifoldLearnerMode"] == "Learn":
            ManifoldLearnerPack = ManSee.gRollingManifold(argDict["ManifoldLearner"], argDict["ISpace"],
                                                      ProjectionMode=argDict["ProjectionMode"],
                                                      RollMode=argDict["RollMode"], st=argDict["st"])
        elif argDict["ManifoldLearnerMode"] == "Unpack":
            ManifoldLearnerPack = ManSee.ManifoldPackUnpack(argDict["ManifoldLearnerID"], argDict['ManifoldLearnerPack'], TemporalExtraction=argDict["TemporalExtraction"])
        ################################################################################################################################################
        Picklefile = open(argDict["WriteFilePath"], 'wb')
        pickle.dump(ManifoldLearnerPack, Picklefile)
        Picklefile.close()
    except Exception as e:
        print(e)

def MacroConnectorsObservatoryRunner(argDict):
    MacroConnectorsObservatoryPack = pe.RollMetric(argDict['data'],
                                                    metric=argDict['metric'],
                                                   ExcludeSet=argDict['ExcludeSet'],
                                                    RollMode=argDict['RollMode'][0],
                                                   st=argDict['RollMode'][1]
                                                        )
    Picklefile = open(argDict['PickleFileName'], 'wb')
    pickle.dump(MacroConnectorsObservatoryPack, Picklefile)
    Picklefile.close()

class DataDeck:

    def __init__(self, DB, **kwargs):

        if "RefDataFetch" in kwargs:
            RefDataFetch = kwargs['RefDataFetch']
        else:
            RefDataFetch = True

        self.DB = DB
        self.AlternativeStorageLocation = "C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/"
        self.conn = sqlite3.connect(self.DB)
        self.AlternativeStorageLocationConn = sqlite3.connect(self.AlternativeStorageLocation+self.DB)
        self.AccCRNCY = "EUR"
        self.field = "PX_LAST" #"FUT_PX"
        self.factSheetReportPath = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\StrategiesFactSheets/"
        self.DataDeckExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
        if self.DB == "DataDeck.db":
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="ActiveStrategies", engine='openpyxl')
        elif self.DB == "DataDeck_Staged.db":
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="StagedStrategies", engine='openpyxl')
        elif self.DB == "DataDeck_Research.db":
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="ResearchStrategies")
        elif self.DB in ["DataDeck_Mega.db", "DataDeck_1950.db"]:
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="DataDeck")["Asset"].to_frame()
        elif self.DB == "DataDeck_Temp.db":
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="ResearchStrategies")["Dragons"].to_frame()
        if self.DB != "DataDeck.db":
            self.conn = self.AlternativeStorageLocationConn
        ########################
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable', self.conn).set_index('index', drop=True)
        ########################
        self.ActiveStrategiesAssetsClusters = []
        unique_values = set()
        for col in self.ActiveStrategiesDF:
            clusterAssetsList = self.ActiveStrategiesDF[col].dropna()
            self.ActiveStrategiesAssetsClusters.append(clusterAssetsList)
            unique_values.update(clusterAssetsList)
        ##### HANDLE ASSETS ####
        "(APPEND UP TO THE 4-TH POINT OF THE CURVE FOR EACH FUTURE)"
        InvestableUniverse = pd.read_excel(self.DataDeckExcel, sheet_name="DataDeck")["Asset"].tolist()
        self.ActiveAssetsDF = pd.DataFrame([x for x in list(unique_values) if x in InvestableUniverse], columns=["Point_1"])
        if self.DB != "DataDeck_1950.db":
            for point in range(2,5):
                self.ActiveAssetsDF["Point_"+str(point)] = self.ActiveAssetsDF["Point_1"].str.replace("1 Index", str(point)+" Index").str.replace("1 Curncy", str(point)+" Curncy").str.replace("1 Comdty", str(point)+" Comdty")
        self.ActiveAssetsDF.to_sql("FuturesTable", self.conn, if_exists='replace')
        self.ActiveAssets = []
        for col in self.ActiveAssetsDF:
            clusterAssetsList = self.ActiveAssetsDF[col].dropna()
            for x in clusterAssetsList:
                if x not in ['FV4 Comdty', 'RX4 Comdty', 'SW3 Index',
                             'YM4 Comdty', 'BTA4 Comdty', 'FB4 Comdty',
                             'BTS4 Comdty', 'OAT4 Comdty', 'WN4 Comdty',
                             'JB4 Comdty', 'OT4 Index', 'G 4 Comdty',
                             'FUE4 Index', 'UXY4 Comdty', 'QO4 Index',
                             'BJ4 Index','KOA4 Comdty']: # EXCLUDING TICKERS WITH NO BLOOMY DATA
                    self.ActiveAssets.append(x)
        #########################
        try:
            currentDBAssets = pd.read_sql('SELECT * FROM DataDeck LIMIT 5', self.conn).set_index('date',drop=True).columns
            self.NewAssetsList = list(set(list(set(self.ActiveAssets) - set(currentDBAssets)) + list(set(currentDBAssets) - set(self.ActiveAssets))))
        except Exception as e:
            print(e)
            self.NewAssetsList = []
        ##### HANDLE INDICATORS ####
        CurrenciesList = ["EUR","JPY","GBP","CAD","CHF","AUD","BRL","NZD","MXN","ZAR"]
        "Create Interest Rates Spreads"
        InterestRatesSpreads = [x[0]+x[1]+"IS Curncy" for x in list(permutations(CurrenciesList, 2))]
        CarryTotalReturnIndexes = [x[0]+x[1]+"CR Curncy" for x in list(permutations(CurrenciesList, 2))]
        #IRDs_DF = pd.Series(InterestRatesSpreads, name="IRDs");IRDs_DF.to_excel("IRDs.xlsx", index=False)
        self.IndicatorsData = pd.read_excel("IndicatorsDashboard.xlsx",engine='openpyxl')
        self.ActiveIndicators = self.IndicatorsData["Indicator"].dropna().tolist()
        for ir in InterestRatesSpreads:
            self.ActiveIndicators.append(ir)
        for cr in CarryTotalReturnIndexes:
            self.ActiveIndicators.append(cr)
        try:
            currentDBIndicators = pd.read_sql('SELECT * FROM IndicatorsDeck LIMIT 5', self.conn).set_index('date',drop=True).columns
            self.NewIndicatorsList = list(set(list(set(self.ActiveIndicators) - set(currentDBIndicators)) + list(set(currentDBIndicators) - set(self.ActiveIndicators))))
        except Exception as e:
            print(e)
            self.NewIndicatorsList = []
        ########################### UPDATE OPERATIONS ########################
        if RefDataFetch:
            "GET ActiveAssetsReferenceData"
            self.con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
            refDataFields = ["FUT_CONT_SIZE", "CRNCY", "LAST_TRADEABLE_DT", "FUT_TRADING_HRS", "FUT_NOTICE_FIRST", "CURR_GENERIC_FUTURES_SHORT_NAME"]
            refData = self.con.ref(self.ActiveAssets, refDataFields)
            refDataList = []
            for fld in refDataFields:
                rd0 = (refData[refData['field'] == fld]).set_index("ticker", drop=True)['value']
                rd0.name = fld
                refDataList.append(rd0)
            self.ActiveAssetsReferenceData = pd.concat(refDataList, axis=1)
            self.ActiveAssetsReferenceData.to_sql("ActiveAssetsReferenceData", self.conn, if_exists='replace')

        if self.DB == "DataDeck_1950.db":
            self.startReqDataDateStrINCEPTION = '19500101'
        else:
            self.startReqDataDateStrINCEPTION = '20000101'
        self.startReqDataDateStr = self.startReqDataDateStrINCEPTION
        "UPDATE DATA FLAG !!!! " ##############################
        print("len(self.NewAssetsList)  = ", len(self.NewAssetsList), ", len(self.NewIndicatorsList)  = ", len(self.NewIndicatorsList))
        if (len(self.NewAssetsList) != 0)|(len(self.NewIndicatorsList) != 0):
            self.SinceInceptionFetch = 1
        else:
            self.SinceInceptionFetch = 0

        "FORCE SETUP since INCEPTION"
        self.SinceInceptionFetch = 1

    def Run(self, RunSettings):

        ######################## DATA RUNNER ################################
        if RunSettings["RunDataMode"]:
            "Check Last Stored Date and GET ASSETS TIME SERIES"
            try:
                LastDateDF = pd.read_sql('SELECT MAX(date) FROM DataDeck', self.conn)
                LastDateDF['LastDateDiffDays'] = (datetime.now() - pd.to_datetime(LastDateDF['MAX(date)'])).dt.days
            except Exception as e:
                print(e)
                LastDateDF = pd.DataFrame([100], columns=['LastDateDiffDays'])

            print("Controllers : LastDateDF['LastDateDiffDays'].iloc[0] = ", LastDateDF['LastDateDiffDays'].iloc[0],
                  ", len(self.NewAssetsList) = ", len(self.NewAssetsList), ", self.SinceInceptionFetch = ", self.SinceInceptionFetch)
            time.sleep(1)
            #if LastDateDF['LastDateDiffDays'].iloc[0] < 0:
            #    print("DB Last Date is SHORTER than today's date! Asians mixed with US and/or EU Assets ???")
            #else:
            if (self.SinceInceptionFetch != 1)&(LastDateDF['LastDateDiffDays'].iloc[0] < 5):
                startReqDataDate = datetime.now() - timedelta(5)
                self.startReqDataDateStr = startReqDataDate.strftime("%Y")+startReqDataDate.strftime('%m')+startReqDataDate.strftime('%d')
                print("Last date = ", self.startReqDataDateStr)
                self.SinceInceptionFetch = 0
            for x in [
                [self.field, "IndicatorsDeck"],
                [self.field, "DataDeck"],
                ["CONTRACT_VALUE", "HistContractValues"],
                [self.field, "forexData"]
            ]:
                ############################## START NEW SESSION FOR EACH DATA GROUP #####################################
                self.con = pdblp.BCon(debug=True, port=8194, timeout=200000).start()
                ############################## WHAT TO UPDATE #####################################
                if x[1] == "forexData":
                    whatToUpdate = list(set([self.AccCRNCY + tempCur + " Curncy" for tempCur in self.ActiveAssetsReferenceData["CRNCY"].unique() if tempCur != self.AccCRNCY]))
                elif x[1] == "IndicatorsDeck":
                    whatToUpdate = list(set(self.ActiveIndicators))
                    print("Indicators to Update")
                    print(whatToUpdate)
                else:
                    whatToUpdate = list(set(self.ActiveAssets))
                ############################## UPDATE SINCE INCEPTION ??? #########################
                if self.SinceInceptionFetch != 0:
                    self.Hist_df = pd.DataFrame()
                else:
                    try:
                        self.Hist_df = pd.read_sql('SELECT * FROM ' + x[1], self.conn).set_index('date', drop=True)
                    except Exception as e:
                        print(e)
                        self.Hist_df = pd.DataFrame()

                #####################################################################################################
                ############################### MAIN UPDATE ACTION ##################################################
                #####################################################################################################

                df = self.con.bdh(whatToUpdate, x[0], self.startReqDataDateStr, '21000630').ffill()
                df.columns = [x[0] for x in df.columns]
                df.to_sql(x[1] + "_Raw", self.conn, if_exists='replace')
                if x[1] != "IndicatorsDeck":
                    df = df.bfill()
                df.to_sql(x[1]+"Temp", self.conn, if_exists='replace')
                df = pd.read_sql('SELECT * FROM '+x[1]+"Temp", self.conn).set_index('date', drop=True)
                print("self.Hist_df.empty =", self.Hist_df.empty)
                if self.Hist_df.empty:
                    print("REFETCHING ALL DATA SINCE 2000!")
                    allDF = df.copy()
                else:
                    print("NO NEW ASSETS ADDED! Updated current data!")
                    allDF = pd.concat([self.Hist_df, df], axis=0)
                    allDF = allDF[~allDF.index.duplicated(keep='last')]
                allDF.to_sql(x[1], self.conn, if_exists='replace')

                print(self.DB + " Successfully Setup the Dataset !")

                ### CLOSE THE RUNNING BLOOMBERG SESSION ###
                self.con.stop()

            time.sleep(0.5)
        ############################# EURODOLLAR FUTURES TO SOFR FUTURES ###################################################
        "https://www.thestreet.com/investing/futures/shift-from-eurodollar-to-sofr-accelerating"
        TransitionDate = "2018-05-07 00:00:00" # Officially announced earlier though by CME
        ###############################################################################################################
        if RunSettings["DataPatchers"]:
            print("Patching Data ... !")
            for patchPack in [["DataDeck","DataDeck_Eurodollar_Futures.csv"], ["HistContractValues", "DataDeck_Eurodollar_Futures_Historical_Contact_Values.csv"]]:
                print("Patching SOFR Futures and Eurodollar Futures : patchPack = ", patchPack)
                ED = pd.read_csv("F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/"+patchPack[1]).set_index('date', drop=True)
                ED.index = ED.index + " 00:00:00"
                EDinfo = ED.sort_index()

                PatchedDataDeckDF = pd.read_sql('SELECT * FROM '+patchPack[0], self.conn)
                PatchedDataDeckDF = PatchedDataDeckDF.rename(columns={PatchedDataDeckDF.columns[0]: 'date'}).set_index('date', drop=True)
                try:
                    PatchedDataDeckDF = PatchedDataDeckDF.drop(ED.columns, axis=1)
                except Exception as e:
                    print(e)
                PatchedDataDeckDF = pd.concat([PatchedDataDeckDF, EDinfo], axis=1).sort_index()
                PatchedDataDeckDF.index.names = ['date']
                PatchedDataDeckDF.to_sql(patchPack[0], self.conn, if_exists='replace')
            print("Patching Done ... ")
        ######################## RETURNS ####################################
        print("Calculating Rets ... ")
        self.rets = pe.dlog(pd.read_sql('SELECT * FROM DataDeck', self.conn).set_index("date", drop=True)).fillna(0)
        self.rets.to_sql("rets", self.conn, if_exists='replace')
        print("done ... ")
        ########################## LOKBACNKS AND DEMAs CALCULATORS ####################################
        if RunSettings["LookBacksDEMAsCalcMode"][0]:
            print("Checking LookBacks ... !")
            ###################### HANDLE LOOKBACK RETS AFTER PATCHERS ####################################
            LookbackRets = self.rets
            for i in range(1, 5):
                LookbackRets.loc[:TransitionDate, "SFR" + str(i) + " Comdty"] = LookbackRets.loc[:TransitionDate,"ED" + str(i) + " Comdty"]
            LookbackRets.to_sql("LookbackRets", self.conn, if_exists='replace')
            #pe.cs(LookbackRets[["SFR1 Comdty","SFR2 Comdty","SFR3 Comdty","SFR4 Comdty","ED1 Comdty","ED2 Comdty","ED3 Comdty","ED4 Comdty"]]).plot()
            #plt.show()
            ###############################################################################################################
            DataDeckIndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index('date', drop=True)
            IRDs = DataDeckIndicatorsDF[[x for x in DataDeckIndicatorsDF.columns if "IS Curncy" in x]]
            dIRDs = pe.d(IRDs)
            LookbackTargets = pd.concat([LookbackRets, dIRDs], axis=1).sort_index(); ID = ""
            #LookbackTargets = LookbackRets[["SFR1 Comdty","SFR2 Comdty","SFR3 Comdty","SFR4 Comdty","ED1 Comdty","ED2 Comdty","ED3 Comdty","ED4 Comdty"]]; ID = "Test"
            print("(Shape) LookbackTargets = ", LookbackTargets.shape)
            ###############################################################################################################
            print("Calculating LookBacks ... !")
            if RunSettings["LookBacksDEMAsCalcMode"][1]:
                LookBackSinceInception = True
                try:
                    OldLookBacks = pd.read_sql('SELECT * FROM LookBacks'+ID, self.conn)
                    OldLookBacks = OldLookBacks.set_index(OldLookBacks.columns[0], drop=True)
                    OldLookBacks.index.names = ['date']
                    LookBackSinceInception = False
                except Exception as e:
                    print(e)
                #OldLookBacks.plot()
                #plt.show()
                "FORCE LookBack Recalculation Since Inception"
                LookBackSinceInception = True
                print("LookBackSinceInception = ", LookBackSinceInception)
                if LookBackSinceInception == False:
                    print("LookBacks Already Exist! Appending ... ")
                    NewLookBacks = pe.DynamicSelectLookBack(LookbackTargets, RollMode="ExpWindow", st=LookbackTargets.shape[0]-10).dropna()
                    LookBacks = pd.concat([OldLookBacks, NewLookBacks], axis=0)
                    LookBacks = LookBacks[~LookBacks.index.duplicated(keep='last')]
                else:
                    print("Need to Calculate LookBacks since inception ... !")
                    LookBacks = pe.DynamicSelectLookBack(LookbackTargets, RollMode="ExpWindow")
                LookBacks.index.names = ['date']
                LookBacks = LookBacks.ffill().bfill().sort_index()
                #for i in range(1,4):
                #    LookBacks["ED"+str(i)+" Comdty"] = LookBacks["SFR"+str(i)+" Comdty"]
                "Write LookBacks in DB"
                try:
                    LookBacks.to_sql("LookBacks"+ID, self.conn, if_exists='replace')
                except Exception as e:
                    print(e)
                    LookBacks.to_sql("LookBacks"+ID, self.AlternativeStorageLocationConn, if_exists='replace')
                LookBacks.plot()
                plt.show()
                print("LookBacks Ready ... !")
            ###################################################################################################################################
            print("Calculating DEMAs ... !")
            if RunSettings["LookBacksDEMAsCalcMode"][2]:
                LookBacks = pd.read_sql('SELECT * FROM LookBacks' + ID, self.conn).set_index("date", drop=True)
                DEMAs = pe.dema(LookbackTargets, LookBacks).fillna(0)
                DEMAs.index.names = ['date']
                DEMAs.to_sql("DEMAs"+ID, self.conn,if_exists='replace')
                print("DEMAs Ready ... !")
        ########################## ECO DATA ##########################################
        if RunSettings["RunEcoDataMode"]:
            print("Running Eco Data Mode ... ")
            ECO_IndicatorsConn = sqlite3.connect("ECO_Indicators.db")
            ECO_Indicators = []
            for country in ["USA", "EU", "UK", "JP", "CAD", "AUS", "BRL"]:
                ECO_Indicators += pd.read_excel("Bloomberg_Key_ECO_Indicators.xlsx", sheet_name=country)["Ticker"].tolist()
            ECO_Con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
            ReqFields = ["PX_LAST", "BN_SURVEY_MEDIAN"]
            fetchedDF = ECO_Con.bdh(ECO_Indicators, ReqFields, '20000101', '21000630').ffill().bfill()

            for field in ReqFields:
                print(field)
                subDF = fetchedDF[[x for x in fetchedDF.columns if x[1] == field]]
                subDF.columns = [x[0] for x in subDF.columns]
                subDF.to_sql("ECO_Indicators_"+field, ECO_IndicatorsConn, if_exists='replace')
            "Spread"
            Surveys = pd.read_sql('SELECT * FROM ECO_Indicators_BN_SURVEY_MEDIAN', ECO_IndicatorsConn).set_index("date", drop=True)
            Last = pd.read_sql('SELECT * FROM ECO_Indicators_PX_LAST', ECO_IndicatorsConn).set_index("date", drop=True)
            SpreadDF = pd.DataFrame(None, index=Last.index, columns=Last.columns)
            for c in SpreadDF.columns:
                try:
                    SpreadDF[c] = Last[c] - Surveys[c]
                except Exception as e:
                    pass
                    #print(e)
            SpreadDF = SpreadDF.dropna(how='all', axis=1)
            SpreadDF.to_sql("ECO_Indicators_Spread", ECO_IndicatorsConn, if_exists='replace')
        ########################## ROLLING HURST ##########################################
        if RunSettings["RunHurst"] in ["Setup", "Update", "Read"]:
            #WhatToHurstID = "Assets"
            #WhatToHurstID = "RVsSpreadsLinear"
            WhatToHurstID = "RVsHedgeRatioPnLExpand"
            ################################################################
            TargetRVsSpace = "Valley"
            #TargetRVsSpace = "Dragons"
            TargetRVs = [x for x in self.rets.columns if x in self.ActiveStrategiesDF[TargetRVsSpace].tolist()]
            #MoreFuturesCurvePoints = pe.getMoreFuturesCurvePoints(self.Assets, self.FuturesTable, [2, 3])
            ################################################################
            Hurst_PicklefileName = self.AlternativeStorageLocation + 'RollHurst_'+WhatToHurstID
            HurstConn = sqlite3.connect(self.AlternativeStorageLocation + "RollHurst.db")
            if RunSettings["RunHurst"] in ["Setup", "Update"]:
                ################################################################################################################################
                if WhatToHurstID == "Assets":
                    WhatToHurst = self.rets[[x for x in self.rets if x in self.FuturesTable["Point_1"].tolist()]]
                elif WhatToHurstID == "RVsSpreadsLinear":
                    WhatToHurst = pe.RV(self.rets[TargetRVs], HedgeRatioConnectionMode="Spreads", mode="Linear", n=25, RVspace="classicCombos")
                elif WhatToHurstID == "RVsHedgeRatioPnLExpand":
                    WhatToHurst = pe.RV(self.rets[TargetRVs], HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Expand", n=25)
                ################################################################################################################################
                for RollMode in [["RollWindow", 250], ['ExpWindow', 25]]:#
                    print("Running Hurst on "+WhatToHurstID+" : RollMode = ", RollMode)
                    ##############################################################################################################
                    RollHurst = pe.gRollingHurst(WhatToHurst, RollMode=RollMode[0], st=RollMode[1])
                    #Hurst_Picklefile = open(Hurst_PicklefileName, 'wb')
                    #pickle.dump(RollHurst, Hurst_Picklefile)
                    #Hurst_Picklefile.close()
                    RollHurst[WhatToHurst.columns].to_sql("RollHurst"+WhatToHurstID+"_"+RollMode[0]+"_"+str(RollMode[1]), HurstConn, if_exists='replace')
                    ##############################################################################################################
                    SpreadsStats = []
                    for asset in RollHurst.columns:
                        SpreadsStats.append([asset, RollHurst[asset].mean(), RollHurst[asset].max(), RollHurst[asset].min()])
                    ##############################################################################################################
                    SpreadsStatsDF = pd.DataFrame(SpreadsStats, columns=["Asset", "AvgHurst", "MaxHurst", "MinHurst"]).set_index("Asset", drop=True)
                    SpreadsStatsDF.to_sql("RollHurstSpreadsStats_"+WhatToHurstID+"_"+RollMode[0]+"_"+str(RollMode[1]), HurstConn, if_exists='replace')
            elif RunSettings["RunHurst"] == "Read":
                TargetAssets = ["SFR1 Comdty", "ED1 Comdty", "ES1 Index", "VG1 Index", "TU1 Comdty", "DU1 Comdty", "TY1 Comdty", "RX1 Comdty", "GC1 Comdty", "CC1 Comdty"]

                #HurstSpace = "RollHurstAssets_ExpWindow_25"
                HurstSpace = "RollHurstAssets_RollWindow_250"
                RollHurst = pd.read_sql('SELECT * FROM '+HurstSpace, HurstConn).set_index('date', drop=True)
                (RollHurst[TargetAssets]-0.5).plot()
                plt.show()

                #RollHurstPickle = open(Hurst_PicklefileName, 'rb')
                #RollHurstPickleData = pickle.load(RollHurstPickle)
                #RollHurstPickle.close()
                #print(RollHurstPickleData)
                #(RollHurstPickleData[TargetAssets]-0.5).plot()
                #plt.show()
        ########################## MANIFOLD LEARNING FEATURES ####################################
        if RunSettings["ManifoldLearnersMode"] in ["Setup", "Update", "Read", "ManifoldPackUnpack"]:
            self.dIndicatorsDF = pe.d(pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index("date", drop=True)).fillna(0)
            self.ActiveStrategiesAssetsClusters.append(pd.Series(list(self.rets.columns), name="ALL").drop_duplicates())
            self.ActiveStrategiesAssetsClusters.append(pd.Series(list(self.rets.columns)+list(self.dIndicatorsDF.columns), name="ALLm").drop_duplicates())
            processList = []
            for ManifoldLearner in ["PCA"]:#"PCA", "Beta", "LLE", "DMAPS", "BetaRegressV", "BetaProject"
                for RollModeIn in [['RollWindow', 250]]: #['ExpWindow', 25]
                    for ProjectionModeIn in ["NoTranspose"]:#NoTranspose,Transpose
                        for StrategiesAssetsCluster in self.ActiveStrategiesAssetsClusters:
                            StrategiesAssetsClusterName = StrategiesAssetsCluster.name
                            if StrategiesAssetsClusterName in ["Endurance"]: #!= "Expedition", "Endurance", StrategiesAssetsClusterName in ["ALL", "ALLm"]
                                ManifoldLearnerID = ManifoldLearner + "_" + ProjectionModeIn + "_" + str(RollModeIn[0]) + "_" + str(RollModeIn[1]) + "_" + StrategiesAssetsClusterName
                                print(ManifoldLearnerID)
                                if RunSettings["ManifoldLearnersMode"] == "ManifoldPackUnpack":
                                    dbfile = open(self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID,'rb')
                                    ManifoldLearnerPack = pickle.load(dbfile)
                                    dbfile.close()
                                    for TemporalExtractionIn in ["LastValue", "PearsonCorrelationVal"]:
                                        argDict = {
                                            "ManifoldLearnerMode": "Unpack",
                                            "ManifoldLearnerID": ManifoldLearnerID,
                                            "ManifoldLearnerPack": ManifoldLearnerPack,
                                            "TemporalExtraction": TemporalExtractionIn,
                                            "WriteFilePath": self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID + "_"+TemporalExtractionIn+"_Unpacked"
                                        }
                                        processList.append(argDict)
                                elif RunSettings["ManifoldLearnersMode"] == "Read":
                                    dbfile = open(self.AlternativeStorageLocation+'ManifoldLearners\\' + ManifoldLearnerID, 'rb')
                                    ManifoldLearnerPack = pickle.load(dbfile)
                                    dbfile.close()
                                    print(len(ManifoldLearnerPack[1]))
                                    subPack = ManifoldLearnerPack[1][100]
                                    print(pd.DataFrame(subPack[2]))
                                    time.sleep(3000)
                                    principalCompsDfList = ManifoldLearnerPack[1]
                                    exPostProjectionsList = ManifoldLearnerPack[2]
                                    #k = 0
                                    k = 1
                                    #k = len(principalCompsDfList)-1
                                    # principalCompsDf = pd.read_sql('SELECT * FROM '+ManifoldLearnerID+'_principalCompsDf_' + str(k), self.conn).set_index("date", drop=True)
                                    #principalCompsDf = principalCompsDfList[k]
                                    principalCompsDf = pe.rowStoch(principalCompsDfList[k])

                                    #principalCompsDf = principalCompsDf[[x for x in self.rets if x in self.FuturesTable["Point_1"].tolist()]]
                                    principalCompsDf = principalCompsDf[["ES1 Index","NQ1 Index",
                                                                         "VG1 Index","GX1 Index",
                                                                         "DU1 Comdty","TU1 Comdty",
                                                                         "RX1 Comdty", "TY1 Comdty",
                                                                         #"GC1 Comdty", "CC1 Comdty",
                                                                         #"NG1 Comdty", "CL1 Comdty",
                                                                         ]]
                                    print(principalCompsDf.iloc[-1])
                                    print(principalCompsDf.corr())
                                    fig, ax = plt.subplots(nrows=2, ncols=1)
                                    principalCompsDf.plot(ax=ax[0])
                                    pe.cs(principalCompsDf).plot(ax=ax[1])
                                    plt.legend(loc='lower left')
                                    plt.show()
                                else:
                                    StrategiesAssetsClusterElements = StrategiesAssetsCluster.tolist()
                                    if StrategiesAssetsClusterName == "ALLm":
                                        ISpace = pd.concat([self.rets, self.dIndicatorsDF], axis=1).sort_index().fillna(0)[StrategiesAssetsClusterElements]
                                    else:
                                        ISpace = self.rets[StrategiesAssetsClusterElements]
                                    #ISpace = ISpace.tail(255)
                                    ################################################################################################################################################
                                    argDict = {
                                        "ManifoldLearnerMode" : "Learn",
                                        "ManifoldLearner": ManifoldLearner,
                                        "ISpace": ISpace,
                                        "ProjectionMode" : ProjectionModeIn,
                                        "RollMode" : RollModeIn[0],
                                        "st" : RollModeIn[1],
                                        "WriteFilePath":self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID
                                    }
                                    #ManifoldLearnerRunner(argDict)
                                    processList.append(argDict)
            p = mp.Pool(mp.cpu_count()-2)
            result = p.map(ManifoldLearnerRunner, tqdm(processList))
            p.close()
            p.join()
        ########################## MACRO CONNECTORS OBSERVATORY ####################################
        if RunSettings["MacroConnectorsObservatory"][0]:
            processList = []
            MacroTargetsList = ["M2US000$ Index", "MXWO000G Index", "MAEUMMT Index", "M2US000V Index", "M2US000G Index",
                            "BFCIUS Index", "BFCIEU Index",
                            "USSWIT2 Curncy", "EUSWI2 Curncy",
                            "VIX Index", "MOVE Index", "JPMVXYG7 Index", "GVZ Index", "OVX Index"]
            
            CheckedSpace = "Rets"
            #CheckedSpace = "CTA"
            if CheckedSpace == "CTA":
                TradedUniverse = pd.read_sql('SELECT * FROM subPortfoliosRetsDF', sqlite3.connect("StrategiesAggregator.db"))
                TradedUniverse = TradedUniverse.set_index(TradedUniverse.columns[0], drop=True).fillna(0)
                TradedUniverse.index.names = ['date']
            else:
                TradedUniverse = self.rets[self.ActiveAssetsDF["Point_1"].tolist()]
                print("TradedUniverse.shape = ", TradedUniverse.shape)

            #PackSpace = ["Indicators", "dIndicators"]
            PackSpace = ["EcoLast","EcoSurvey","EcoSpread"] #"Indicators","dIndicators","EcoLast","EcoSurvey","EcoSpread"
            for pack in PackSpace:
                ########################################################################################################################################
                if RunSettings["MacroConnectorsObservatory"][1] == "DataSetup":
                    print(pack)
                    if pack == "Indicators":
                        subPackDF = pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index("date", drop=True)
                    elif pack == "dIndicators":
                        subPackDF = pe.d(pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index("date", drop=True))
                    elif pack == "EcoLast":
                        subPackDF = pd.read_sql('SELECT * FROM ECO_Indicators_PX_LAST', sqlite3.connect("ECO_Indicators.db")).set_index("date", drop=True)
                    elif pack == "EcoSurvey":
                        subPackDF = pd.read_sql('SELECT * FROM ECO_Indicators_BN_SURVEY_MEDIAN', sqlite3.connect("ECO_Indicators.db")).set_index("date", drop=True)
                    elif pack == "EcoSpread":
                        subPackDF = pd.read_sql('SELECT * FROM ECO_Indicators_Spread', sqlite3.connect("ECO_Indicators.db")).set_index("date", drop=True)

                    try:
                        subPackDF = subPackDF[MacroTargetsList]
                    except Exception as e:
                        print(e)

                    ##### WRITE TO DB ###
                    MacroConnectorsObservatoryDF = pd.concat([TradedUniverse, subPackDF], axis=1).sort_index()
                    if pack not in ["dIndicators"]:
                        ffillCols = [x for x in subPackDF.columns]
                        MacroConnectorsObservatoryDF.loc[:,ffillCols] = MacroConnectorsObservatoryDF.loc[:,ffillCols].ffill()
                    MacroConnectorsObservatoryDF = MacroConnectorsObservatoryDF.loc[TradedUniverse.index, :]
                    MacroConnectorsObservatoryDF.to_sql("MacroConnectorsObservatoryDF_"+pack+"_"+CheckedSpace, sqlite3.connect(self.AlternativeStorageLocation+"MacroConnectorsObservatory.db"), if_exists='replace')
                ################################################################################################################################################################
                MacroConnectorsObservatoryDF = pd.read_sql('SELECT * FROM MacroConnectorsObservatoryDF_' + pack+"_"+CheckedSpace,sqlite3.connect(self.AlternativeStorageLocation+"MacroConnectorsObservatory.db")).set_index("date",drop=True)
                MacroConnectorsObservatoryDF = MacroConnectorsObservatoryDF.fillna(0)
                MacroConnectorsObservatoryDF.corr().to_excel(self.AlternativeStorageLocation + 'MacroConnectorsObservatory\\' + pack+"_"+CheckedSpace + "_CorrMatrix_ExPost.xlsx")
                if RunSettings["MacroConnectorsObservatory"][2] == "RollMetric":
                    for metricIn in ["MI"]:#, "AdjMI", "NormMI"
                        for RollMode in [["RollWindow", 25], ["RollWindow", 250], ["ExpWindow", 25]]:
                            MacroConnectorsObservatoryID = pack +"_"+CheckedSpace + "_" + RunSettings["MacroConnectorsObservatory"][2]+"_"+metricIn+"_"+RollMode[0]+"_"+str(RollMode[1])
                            print(MacroConnectorsObservatoryID)
                            processDict = {
                                "data":MacroConnectorsObservatoryDF,#
                                "metric":metricIn,
                                "RollMode":RollMode,
                                "ExcludeSet":[x for x in MacroConnectorsObservatoryDF.columns if x not in TradedUniverse],
                                "PickleFileName":self.AlternativeStorageLocation+'MacroConnectorsObservatory\\' + MacroConnectorsObservatoryID
                                           }
                            ###############################"RUNNER" ##################################
                            processList.append(processDict)
                elif RunSettings["MacroConnectorsObservatory"][2] == "ReadMetric":
                    def returncolname(row, colnames, ReturnList):
                        if ReturnList == "MaxColumn":
                            return colnames[np.argmax(row.values)]
                        elif ReturnList == "SortedColumns":
                            return [x.split(" ")[0] for x in list(row.sort_values(ascending=False).index)]
                    TargetMacroConnectorsObservatoryID = "dIndicators_Rets_RollMetric_MI_RollWindow_250"
                    #TargetMacroConnectorsObservatoryID = "dIndicators_CTA_RollMetric_MI_RollWindow_250"
                    #TargetMacroConnectorsObservatoryID = "Indicators_CTA_RollMetric_MI_RollWindow_250"
                    with open(self.AlternativeStorageLocation+'MacroConnectorsObservatory\\' + TargetMacroConnectorsObservatoryID, 'rb') as f:
                        RollMetricPackDF = pickle.load(f)
                    MacroFactorConnectorsList = []
                    for MacroFactor in tqdm(MacroTargetsList):
                        MacroFactorDF = RollMetricPackDF[[x for x in RollMetricPackDF.columns if (x.split("_")[1] == MacroFactor)&(x.split("_")[0] not in MacroTargetsList+["SFR1 Comdty","SFR2 Comdty"])]]
                        MacroFactorConnectorsDF = MacroFactorDF.apply(lambda x: returncolname(x, MacroFactorDF.columns, "SortedColumns"), axis=1)
                        MacroFactorConnectorsDF.name = MacroFactor
                        #print(MacroFactor,MacroFactorConnectorsDF.iloc[-1])
                        MacroFactorDF.plot()
                        plt.show()
                        MacroFactorConnectorsList.append(MacroFactorConnectorsDF)
                    MacroFactorConnectorsPicklefile = open(self.AlternativeStorageLocation+'MacroConnectorsObservatory\\' + TargetMacroConnectorsObservatoryID + "_MacroFactorConnectors", 'wb')
                    pickle.dump(MacroFactorConnectorsList, MacroFactorConnectorsPicklefile)
                    MacroFactorConnectorsPicklefile.close()
            if RunSettings["MacroConnectorsObservatory"][2] == "RollMetric":
                #processList = [processList[0]]
                p = mp.Pool(mp.cpu_count())
                result = p.map(MacroConnectorsObservatoryRunner, tqdm(processList))
                p.close()
                p.join()
        ########################## BENCHMARK PORTFOLIOS ##########################################
        if RunSettings["RunBenchmarkPortfolios"]:
            "All Weather Portfolio"
            self.AllWeatherPortfolioContributions = self.rets[["ES1 Index", "TY1 Comdty", "FV1 Comdty", "CL1 Comdty", "C 1 Comdty", "GC1 Comdty"]]
            self.AllWeatherPortfolioContributions.index = pd.to_datetime(self.AllWeatherPortfolioContributions.index)
            self.AllWeatherPortfolioDF = 0.25 * self.AllWeatherPortfolioContributions["ES1 Index"]+\
                                        0.125 * self.AllWeatherPortfolioContributions["TY1 Comdty"]+\
                                        0.125 * self.AllWeatherPortfolioContributions["FV1 Comdty"]+\
                                        0.125 * self.AllWeatherPortfolioContributions["CL1 Comdty"]+\
                                        0.125 * self.AllWeatherPortfolioContributions["C 1 Comdty"]+\
                                        0.25 * self.AllWeatherPortfolioContributions["GC1 Comdty"]
            qs.reports.html(self.AllWeatherPortfolioDF, compounded=False, title="All Weather (Ray Dalio)", output=self.factSheetReportPath+"AllWeatherPortfolio.html")

############################################################################################################################################################
#### CALLERS ####
############################################################################################################################################################
"EXTRA FUNCTIONS"
def ExtraFunctions(mode, TargetAssets):
    LocalConn = sqlite3.connect("DataDeck.db")
    retsDF = pd.read_sql('SELECT * FROM rets', LocalConn).set_index('date', drop=True)#.tail(50)
    IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', LocalConn).set_index('date', drop=True)#.tail(50)

    if mode == "LookBacks":
        AllDF = pe.fd(pd.concat([retsDF, IndicatorsDF], axis=1).sort_index())

        OldLookBacks = pd.read_sql('SELECT * FROM LookBacks', sqlite3.connect("DataDeck.db"))
        OldLookBacks = OldLookBacks.set_index(OldLookBacks.columns[0], drop=True)
        OldLookBacks.index.names = ['date']

        TargetAssets_LookBacks = pe.DynamicSelectLookBack(AllDF[TargetAssets], RollMode="ExpWindow")

        LookBacks = pd.concat([OldLookBacks, TargetAssets_LookBacks], axis=1)
        LookBacks.index.names = ['date']
        LookBacks = LookBacks.ffill().bfill().sort_index()

        "Write LookBacks in DB"
        LookBacks.to_sql("LookBacks", LocalConn, if_exists='replace')
    elif mode == "MacroDecisionTrees":
        OldIsThere = False
        for MacroSplitter in ["GVZ Index"]:
            #"BFCIUS Index", "BFCIEU Index","VIX Index", "MOVE Index", "USSWIT2 Curncy", "EUSWI2 Curncy", "JPMVXYG7 Index","GVZ Index", "OVX Index"
            print(MacroSplitter)
            try:
                OldDecisionTrees_RV = pd.read_sql('SELECT * FROM DecisionTrees_RV', sqlite3.connect("DataDeck.db"))
                OldDecisionTrees_RV = OldDecisionTrees_RV.set_index(OldDecisionTrees_RV.columns[0], drop=True)
                OldDecisionTrees_RV.index.names = ['date']
                OldIsThere = True
                #print("OldDecisionTrees_RV")
                #print(OldDecisionTrees_RV)
            except Exception as e:
                print(e)
            #"FORCE ALL DECISION TREES CALCULACTIONS SINCE INCEPTION"
            #OldIsThere = False

            AllDF = pe.fd(pd.concat([retsDF, IndicatorsDF[MacroSplitter]], axis=1).sort_index())

            cc = [x for x in list(permutations(AllDF.columns, 2)) if x[0] == MacroSplitter]

            NewDecisionTrees_RV = pd.concat([ML.RollDecisionTree(ML.binarize(AllDF, targetColumns=[c[1]]), X=[c[0]], Y=[c[1]],
                                                                 #RollMode="RollWindow", st=25)
                                                                RollMode="ExpWindow")
                                                                for c in tqdm(cc)], axis=1, keys=cc)#RollMode="ExpWin"
            NewDecisionTrees_RV.columns = NewDecisionTrees_RV.columns.map('_'.join)
            NewDecisionTrees_RV.columns = [x.replace("_TreeThreshold", "") for x in NewDecisionTrees_RV.columns]

            if OldIsThere == True:
                DecisionTrees_RV = pd.concat([OldDecisionTrees_RV, NewDecisionTrees_RV], axis=1)
            else:
                DecisionTrees_RV = NewDecisionTrees_RV

            DecisionTrees_RV.index.names = ['date']
            DecisionTrees_RV = DecisionTrees_RV.ffill().bfill().sort_index()
            DecisionTrees_RV.to_sql("DecisionTrees_RV", LocalConn, if_exists='replace')
    elif mode == "CleanDecisionTrees":
        Raws = pd.read_sql('SELECT * FROM IndicatorsDeck_Raw', sqlite3.connect("DataDeck.db")).set_index("date", drop=True)
        DecisionTrees_RV = pd.read_sql('SELECT * FROM DecisionTrees_RV', sqlite3.connect("DataDeck.db")).set_index("date", drop=True)
        for c in DecisionTrees_RV.columns:
            DecisionTrees_RV.loc[Raws.loc[:,c.split("_")[0]].isna(), c] = None
        DecisionTrees_RV.to_sql("DecisionTrees_RV", LocalConn, if_exists='replace')
"DATABASE HANDLERS"
def Transfer_Data():
    #SourceDB_conn = sqlite3.connect("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/DataDeck_Temp.db")
    SourceDB_conn = sqlite3.connect("DataDeck.db")
    TargetDB_conn = sqlite3.connect("DataDeck.db")

    dfSource = pd.read_sql('SELECT * FROM LookBacksTest', SourceDB_conn).set_index('date', drop=True)
    dfTarget = pd.read_sql('SELECT * FROM LookBacks', TargetDB_conn).set_index('date', drop=True)

    for c in dfSource.columns:
        dfTarget[c] = dfSource[c]
    dfTarget.ffill().bfill().to_sql("LookBacks", SourceDB_conn, if_exists='replace')
def Insert_Tables_in_DB():
    LocalConn = sqlite3.connect("DataDeck.db")
    for pack in ["DecisionTrees_RV", "DEMAsTest", "DEMAs", "LookBacksTest", "LookBacks"]:
        print(pack)
        df = pd.read_csv(pack+".csv").set_index("date", drop=True).sort_index()
        df.to_sql(pack, LocalConn, if_exists='replace')
def SyncIndicatorsOnAllDBs(fromDB):
    for table in ['IndicatorsDeckTemp', 'IndicatorsDeck']:
        df = pd.read_sql('SELECT * FROM ' + table, sqlite3.connect(fromDB)).set_index('date', drop=True)
        for targetDB in ["DataDeck.db", "DataDeck_Mega.db", "DataDeck_Staged.db", "DataDeck_Research.db"]:
            if targetDB != fromDB:
                print(targetDB)
                df.to_sql(table, sqlite3.connect(targetDB), if_exists='replace')

#obj = DataDeck("DataDeck_Staged.db")
#obj = DataDeck("DataDeck_Research.db")
#obj = DataDeck("DataDeck_Mega.db")
#obj = DataDeck("DataDeck_Temp.db")
#obj = DataDeck("DataDeck_1950.db")

if __name__ == '__main__':

    RunSettings = {"RunDataMode": False,
                   "DataPatchers": False,
                   "LookBacksDEMAsCalcMode": [False,True,True],#LookbacksDEMAsAll, Lookbacks, DEMAs
                   "RunEcoDataMode": False,
                   "RunHurst": "Break", #Setup, Update, Read, Break
                   "ManifoldLearnersMode": "ManifoldPackUnpack", #Setup, Update, Read, ManifoldPackUnpack, Break
                   "MacroConnectorsObservatory": [False, "Break", "ReadMetric"],#"DataSetup", "RollMetric", "ReadMetric"
                   "RunBenchmarkPortfolios": False}

    obj = DataDeck("DataDeck.db", RefDataFetch=RunSettings["RunDataMode"])
    obj.Run(RunSettings)

    #ExtraFunctions("LookBacks", ["ES1 Index", "NQ1 Index"])
    #ExtraFunctions("MacroDecisionTrees", [])
    #ExtraFunctions("CleanDecisionTrees", [])

    #Transfer_Data()
    #Insert_Tables_in_DB()
    #SyncIndicatorsOnAllDBs("DataDeck.db")