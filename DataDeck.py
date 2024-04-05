import pandas as pd, numpy as np, matplotlib.pyplot as plt, pdblp, sqlite3, os, sys, time, pickle, glob, inspect, zipfile, shutil
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')
from tqdm import tqdm
from datetime import datetime, timedelta
from pyerb import pyerb as pe
from pyerbML import ML, ManSee
import quantstats as qs
from itertools import combinations, permutations
from ripser import Rips
import streamlit
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

"MULTIPROCESSING LOCAL RUNNERS"
def DecisionTrees_RV_Runner(argDict):
    
    AllDF = pe.fd(pd.concat([argDict["DT_RetsDF"], argDict["IndicatorsDF"][argDict["MacroSplitter"]]], axis=1).sort_index())
    AllDF = AllDF.loc[:,~AllDF.columns.duplicated()].copy()
    cc = [x for x in list(permutations(AllDF.columns, 2)) if x[0] == argDict["MacroSplitter"]]
    print(AllDF.columns)
    print(cc)

    if argDict["Mode"] == "Update":
        startLevel = AllDF.shape[0] - argDict['DT_Update_Depth']
    else:
        startLevel = argDict["st"]
        #startLevel = AllDF.shape[0] - 10 ### THIS IS A DEBUGGER !!!

    DecisionTrees_RV = pd.concat(
        [ML.RollDecisionTree(ML.binarize(AllDF, targetColumns=[c[1]]), X=[c[0]], Y=[c[1]], RollMode=argDict["RollMode"], st=startLevel) for
         c in tqdm(cc)], axis=1, keys=cc)
    DecisionTrees_RV.columns = DecisionTrees_RV.columns.map(','.join)
    DecisionTrees_RV.columns = [x.replace("_TreeThreshold", "") for x in DecisionTrees_RV.columns]
    DecisionTrees_RV.index.names = ['date']
    ###########################################################################################
    if argDict["Mode"] == "Update":
        #try:
        DecisionTrees_RV = pe.updatePickleDF(DecisionTrees_RV, argDict["WriteFilePath"])
        #except Exception as e:
        #    print(e)
    ###########################################################################################
    DTPickle = open(argDict["WriteFilePath"], 'wb')
    pickle.dump(DecisionTrees_RV, DTPickle)
    DTPickle.close()
    ###########################################################################################
    return "Success"
def ManifoldLearnerRunner(argDict):
    if argDict["ManifoldLearnerMode"] == "Learn":
        ManifoldLearnerPack = ManSee.gRollingManifold(argDict["ManifoldLearner"], argDict["ISpace"],
                                                  ProjectionMode=argDict["ProjectionMode"],
                                                  RollMode=argDict["RollMode"], st=argDict["st"])
    elif argDict["ManifoldLearnerMode"] == "Unpack":
        ManifoldLearnerPack = ManSee.ManifoldPackUnpack(argDict["ManifoldLearner"],
                                                        argDict['ManifoldLearnerPack'],
                                                        argDict['ProjectionStyle'],
                                                        TemporalExtraction=argDict["TemporalExtraction"])
    ################################################################################################################################################
    Picklefile = open(argDict["WriteFilePath"], 'wb')
    pickle.dump(ManifoldLearnerPack, Picklefile)
    Picklefile.close()
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

####################################################################################################################

class DataDeck:

    def __init__(self, DB, **kwargs):

        if "RefDataFetch" in kwargs:
            RefDataFetch = kwargs['RefDataFetch']
        else:
            RefDataFetch = True

        self.DB = DB
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.conn = sqlite3.connect(self.DB)
        self.AlternativeStorageLocationConn = sqlite3.connect(self.AlternativeStorageLocation+self.DB)
        self.AccCRNCY = "EUR"
        self.field = "PX_LAST" #"FUT_PX"
        self.factSheetReportPath = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\StrategiesFactSheets/"
        self.DataDeckExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
        self.AssetsTypes = pd.read_excel(self.DataDeckExcel, sheet_name="DataDeck", engine='openpyxl')[["Asset", "AIFMD Exposure Reporting Asset Type"]].dropna().set_index("Asset",drop=True)
        self.UpdateLookback = 10
        if self.DB == "DataDeck.db":
            self.ActiveStrategiesDF = pd.read_excel(self.DataDeckExcel, sheet_name="ActiveStrategies", engine='openpyxl')
        elif self.DB == "DataDeckUnAdjusted.db":
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
        #################### AssetsDashboard ################
        self.ActiveStrategiesFactorsControl = pd.read_excel("AssetsDashboard.xlsx",sheet_name="ActiveStrategiesFactorsControl").set_index("Asset", drop=True)
        #################### HEALTH CHECKS DB ################
        self.HealthChecksConn = sqlite3.connect(self.AlternativeStorageLocation+"HealthChecks.db")
        ######################################################
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
        ########################
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable', self.conn).set_index('index', drop=True)
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
        self.IndicatorsData = pd.read_excel("IndicatorsDashboard.xlsx",engine='openpyxl').dropna(subset=['Selector'])
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
            if (self.SinceInceptionFetch != 1)&(LastDateDF['LastDateDiffDays'].iloc[0] < 5):
                startReqDataDate = datetime.now() - timedelta(5)
                self.startReqDataDateStr = startReqDataDate.strftime("%Y")+startReqDataDate.strftime('%m')+startReqDataDate.strftime('%d')
                print("Last date = ", self.startReqDataDateStr)
                self.SinceInceptionFetch = 0
            for x in [
                [self.field, "DataDeck"],
                [self.field, "IndicatorsDeck"],
                [self.field, "forexData"],
                ["PX_VOLUME", "VolumeDeck"],
                ["CONTRACT_VALUE", "HistContractValues"]
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
                ############################## SORT VALUES #########################
                whatToUpdate = pd.Series(whatToUpdate).sort_values().tolist()
                ############################## UPDATE SINCE INCEPTION ??? #########################
                if self.SinceInceptionFetch != 0:
                    self.Hist_df = pd.DataFrame()
                else:
                    try:
                        self.Hist_df = pd.read_sql('SELECT * FROM ' + x[1], self.conn).set_index('date', drop=True)
                    except Exception as e:
                        print(e)
                        self.Hist_df = pd.DataFrame()
                ############################### MAIN UPDATE ACTION ##################################################
                df = self.con.bdh(whatToUpdate, x[0], self.startReqDataDateStr, '21000630')
                #####################################################################################################
                tomorrow = pd.Timestamp.today().date() + pd.Timedelta(days=1)
                try:
                    df = df.drop(tomorrow)
                except Exception as e:
                    print(e)
                #####################################################################################################
                df.columns = [x[0] for x in df.columns]
                df.to_sql(x[1] + "_Raw", self.conn, if_exists='replace')
                if x[1] not in ["VolumeDeck"]:
                    df = df.ffill()
                if x[1] not in ["IndicatorsDeck", "VolumeDeck"]:
                    df = df.bfill()
                if x[1] in ["VolumeDeck"]:
                    df = df.fillna(0)
                    ##############################################################################################################################
                    print("Patching SOFR / ED Volumes with FF ones ... ")
                    for dropC in ["SFR1 Comdty","SFR2 Comdty","SFR3 Comdty","SFR4 Comdty","ED1 Comdty","ED2 Comdty","ED3 Comdty","ED4 Comdty"]:
                        try:
                            df = df.drop(dropC,axis=1)
                        except Exception as e:
                            #print(e)
                            pass
                    for fillC in [["SFR1 Comdty", "FF1 Comdty"],
                                  ["SFR2 Comdty", "FF2 Comdty"],
                                  ["SFR3 Comdty", "FF3 Comdty"],
                                  ["SFR4 Comdty", "FF4 Comdty"],
                                  ##############################
                                  ["ED1 Comdty", "FF1 Comdty"],
                                  ["ED2 Comdty", "FF2 Comdty"],
                                  ["ED3 Comdty", "FF3 Comdty"],
                                  ["ED4 Comdty", "FF4 Comdty"]]:
                        df[fillC[0]] = df[fillC[1]]
                    print("Done ... ")
                    ##############################################################################################################################
                    print("Calculating Volume EMAs ... ")
                    for emaL in [3,5]:
                        pe.ema(df.fillna(0), nperiods=emaL).round(0).to_sql(x[1] + "_EMA"+str(emaL), self.conn, if_exists='replace')
                    ##############################################################################################################################
                    print("Calculating Daily Turnover based on Current Futures Contract Sizes ... ")
                    DailyTurnoverDF = pd.DataFrame(None, index=df.index, columns=df.columns)
                    for turnC in df.columns:
                        DailyTurnoverDF[turnC] = df[turnC] * self.ActiveAssetsReferenceData.loc[turnC, "FUT_CONT_SIZE"]
                    DataDeckDF = pd.read_sql('SELECT * FROM DataDeck', self.conn).set_index("date", drop=True)
                    DailyTurnoverDF[[x for x in DataDeckDF.columns if x not in DailyTurnoverDF.columns]] = 0
                    DailyTurnoverDF.to_sql("DailyTurnoverDF", self.conn, if_exists='replace')
                    ##############################################################################################################################
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
                ###############################################################################################
                print("Make sure to match the columns with the DataDeck!")
                if x[1] in ["VolumeDeck", "HistContractValues"]:
                    DataDeckDF = pd.read_sql('SELECT * FROM DataDeck', self.conn).set_index("date", drop=True)
                    if x[1] == "VolumeDeck":
                        replacer = 0
                    elif x[1] == "HistContractValues":
                        replacer = None
                    allDF[[x for x in DataDeckDF.columns if x not in allDF.columns]] = replacer
                ###############################################################################################
                allDF.to_sql(x[1], self.conn, if_exists='replace')
                print(self.DB + " Successfully Setup the Dataset !")
                self.con.stop()
        ############################# EURODOLLAR FUTURES TO SOFR FUTURES ###################################################
        "https://www.thestreet.com/investing/futures/shift-from-eurodollar-to-sofr-accelerating"
        TransitionDate = "2018-05-07 00:00:00" # Officially announced earlier though by CME
        ####################################################################################################################
        if RunSettings["DataPatchers"]:
            print("Patching Data ... !")
            for patchPack in [["DataDeck","DataDeck_Eurodollar_Futures.csv"],
                              ["HistContractValues", "DataDeck_Eurodollar_Futures_Historical_Contact_Values.csv"]]:
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
        ######################## RETURNS ###################################################################################
        print("Calculating Rets ... ")
        self.rets = pe.dlog(pd.read_sql('SELECT * FROM DataDeck', self.conn).set_index("date", drop=True)).fillna(0)
        self.rets.to_sql("rets", self.conn, if_exists='replace')
        print("done ... ")
        ####################### INDICATORS #################################################################################
        print("Reading Indicators ... ")
        IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index('date', drop=True)
        IndicatorsDF = pe.IndicatorsHandler(IndicatorsDF, self.IndicatorsData,SelectorHandler=[[0, "exclude"], [2, "diff"]])
        Raws = pd.read_sql('SELECT * FROM IndicatorsDeck_Raw', sqlite3.connect("DataDeck.db")).set_index("date",drop=True)
        print("done ... ")
        ####################################################################################################################
        if RunSettings["AlternativeDataFields"][0]:
            if RunSettings["AlternativeDataFields"][1]["DataFetch"]:
                AlternativeDataFields_con = pdblp.BCon(debug=True, port=8194, timeout=200000).start()
                AlternativeDataFields_Pack = pd.read_excel(self.DataDeckExcel, sheet_name="AlternativeDataFields",engine='openpyxl')
                for idx,row in AlternativeDataFields_Pack.iterrows():
                    print("AlternativeDataFields_Pack : ", row["Field"])
                    AssetsList = row["Assets"].split(",")
                    df = AlternativeDataFields_con.bdh(AssetsList, row["Field"], self.startReqDataDateStr, '21000630')
                    df.columns = [x[0] for x in df.columns]
                    df.to_sql("AlternativeDataFields_"+row["Field"], self.conn, if_exists='replace')
                AlternativeDataFields_con.stop()
            ###############################################################################################################
            if RunSettings["AlternativeDataFields"][1]["AlternativeIndicators"]:
                ##################### EARNINGS YIELDS OVER BONDS ONES ########################
                EARN_YLD_DF = pd.read_sql('SELECT * FROM AlternativeDataFields_EARN_YLD', self.conn).set_index("date", drop=True)
                for premiumOverWhat in [
                    "USGG5YR Index", "USGG3M Index", "USGG2YR Index", "USGG10YR Index", "USGB090Y Index",
                    "GTEUR5Y Govt", "GTEUR2Y Govt", "GTEUR10Y Govt",
                    "EUSA5 Curncy", "EUSA2 Curncy", "EUSA10 Curncy",
                ]:
                    for e_y_baseAsset in EARN_YLD_DF.columns:
                        IndicatorsDF[e_y_baseAsset+"@EARNYLDOVER@"+premiumOverWhat] = (EARN_YLD_DF[e_y_baseAsset]-IndicatorsDF[premiumOverWhat]).ffill().bfill()
                AlternativeDataFieldsIndicators = IndicatorsDF[[x for x in IndicatorsDF.columns if "@EARNYLDOVER@" in x]]
                ##################### MOVE / VIX #############################################
                AlternativeDataFieldsIndicators["MOVE@VIX@RATIO"] = IndicatorsDF["MOVE Index"] / IndicatorsDF["VIX Index"]
                ##################### IBOXHY Index / IBOXIG Index #############################################
                AlternativeDataFieldsIndicators["IBOXHY@IBOXIG@RATIO"] = IndicatorsDF["IBOXHY Index"] / IndicatorsDF["IBOXIG Index"]
                ##################### SAVE TO DATABASE ##################################################
                AlternativeDataFieldsIndicators.to_sql("AlternativeDataFieldsIndicators", self.conn, if_exists='replace')
                ##################### PLOTS ##################################################
                #fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True)
                #AlternativeDataFieldsIndicators.plot(ax=ax[0],legend=None)
                #pe.cs(self.rets[["ES1 Index", "NQ1 Index"]]).plot(ax=ax[1])
                #plt.show()
        ################### AlternativeDataFieldsIndicators ################################################################
        print("Reading Indicators ... ")
        AlternativeDataFieldsIndicators = pd.read_sql('SELECT * FROM AlternativeDataFieldsIndicators', self.conn).set_index('date', drop=True)
        print("done ... ")
        ########################## Update IndicatorsDF with Alternative Indicators ##################################################
        IndicatorsDF = pd.concat([IndicatorsDF, AlternativeDataFieldsIndicators], axis=1).sort_index()
        ########################## HEDGE RATIOS RVs ###################################################
        if RunSettings["HedgeRatiosRVs"][0]:
            print("Calculating HedgeRatiosRVs ... ")
            for TargetRVsSpace in [
                #"Endurance",
                #"Coast",
                "Brotherhood",
                #"ShoreDM",
                #"ShoreEM",
                #"Valley",
                #"Dragons",
                #"Lumen"
                ]:
                LookbackRets = pd.read_sql('SELECT * FROM LookbackRets', self.conn).set_index("date", drop=True).sort_index()
                targetPairs = list(permutations(pe.getMoreFuturesCurvePoints([x for x in self.ActiveStrategiesDF[TargetRVsSpace].dropna().tolist() if x in self.FuturesTable["Point_1"].tolist()], self.FuturesTable.set_index("Point_1", drop=True),[2, 3]), 2))
                targetPairs = [x for x in targetPairs if (x[0] in self.rets.columns) & (x[1] in self.rets.columns)]
                if TargetRVsSpace in ["Brotherhood"]:
                    targetPairs = [x for x in targetPairs if (self.AssetsTypes.loc[self.FuturesTable["Point_1"].iloc[pe.getIndexes(self.FuturesTable, x[0])[0][0]]].values[0] != self.AssetsTypes.loc[self.FuturesTable["Point_1"].iloc[pe.getIndexes(self.FuturesTable, x[1])[0][0]]].values[0])]
                for RollModeIn in [["HedgeRatio_Expanding", 25]]:#["HedgeRatio_Rolling", 250]
                    print("Calculating Hedge Ratios : ", RollModeIn, ", ", TargetRVsSpace)
                    TargetHRsPickleFile = self.AlternativeStorageLocation + "HedgeRatios\\HedgeRatioDF_" + TargetRVsSpace + "_" + str(RollModeIn[0]) + "_" + str(RollModeIn[1])
                    #out = pe.readPickleDF(TargetHRsPickleFile); out.plot(); plt.show()
                    if RunSettings["HedgeRatiosRVs"][1]["Mode"] == "Setup":
                        HedgeRatioDF = pe.RV(LookbackRets, ConnectionMode="+", mode="HedgeRatio_Expanding", n=25, RVspace="specificPairs", targetPairs=targetPairs)
                    elif RunSettings["HedgeRatiosRVs"][1]["Mode"]=="Update":
                        NewHedgeRatioDF = pe.RV(LookbackRets, ConnectionMode="+", mode="HedgeRatio_Expanding", n=LookbackRets.shape[0]-5,RVspace="specificPairs", targetPairs=targetPairs)
                        HedgeRatioDF = pe.updatePickleDF(NewHedgeRatioDF,TargetHRsPickleFile)
                    ################## SAVE DATA #######################
                    pe.savePickleDF(HedgeRatioDF,TargetHRsPickleFile)
        ########################## TDA ####################################
        if RunSettings["Perform_TDA"][0]:
            if RunSettings["Perform_TDA"][1]["Wasserstein_Distances"]:

                selAsset = "NQ1 Index"
                #selAsset = "TY1 Comdty"
                st = 5
                rips = Rips(maxdim=2)
                wasserstein_dists = pe.compute_wasserstein_distances(self.rets[selAsset].values, st, rips)
                print(self.rets.shape)
                print(len(wasserstein_dists))
                dataPlot = pd.DataFrame(None,index=self.rets.index)
                dataPlot["Asset"] = self.rets[selAsset]
                dataPlot["WDists"] = None
                print(dataPlot.index[st:])
                dataPlot.loc[dataPlot.index[st+1:],"WDists"] = wasserstein_dists
                print(dataPlot)
                fig,ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                pe.cs(dataPlot["Asset"]).plot(ax=ax[0])
                dataPlot["WDists"].plot(ax=ax[1])
                plt.show()
        ########################## LOKBACKS AND DEMAs CALCULATORS ####################################
        if RunSettings["LookBacksDEMAsCalcMode"][0]:
            print("Calculating LookBacks ... !")
            ###################### HANDLE LOOKBACK RETS AFTER PATCHERS ####################################
            LookbackRets = self.rets
            for i in range(1, 5):
                LookbackRets.loc[:TransitionDate, "SFR" + str(i) + " Comdty"] = LookbackRets.loc[:TransitionDate,"ED" + str(i) + " Comdty"]
            LookbackRets.to_sql("LookbackRets", self.conn, if_exists='replace')
            ###############################################################################################################
            DataDeckIndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index('date', drop=True)
            IRDs = DataDeckIndicatorsDF[[x for x in DataDeckIndicatorsDF.columns if "IS Curncy" in x]]
            dIRDs = pe.d(IRDs)
            ###############################################################################################################
            if RunSettings["LookBacksDEMAsCalcMode"][1] in ["Setup", "Update"]:
                "LOOKBACK SPECS !"
                ID = RunSettings["LookBacksDEMAsCalcMode"][3][1]
                SpecsID = RunSettings["LookBacksDEMAsCalcMode"][3][2]
                LookbackMethod = RunSettings["LookBacksDEMAsCalcMode"][3][3]
                if LookbackMethod == "BB":
                    LookbackStdList = RunSettings["LookBacksDEMAsCalcMode"][3][4]
                else:
                    LookbackStdList = [0.5, 1, 1.5, 2, 2.5, 3]
                ###########################################################################
                if SpecsID == "DefaultSingleLookBack":
                    Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly","OptimizedMetric": "Sharpe", "OptimizationMode": "Raw",'LookBackSelectorMode':'SingleLookBack'} # DEFAULT SINGLE LOOKBACK!!!
                elif SpecsID == "FastTrendSingleLookBack":
                    Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "FastTrendFollowingOnly","OptimizedMetric": "Sharpe", "OptimizationMode": "Raw",'LookBackSelectorMode':'SingleLookBack'} # DEFAULT SINGLE LOOKBACK!!!
                elif SpecsID == "LookBacksList":
                    Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "Range-10-500-10","OptimizedMetric": "Sharpe", "OptimizationMode": "Raw",'LookBackSelectorMode':'LookBacksList'} # DEFAULT LOOKBACKs LIST!!!
                elif SpecsID == "MRS1":
                    Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "MeanReversionOnlyShift1","OptimizedMetric": "Sharpe", "OptimizationMode": "Abs",'LookBackSelectorMode': 'SingleLookBack'}
                #Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly", "OptimizedMetric": "Sharpe", "OptimizationMode":"Abs",'LookBackSelectorMode':'SingleLookBack'}
                # Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly", "OptimizedMetric": "Sortino", "OptimizationMode":"Raw",'LookBackSelectorMode':'SingleLookBack'}
                # Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly", "OptimizedMetric": "Sortino", "OptimizationMode":"Abs",'LookBackSelectorMode':'SingleLookBack'}
                # Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly", "OptimizedMetric": "Calmar", "OptimizationMode":"Raw",'LookBackSelectorMode':'SingleLookBack'}
                # Specs = {"RollMode": "ExpWindow", "DynamicsSpace": "SlowTrendFollowingOnly", "OptimizedMetric": "Calmar", "OptimizationMode":"Abs",'LookBackSelectorMode':'SingleLookBack'}
                if (RunSettings["LookBacksDEMAsCalcMode"][3][0] == "Rets")|(RunSettings["LookBacksDEMAsCalcMode"][3][0] == "RetsLive"):
                    LookbackTargets = LookbackRets.sort_index()
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "Rets_Point_1":
                    LookbackTargets = LookbackRets.sort_index()[[x for x in LookbackRets.columns if x in self.FuturesTable["Point_1"].tolist()]]
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "Rets_Point_2":
                    LookbackTargets = LookbackRets.sort_index()[[x for x in LookbackRets.columns if x in self.FuturesTable["Point_2"].tolist()]]
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "Rets_Point_3":
                    LookbackTargets = LookbackRets.sort_index()[[x for x in LookbackRets.columns if x in self.FuturesTable["Point_3"].tolist()]]
                    #pack = '_'.join([str(k + "_" + v) for k, v in Specs.items()])
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "RVs":
                    # TargetRVsSpace = "Endurance"
                    # TargetRVsSpace = "Coast"
                    #TargetRVsSpace = "Brotherhood"
                    # TargetRVsSpace = "ShoreDM"
                    # TargetRVsSpace = "ShoreEM"
                    TargetRVsSpace = "Valley"
                    # TargetRVsSpace = "Dragons"
                    # TargetRVsSpace = "Lumen"
                    TargetPickleFile = self.AlternativeStorageLocation + "HedgeRatios\\HedgeRatioDF_" + TargetRVsSpace + "_HedgeRatio_Expanding_25"
                    HRsPickle = open(TargetPickleFile, 'rb')
                    HedgeRatioDF = pickle.load(HRsPickle)
                    HRsPickle.close()
                    subSel = [3000,HedgeRatioDF.shape[1]]; HedgeRatioDF = HedgeRatioDF.iloc[:,range(subSel[0], subSel[1])]
                    ####################################################################################################################
                    SpecificAsset = "TY1 Comdty"
                    DoL = "Driver"
                    if DoL == "Driver":
                       DoLid = 0
                    elif DoL == "Lagger":
                       DoLid = 1
                    HedgeRatioDF = HedgeRatioDF[[x for x in HedgeRatioDF.columns if x.split("_")[DoLid]==SpecificAsset]]
                    ####################################################################################################################
                    "Calculate RVs"
                    RVs = pd.DataFrame(1, index=HedgeRatioDF.index, columns=HedgeRatioDF.columns)
                    for c in HedgeRatioDF.columns:
                       cSplit = c.split("_")
                       try:
                           RVs[c] = LookbackRets[cSplit[0]] + HedgeRatioDF[c] * LookbackRets[cSplit[1]]
                       except Exception as e:
                           print(e)
                    ########################################################################
                    LookbackTargets = LookbackRets.sort_index(); ID = Specs["DynamicsSpace"] + "_" + Specs["SharpeSpace"] # DEFAULT!!!
                    LookbackTargets = pd.concat([LookbackRets, dIRDs], axis=1).sort_index()
                    LookbackTargets = RVs; ID = TargetRVsSpace + "_RV_"+str(subSel)
                    ########################################################################
                    LookbackTargets = RVs[[x for x in RVs if x.split("_")[0]==SpecificAsset]]; ID = TargetRVsSpace + "_RV_" + SpecificAsset + "_"+DoL
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "ManifoldLearner":
                    ###############################################################################################################
                    ########################################################################
                    ManifoldUnpacked = pe.readPickleDF(self.AlternativeStorageLocation + "/ManifoldLearners/" + ID)
                    NativeSpace = ManifoldUnpacked[0]
                    MPsList = ManSee.getManifoldProjections(ManifoldUnpacked, ProjectionShift=0)
                    DT_Proj_Rets_List = []
                    for Projection in MPsList:
                        DT_Proj_Rets_List.append(pe.rs(Projection))
                    LookbackTargets = pd.concat(DT_Proj_Rets_List, axis=1).sort_index()
                    LookbackTargets.columns = ["Proj_" + str(x) for x in range(len(LookbackTargets.columns))]
                elif RunSettings["LookBacksDEMAsCalcMode"][3][0] == "Test":
                    #########################################################################
                    LookbackTargets = LookbackRets[["ES1 Index", "NQ1 Index", "CC1 Comdty", "UX1 Index"]].sort_index()
                    ###############################################################################################################
                print("(Shape) LookbackTargets ", Specs, " : " + ID + " = ", LookbackTargets.shape)
                ###############################################################################################################
                TargetPickleFile = self.AlternativeStorageLocation + "LookbacksPacks/"+ID+"_"+SpecsID+"_LookBacksPack"
                if RunSettings["LookBacksDEMAsCalcMode"][1] == "Setup":
                    [LookBacks, LookBacksDirections] = pe.DynamicSelectLookBack(LookbackTargets,
                                                                                method=LookbackMethod,
                                                                                stdList=LookbackStdList,
                                                                                RollMode=Specs['RollMode'],DynamicsSpace=Specs["DynamicsSpace"],
                                                                                OptimizedMetric=Specs["OptimizedMetric"], OptimizationMode=Specs['OptimizationMode'],
                                                                                LookBackSelectorMode=Specs['LookBackSelectorMode'])
                elif RunSettings["LookBacksDEMAsCalcMode"][1] == "Update":
                    print("Updating Lookbacks : ", "LookbacksPacks/"+ID+SpecsID)
                    NewLookBacksPack = pe.DynamicSelectLookBack(LookbackTargets,
                                                                method=LookbackMethod,
                                                                stdList=LookbackStdList,
                                                                RollMode=Specs['RollMode'],DynamicsSpace=Specs["DynamicsSpace"],
                                                                OptimizedMetric=Specs["OptimizedMetric"], OptimizationMode=Specs['OptimizationMode'],
                                                                LookBackSelectorMode=Specs['LookBackSelectorMode'],
                                                                st=LookbackTargets.shape[0] - self.UpdateLookback)
                    PickledLookBacksPack = pe.readPickleDF(TargetPickleFile)
                    pe.savePickleDF(PickledLookBacksPack, self.AlternativeStorageLocation + "LookbacksPacks/BeforeUpdate/"+ID+SpecsID+"_LookBacksPack")
                    LookBacks = pe.updateDF(PickledLookBacksPack[1], NewLookBacksPack[0])
                    LookBacksDirections = pe.updateDF(PickledLookBacksPack[2], NewLookBacksPack[1])
                ###############################################################################################################
                LookBacks = LookBacks.ffill().bfill().sort_index()#[self.rets.columns]
                LookBacksDirections = LookBacksDirections.fillna(0)#[self.rets.columns]
                #LookBacks.iloc[:,0:5].plot()
                #plt.show()
                ###############################################################################################################
                pe.savePickleDF([LookbackTargets, LookBacks, LookBacksDirections], TargetPickleFile)
                #LookbackTargets.tail(50).to_sql("Latest_LookbackTargets_" + ID + "_" + pack, self.conn, if_exists='replace')
                #LookBacks.tail(50).to_sql("Latest_LookBacks_" + ID + "_" + pack, self.conn, if_exists='replace')
                #LookBacksDirections.tail(50).to_sql("Latest_LookBacksDirections_" + ID + "_" + pack, self.conn,if_exists='replace')
            print("LookBacks Ready ... !")
            ###################################################################################################################################
            print("Calculating DEMAs ... !")
            if RunSettings["LookBacksDEMAsCalcMode"][2]:
                for filename in glob.glob(self.AlternativeStorageLocation + "LookBacksPacks/*_LookBacksPack"):
                    pack = filename.split("/")[-1]
                    LookBacksPack = pe.readPickleDF(self.AlternativeStorageLocation + pack)
                    print(pack.split("\\")[-1], ", len(LookBacksPack) : ", len(LookBacksPack))
                    "Treat Zeros on Lookbacks"
                    LookBacksPack[1][LookBacksPack[1]==0] = 1
                    try:
                        ####################################################################################################
                        DEMAs = pe.dema(LookBacksPack[0], LookBacksPack[1]).sort_index()
                        #DEMAs = pe.dema(LookBacksPack[0], LookBacksPack[1], LookBacksDirections=LookBacksPack[2]).sort_index()
                        DEMAs.index.names = ['date']
                        out = pe.savePickleDF(DEMAs, self.AlternativeStorageLocation + "DEMAs/" + pack.split("\\")[-1]+"_DEMAs")
                        #DEMAs.tail(50).to_sql("Latest_DEMAs_"+pack, self.conn,if_exists='replace')
                        print("DEMAs for "+pack+" Ready ... !")
                    except Exception as e:
                        print(e)
                        print(LookBacksPack[1])
                        print("POSSIBLE NEGATIVES in Lookbacks ???! PLS CHECK !!!")
                        time.sleep(300000)
        ########################## DECISION TREES ###################################################
        if RunSettings["DecisionTrees"][0]:
            print("Running Decision Trees ... ")
            ######################################################################################################################################################
            RunMode = RunSettings["DecisionTrees"][1]
            DT_Space = RunSettings["DecisionTrees"][2][0]
            DT_Space_ID = RunSettings["DecisionTrees"][2][1]
            DT_Update_Depth = RunSettings["DecisionTrees"][3]
            ######################################################################################################################################################
            if "Rets" in DT_Space:
                DT_RetsDF = self.rets#.tail(50)
                if RunMode == "Setup":
                    AlreadyCalculatedAlphasFile = "__DefaultSingleLookBack_Alphas_DT_Filters_sh.xlsx"
                    DT_Alphas = pd.read_excel(self.AlternativeStorageLocation+"DT_Filters_Alphas/"+AlreadyCalculatedAlphasFile)
                    try:
                        AlreadyCalculatedMacros = list(set(DT_Alphas["MacroFactor"].dropna().tolist()))
                    except Exception as e:
                        print(e)
                    #WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns]
                    #WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns if x not in AlreadyCalculatedMacros]
                    WhichMarcoSplitterToScan = self.IndicatorsData["Rerun"].dropna().tolist()
                    #WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns if "IS Curncy" in x]#@EARNYLDOVER@
                    #WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns if ("MOVE@VIX@RATIO" in x)|("IBOXHY@IBOXIG@RATIO" in x)]
                elif RunMode == "Update":
                    DT_Alphas = pd.read_excel(self.DataDeckExcel, sheet_name="ActiveStrategiesFactorsControl")[["Asset","SingleDecisionTreesControllers_Positive","SingleDecisionTreesControllers_Negative"]]
                    DT_Alphas["SingleDecisionTreesControllers_Positive"] = DT_Alphas["SingleDecisionTreesControllers_Positive"].str.replace("_Lower", "").str.replace("_Upper", "")
                    DT_Alphas["SingleDecisionTreesControllers_Negative"] = DT_Alphas["SingleDecisionTreesControllers_Negative"].str.replace("_Lower", "").str.replace("_Upper", "")
                    MacroFactor_Asset = pd.concat([DT_Alphas["SingleDecisionTreesControllers_Positive"]+"_"+DT_Alphas["Asset"],DT_Alphas["SingleDecisionTreesControllers_Negative"]+"_"+DT_Alphas["Asset"]])
                    ######################################################################################################################################################
                    AlreadyCalculatedMacros = list(set(list(set(DT_Alphas["SingleDecisionTreesControllers_Positive"].dropna().tolist()))+list(set(DT_Alphas["SingleDecisionTreesControllers_Negative"].dropna().tolist()))))
                    plainedFactorDataList = []
                    for mFactor in AlreadyCalculatedMacros:
                        subData = MacroFactor_Asset.loc[MacroFactor_Asset.str.contains(mFactor).fillna(False)].str.split("_").str[1].dropna().reset_index().drop("index",axis=1)
                        subData.columns = [mFactor]
                        plainedFactorDataList.append(subData)
                    plainedFactorDataDF = pd.concat(plainedFactorDataList, axis=1)
                    plainedFactorDataDF.to_excel(self.AlternativeStorageLocation+"HealthChecksPacks/plainedFactorDataDF_"+DT_Space+".xlsx", index=False)
                    #WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns]
                    WhichMarcoSplitterToScan = [x for x in IndicatorsDF.columns if x in AlreadyCalculatedMacros]
                    #WhichMarcoSplitterToScan = [x for x in WhichMarcoSplitterToScan if "@EARNYLDOVER@" in x]
                    #WhichMarcoSplitterToScan = [x for x in WhichMarcoSplitterToScan if "GBP" in x]
            elif DT_Space == "ManifoldLearner":
                ManifoldUnpacked = pe.readPickleDF(self.AlternativeStorageLocation+"/ManifoldLearners/"+DT_Space_ID)
                DT_Space += "_"+DT_Space_ID
                NativeSpace = ManifoldUnpacked[0]
                MPsList = ManSee.getManifoldProjections(ManifoldUnpacked, ProjectionShift=0)
                DT_Proj_Rets_List = []
                for Projection in MPsList:
                    DT_Proj_Rets_List.append(pe.rs(Projection))
                DT_RetsDF = pd.concat(DT_Proj_Rets_List, axis=1).sort_index()
                DT_RetsDF.columns = ["Proj_"+str(x) for x in range(len(DT_RetsDF.columns))]
                #pe.cs(DT_RetsDF).plot()
                #plt.show()
                AlreadyCalculatedAlphasFile = "Start.xlsx"
            ######################################################################################################################################################
            "Store Previous Versions in subfolder"
            source = self.AlternativeStorageLocation + "DecisionTrees/"+DT_Space+"/"
            destination = source+"BeforeUpdate/"
            if not os.path.exists(destination):
                os.makedirs(destination)
            files_list = os.listdir(source)
            for files in files_list:
                if files != "BeforeUpdate":
                    shutil.copy(source+files, destination)
            ######################################################################################################################################################
            for MacroMode in [""]:#"diff",""
                ##################################################################################################################################
                if MacroMode == "diff":
                    IndicatorsDF = pe.d(IndicatorsDF).fillna(0)
                    Raws = pe.d(Raws).fillna(0)
                ##################################################################################################################################
                print("WhichMarcoSplitterToScan (length) : ", len(WhichMarcoSplitterToScan))
                processList = []
                if DT_Space == "RetsLive":
                    RollModeScanList = [["RollWindow", 250]]#["ExpWindow", 25], ["RollWindow", 250], ["RollWindow", 500], ["RollWindow", 1000]
                elif DT_Space == "Rets":
                    RollModeScanList = [["RollWindow", 25]]#["ExpWindow", 25],, ["RollWindow", 50], ["RollWindow", 250]
                elif DT_Space == "RetsExpWindow":
                    RollModeScanList = [["ExpWindow", 25]]#["ExpWindow", 25],, ["RollWindow", 50], ["RollWindow", 250]
                    #WhichMarcoSplitterToScan = WhichMarcoSplitterToScan[:100]
                ##################################################################################################################################
                for RollModeIn in RollModeScanList:
                    for MacroSplitter in WhichMarcoSplitterToScan:
                        DT_ID = "DecisionTrees_RV"+"_"+DT_Space.replace("Live","")+"_"+RollModeIn[0]+"_"+str(RollModeIn[1])+"_"+MacroSplitter
                        argDict = {
                            "Mode": RunMode,
                            "DT_RetsDF": DT_RetsDF,
                            "IndicatorsDF": IndicatorsDF,
                            "Raws": Raws,
                            "RollMode": RollModeIn[0],
                            "st": RollModeIn[1],
                            "MacroSplitter": MacroSplitter,
                            "DT_Update_Depth": DT_Update_Depth,
                            "WriteFilePath": self.AlternativeStorageLocation + "DecisionTrees/"+DT_Space+"/"+MacroMode+DT_ID
                        }
                        processList.append(argDict)
            ###################### MUTLIPROCESSING DT ##########################################################################################
            print("len(processList) : ", len(processList))
            p = mp.Pool(mp.cpu_count() - 2)
            result = p.map(DecisionTrees_RV_Runner, tqdm(processList))
            p.close()
            p.join()
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
            LookbackRets = pd.read_sql('SELECT * FROM LookbackRets', self.conn).set_index("date", drop=True).drop([x for x in ["ED1 Comdty", "ED2 Comdty", "ED3 Comdty", "ED4 Comdty"]], axis=1).sort_index()
            #WhatToHurstID = "Assets"
            WhatToHurstID = "RVsHedgeRatios"
            ################################################################
            #TargetRVsSpace = "Dragons"
            #TargetRVsSpace = "Endurance"
            #TargetRVsSpace = "Coast"
            #TargetRVsSpace = "Shore"
            #TargetRVsSpace = "Lumen"
            TargetRVsSpace = "Valley"
            #TargetRVsSpace = "Fidei"
            #TargetRVs = [x for x in self.rets.columns if x in self.ActiveStrategiesDF[TargetRVsSpace].tolist()]
            TargetRVs = pe.getMoreFuturesCurvePoints([x for x in self.ActiveStrategiesDF[TargetRVsSpace].dropna().tolist() if x in self.FuturesTable["Point_1"].tolist()], self.FuturesTable.set_index("Point_1", drop=True), [2, 3])
            if TargetRVsSpace == "Valley":
                excCols = ["ED1 Comdty", "ED2 Comdty", "ED3 Comdty", "SFR1 Comdty", "SFR2 Comdty", "SFR3 Comdty", ]
                TargetRVs = [x for x in [c for c in list(permutations(TargetRVs, 2))] if (x[0] not in excCols) & (x[1] not in excCols)]
            else:
                TargetRVs = [c for c in list(permutations(TargetRVs, 2))]
            ################################################################
            HurstConn = sqlite3.connect(self.AlternativeStorageLocation + "RollHurst.db")
            if RunSettings["RunHurst"] in ["Setup", "Update"]:
                ################################################################################################################################
                if WhatToHurstID == "Assets":
                    WhatToHurst = LookbackRets[[x for x in LookbackRets if x in self.FuturesTable["Point_1"].tolist()]]
                elif WhatToHurstID == "RVsHedgeRatios":
                    HRs = pe.RV(LookbackRets, ConnectionMode="+", mode="HedgeRatio_Expanding", n=25, RVspace="specificPairs", targetPairs=TargetRVs)
                    #HRs = pd.read_sql('SELECT * FROM HRs_Shore'+TargetRVsSpace, sqlite3.connect(self.AlternativeStorageLocation+"StrategiesCrossValidation.db")).set_index('date', drop=True)
                    WhatToHurst = pd.DataFrame(1, index=HRs.index, columns=HRs.columns)
                    for c in HRs.columns:
                        cSplit = c.split("_")
                        WhatToHurst[c] = LookbackRets[cSplit[0]] + HRs[c] * LookbackRets[cSplit[1]]
                    WhatToHurst = WhatToHurst.fillna(0)
                WhatToHurstID += TargetRVsSpace
                ################################################################################################################################
                for RollMode in [['ExpWindow', 25]]:#['ExpWindow', 25], ["RollWindow", 250]
                    print("Running Hurst on "+WhatToHurstID+" : RollMode = ", RollMode)
                    ##############################################################################################################
                    RollHurst = pe.gRollingHurst(WhatToHurst, RollMode=RollMode[0], st=RollMode[1])
                    out = pe.savePickleDF(RollHurst, self.AlternativeStorageLocation + 'RollHurst_' + WhatToHurstID)
                    ##############################################################################################################
                    SpreadsStats = []
                    for asset in RollHurst.columns:
                        SpreadsStats.append([asset, RollHurst[asset].mean(), RollHurst[asset].max(), RollHurst[asset].min()])
                    ##############################################################################################################
                    SpreadsStatsDF = pd.DataFrame(SpreadsStats, columns=["Asset", "AvgHurst", "MaxHurst", "MinHurst"]).set_index("Asset", drop=True)
                    SpreadsStatsDF.to_sql("RollHurstSpreadsStats_"+WhatToHurstID+"_"+RollMode[0]+"_"+str(RollMode[1]), HurstConn, if_exists='replace')
            elif RunSettings["RunHurst"] == "Read":
                #TargetAssets = ["SFR1 Comdty", "ED1 Comdty", "ES1 Index", "VG1 Index", "TU1 Comdty", "DU1 Comdty", "TY1 Comdty", "RX1 Comdty", "GC1 Comdty", "CC1 Comdty"]

                Hurst_PicklefileName = self.AlternativeStorageLocation + "RollHurst_RVsHedgeRatiosLumen"

                RollHurstPickle = open(Hurst_PicklefileName, 'rb')
                RollHurstPickleData = pickle.load(RollHurstPickle)
                RollHurstPickle.close()
                print(RollHurstPickleData)
                RollHurstPickleData.plot()
                #(RollHurstPickleData[TargetAssets]-0.5).plot()
                plt.show()
        ########################## MANIFOLD LEARNING FEATURES ####################################
        if RunSettings["ManifoldLearnersMode"] in ["Setup", "Update", "Read", "ManifoldPackUnpack", "AlphaExtraction"]:
            self.dIndicatorsDF = pe.d(pd.read_sql('SELECT * FROM IndicatorsDeck', self.conn).set_index("date", drop=True)).fillna(0)
            self.ActiveStrategiesAssetsClusters.append(pd.Series(list(self.rets.columns), name="ALL").drop_duplicates())
            self.ActiveStrategiesAssetsClusters.append(pd.Series(list(self.rets.columns)+list(self.dIndicatorsDF.columns), name="ALLm").drop_duplicates())
            if RunSettings["ManifoldLearnersMode"] not in ["AlphaExtraction"]:
                RunMultiProcessing = True
                processList = []
                for ManifoldLearner in ["PCA", "Beta"]:#"PCA", "LLE", "DMAPS", "Beta", "BetaRegressV", "BetaProject"
                    for RollModeIn in [['ExpWindow', 25]]: #['RollWindow', 250], ['ExpWindow', 25]
                        for ProjectionModeIn in ["NoTranspose"]: #NoTranspose,Transpose
                            for StrategiesAssetsCluster in self.ActiveStrategiesAssetsClusters:
                                StrategiesAssetsClusterName = StrategiesAssetsCluster.name
                                if StrategiesAssetsClusterName in ["Endurance"]: #!= "Expedition", "Endurance", StrategiesAssetsClusterName in ["ALL", "ALLm"]
                                    if (ManifoldLearner == "PCA") & (ProjectionModeIn == "NoTranspose"):
                                        ProjectionStyle = "Spatial"
                                    elif (ManifoldLearner == "PCA") & (ProjectionModeIn == "Transpose"):
                                        ProjectionStyle = "Temporal"
                                    elif (ManifoldLearner == "Beta") & (ProjectionModeIn == "NoTranspose"):
                                        ProjectionStyle = "Spatial"
                                    elif (ManifoldLearner == "Beta") & (ProjectionModeIn == "Transpose"):
                                        ProjectionStyle = "Temporal"
                                        #############################
                                    elif (ManifoldLearner == "LLE") & (ProjectionModeIn == "NoTranspose"):
                                        ProjectionStyle = "Temporal"
                                    elif (ManifoldLearner == "LLE") & (ProjectionModeIn == "Transpose"):
                                        ProjectionStyle = "Spatial"
                                        #############################
                                    elif (ManifoldLearner == "DMAPS") & (ProjectionModeIn == "NoTranspose"):
                                        ProjectionStyle = "Temporal"
                                    elif (ManifoldLearner == "DMAPS") & (ProjectionModeIn == "Transpose"):
                                        ProjectionStyle = "Spatial"
                                    ##########################################################################################
                                    ManifoldLearnerID = ManifoldLearner + "_" + ProjectionStyle + "_" + str(RollModeIn[0]) + "_" + str(RollModeIn[1]) + "_" + StrategiesAssetsClusterName
                                    ##########################################################################################
                                    print(ManifoldLearnerID, ", ", ProjectionStyle)
                                    if RunSettings["ManifoldLearnersMode"] == "ManifoldPackUnpack":
                                        dbfile = open(self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID,'rb')
                                        ManifoldLearnerPack = pickle.load(dbfile)
                                        dbfile.close()
                                        ##########################################################################################
                                        if ProjectionStyle == "Spatial":
                                            TemporalExtractionIn = "LastValue"
                                            argDict = {
                                                "ManifoldLearnerMode": "Unpack",
                                                "ManifoldLearner": ManifoldLearner,
                                                "ManifoldLearnerPack": ManifoldLearnerPack,
                                                "ProjectionStyle":ProjectionStyle,
                                                "TemporalExtraction": TemporalExtractionIn,
                                                "WriteFilePath": self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID + "_" + TemporalExtractionIn + "_" + ProjectionStyle + "_Unpacked",
                                            }
                                            processList.append(argDict)
                                        else:
                                            for TemporalExtractionIn in ["LastValue", "PearsonCorrelationVal"]:#"LastValue", "PearsonCorrelationVal"
                                                argDict = {
                                                    "ManifoldLearnerMode": "Unpack",
                                                    "ManifoldLearner": ManifoldLearner,
                                                    "ManifoldLearnerPack": ManifoldLearnerPack,
                                                    "ProjectionStyle": ProjectionStyle,
                                                    "TemporalExtraction": TemporalExtractionIn,
                                                    "WriteFilePath": self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID + "_"+TemporalExtractionIn+"_" + ProjectionStyle+"_Unpacked",
                                                }
                                                processList.append(argDict)
                                        ############# DEBUGGING CALLER ############
                                        out = ManifoldLearnerRunner(argDict); RunMultiProcessing = False
                                    elif RunSettings["ManifoldLearnersMode"] == "Read":
                                        #dbfile = open(self.AlternativeStorageLocation+'ManifoldLearners\\' + ManifoldLearnerID, 'rb')
                                        #whatToOpen = "PCA_NoTranspose_RollWindow_250_Endurance_LastValue_Spatial_Unpacked"
                                        #whatToOpen = "PCA_Transpose_RollWindow_250_Endurance_PearsonCorrelationVal_Temporal_Unpacked"
                                        #whatToOpen = "LLE_Transpose_RollWindow_250_Endurance_LastValue_Spatial_Unpacked"
                                        #whatToOpen = "LLE_NoTranspose_RollWindow_250_Endurance_PearsonCorrelationVal_Temporal_Unpacked"
                                        whatToOpen = "DMAPS_Transpose_RollWindow_250_Endurance_LastValue_Spatial_Unpacked"
                                        #whatToOpen = "DMAPS_NoTranspose_RollWindow_250_Endurance_PearsonCorrelationVal_Temporal_Unpacked"
                                        dbfile = open(self.AlternativeStorageLocation + 'ManifoldLearners\\'+whatToOpen, 'rb')
                                        ManifoldLearnerPack = pickle.load(dbfile)
                                        dbfile.close()
                                        print(ManifoldLearnerPack)
                                        print(ManifoldLearnerPack[0])
                                        ManifoldLearnerPack[0].plot()
                                        plt.show()
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
                if RunMultiProcessing:
                    p = mp.Pool(mp.cpu_count()-2)
                    result = p.map(ManifoldLearnerRunner, tqdm(processList))
                    p.close()
                    p.join()
            elif RunSettings["ManifoldLearnersMode"] == "AlphaExtraction":
                ############# GREEDY ALPHA (CORRELATION MATRIX WITH INPUT SPACE) ############
                ManifoldFactorsList = []
                for name in glob.glob(self.AlternativeStorageLocation + 'ManifoldLearners\\*_Unpacked'):
                    CleanName = name.split("\\")[-1]
                    dbfile = open(name, 'rb')
                    ManifoldLearnerPack = pickle.load(dbfile)
                    dbfile.close()
                    InputData = ManifoldLearnerPack[0]
                    ProjNum = 0
                    for subDF in tqdm(ManifoldLearnerPack[1]):
                        subDF = subDF.fillna(0)
                        for sigMode in ["Raw", "Binary"]:
                            if sigMode == "Binary":
                                subDF = pe.sign(subDF)
                            for s in [1, 2, 3]:
                                if (len(subDF.columns) == len(InputData.columns))&("Temporal" not in subDF.columns[0]): # AS IN PCA
                                    ################################################################################################
                                    LinearProjectionDF = pe.fd((pe.S(subDF, nperiods=s) * InputData))
                                    LinearProjectionDF["AggProjection_"+str(ProjNum)+"_Shift_"+str(s)] = pe.rs(LinearProjectionDF)
                                    LinearProjection_sShift_PnL_Sharpe = np.sqrt(252) * pe.sharpe(LinearProjectionDF)
                                    LinearProjection_sShift_PnL_Sharpe.index = [CleanName + "_" + sigMode + "_" + x for x in LinearProjection_sShift_PnL_Sharpe.index]
                                    ManifoldFactorsList.append(LinearProjection_sShift_PnL_Sharpe)
                                    ############################### SEMAs ##########################
                                    for SemaS in [1, 2, 3]:
                                        for L in [3, 5, 10, 25, 50, 250]:
                                            semaLinearProjectionDF = pe.fd(pe.sign(pe.S(pe.ema(LinearProjectionDF, nperiods=L),nperiods=SemaS)) * LinearProjectionDF)
                                            semaLinearProjection_sShift_PnL_Sharpe = np.sqrt(252) * pe.sharpe(semaLinearProjectionDF)
                                            semaLinearProjection_sShift_PnL_Sharpe.index = [CleanName + "_" + sigMode + "_SemaShift_" + str(SemaS) + "_EMAlag_" + str(L) + "_" + x for x in semaLinearProjection_sShift_PnL_Sharpe.index]
                                            ManifoldFactorsList.append(semaLinearProjection_sShift_PnL_Sharpe)
                                else:
                                    for c in subDF.columns:
                                        LinearProjectionDF = pe.fd(InputData.mul(pe.S(subDF[c], nperiods=s), axis=0))
                                        LinearProjection_sShift_PnL_Sharpe = np.sqrt(252) * pe.sharpe(LinearProjectionDF)
                                        LinearProjection_sShift_PnL_Sharpe.index = [CleanName + "_" + sigMode + "_" + x for x in LinearProjection_sShift_PnL_Sharpe.index]
                                        ManifoldFactorsList.append(LinearProjection_sShift_PnL_Sharpe)
                                        ############################### SEMAs ##########################
                                        for SemaS in [1, 2, 3]:
                                            for L in [3, 5, 10, 25, 50, 250]:
                                                semaLinearProjectionDF = pe.fd(pe.sign(pe.S(pe.ema(LinearProjectionDF, nperiods=L),nperiods=SemaS)) * LinearProjectionDF)
                                                semaLinearProjection_sShift_PnL_Sharpe = np.sqrt(252) * pe.sharpe(semaLinearProjectionDF)
                                                semaLinearProjection_sShift_PnL_Sharpe.index = [CleanName + "_" + sigMode + "_SemaShift_" + str(SemaS) + "_emaL_" + str(L) + "_" + x for x in semaLinearProjection_sShift_PnL_Sharpe.index]
                                                ManifoldFactorsList.append(semaLinearProjection_sShift_PnL_Sharpe)

                        ProjNum += 1

                ManifoldFactorsDF = pd.concat(ManifoldFactorsList).sort_values(ascending=False)
                ManifoldFactorsDF.to_excel(self.AlternativeStorageLocation + 'ManifoldLearners\\AlphaExtraction.xlsx')
                print(ManifoldFactorsDF)
        ########################## MACRO CONNECTORS OBSERVATORY ####################################
        if RunSettings["MacroConnectorsObservatory"][0]:
            processList = []
            MacroTargetsList = list(self.ActiveIndicators)
            #MacroTargetsList = ["MOVE Index", "VIX Index", "BFCIUS Index", "BFCIEU Index", "USSWIT2 Curncy", "EUSWI2 Curncy",]
                    #"JPMVXYG7 Index", "GVZ Index", "OVX Index","EURCHFIS Curncy", "CHFUSDIS Curncy",
                    #"JPYUSDIS Curncy","BRLUSDIS Curncy", "USYC2Y10 Index",
                    #######################################################
                    #"JPMVXYEM Index", "BZFCIBBC Index", "GSZAFCI Index",
                    #"GSJPFCI Index","GSNZFCI Index","GSAUFCI Index","GSCAFCI Index","GSMXFCI Index",
                    #"GSRUFCI Index","GSUSFCI Index","GSBRFCI Index","GSGBFCI Index","GSMXRFCI Index",
                    #"GSEAFCI Index","BCOM Index","GSBRRFCI Index",
                    ######
                    #"GSCNFCI Index", "GSRURFCI Index"]

            CheckedSpace = "Rets"
            #CheckedSpace = "RetsSpecific"
            #CheckedSpace = "CTA"
            #CheckedSpace = "Manifold"
            if CheckedSpace == "CTA":
                TradedUniverse = pd.read_sql('SELECT * FROM subPortfoliosRetsDF', sqlite3.connect("StrategiesAggregator.db"))
                TradedUniverse = TradedUniverse.set_index(TradedUniverse.columns[0], drop=True).fillna(0)
                TradedUniverse.index.names = ['date']
            elif CheckedSpace == "Manifold":
                manifoldLearner = "PCA"
                #TargetSubSpace = "Endurance"
                #TargetSubSpace = "Coast"
                # TargetSubSpace = "Brotherhood"
                #TargetSubSpace = "ShoreDM"
                #TargetSubSpace = "ShoreEM"
                TargetSubSpace = "Valley"
                #TargetSubSpace = "Dragons"
                # TargetSubSpace = "Lumen"

                ManifoldLearnerID = manifoldLearner + "_Spatial_ExpWindow_25_" + TargetSubSpace + "_LastValue_Spatial_Unpacked"
                dbfile = open(self.AlternativeStorageLocation + 'ManifoldLearners\\' + ManifoldLearnerID, 'rb')
                ManifoldUnpacked = pickle.load(dbfile)
                dbfile.close()

                NativeSpace = ManifoldUnpacked[0]

                MPsList = ManSee.getManifoldProjections(ManifoldUnpacked, ProjectionShift=0)
                TargetProjDF = MPsList[0]
                for i in range(1,round(len(MPsList)/2)):
                    TargetProjDF += MPsList[i]

                fig, ax = plt.subplots(nrows=2, ncols=1)
                pe.cs(TargetProjDF).plot(ax=ax[0])
                pe.cs(pe.rs(TargetProjDF)).plot(ax=ax[1])
                plt.show()
            elif CheckedSpace == "Rets":
                TradedUniverse = self.rets[self.ActiveAssetsDF["Point_1"].tolist()]
            elif CheckedSpace == "RetsSpecific":
                TradedUniverse = self.rets[["ES1 Index", "VG1 Index",
                                            "TY1 Comdty", "RX1 Comdty",
                                            "TU1 Comdty", "DU1 Comdty",
                                            #"DX1 Curncy", "BR1 Curncy",
                                            #"FF1 Comdty", "ZB1 Comdty",
                                            #"CL1 Comdty", "GC1 Comdty",
                                            #"HG1 Comdty"
                                            ]]
                print("TradedUniverse.shape = ", TradedUniverse.shape)

            WhereToLookForConnection = "Raw"
            #WhereToLookForConnection = "EMA_25"
            #WhereToLookForConnection = "EMA_250"
            PackSpace = ["dIndicators","Indicators"]
            #PackSpace = ["EcoLast","EcoSurvey","EcoSpread"] #"Indicators","dIndicators","EcoLast","EcoSurvey","EcoSpread"
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
                MacroConnectorsObservatoryDF = MacroConnectorsObservatoryDF[[x for x in MacroConnectorsObservatoryDF.columns if (x in TradedUniverse.columns)|(x in MacroTargetsList)]]
                if "EMA" in WhereToLookForConnection:
                    MacroConnectorsObservatoryDF = pe.ema(MacroConnectorsObservatoryDF, nperiods=int(WhereToLookForConnection.split("_")[1]))
                MacroConnectorsObservatoryDF.corr().to_excel(self.AlternativeStorageLocation + 'MacroConnectorsObservatory\\' + pack+"_"+CheckedSpace + "_CorrMatrix_ExPost.xlsx")
                print("Calculating Static SubSpaceIgnoreConnections ...")
                for subSets in [["Corr", "IgnorePositives"]]:#["MI", "IgnorePositives"]
                    connectionData = pe.SubSpaceIgnoreConnections(MacroConnectorsObservatoryDF,subSets[0], subSets[1])
                    if CheckedSpace == "Rets":
                        connectionData = connectionData[[x for x in MacroConnectorsObservatoryDF.columns if x in self.FuturesTable["Point_1"].tolist()]]
                    connectionData.to_excel(self.AlternativeStorageLocation + 'MacroConnectorsObservatory\\' + WhereToLookForConnection + "_" + pack + "_" + CheckedSpace + "_"+subSets[0]+"_"+subSets[1]+".xlsx")
                print("Calculating Rolling SubSpaceIgnoreConnections ...")
                if RunSettings["MacroConnectorsObservatory"][2] == "RollMetric":
                    for metricIn in ["SubSpaceIgnoreConnections_IgnorePositives_Corr"]:#, "AdjMI", "NormMI"
                        for RollMode in [["ExpWindow", 25]]:#["ExpWindow", 25], ["RollWindow", 250]
                            MacroConnectorsObservatoryID = WhereToLookForConnection + "_" + pack +"_"+CheckedSpace + "_" + RunSettings["MacroConnectorsObservatory"][2]+"_"+metricIn+"_"+RollMode[0]+"_"+str(RollMode[1])
                            print(MacroConnectorsObservatoryID)
                            processDict = {
                                "data":MacroConnectorsObservatoryDF,
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
        ########################## HEALTH CHECKS #################################################
        if RunSettings["HealthChecks"][0]:
            TailPointsStored = RunSettings["HealthChecks"][1]
            VisualCheckSettings = RunSettings["HealthChecks"][2]
            VisualCheckPath = self.AlternativeStorageLocation + "/HealthChecksPacks/DT_VisualCheck/"
            ############################################ EXCEL PACK ##############################
            print("Cleaning up " + VisualCheckPath + " ... ")
            files_list = os.listdir(VisualCheckPath)
            for files in files_list:
                if files!="HTMLs":
                    os.remove(VisualCheckPath+files)
            print("Done ...")
            ############################################ EXCEL PACK ##############################
            CheckFilesList = pd.read_excel(self.DataDeckExcel, sheet_name="SystemHealthCheck")
            for idx, row in CheckFilesList.iterrows():
                print("Health Checking for ", row['Item'], " ...")
                try:
                    out = pe.readPickleDF(self.AlternativeStorageLocation+row['Item'])
                    if type(out) is list:
                        for i in range(len(out)):
                            out[i].tail(TailPointsStored).to_sql(row['ID']+"_"+str(i), self.HealthChecksConn, if_exists='replace')
                    else:
                        out.tail(TailPointsStored).to_sql(row['ID'], self.HealthChecksConn, if_exists='replace')
                except Exception as e:
                    print(e)
                print("Done ... ")
            ############################################ DT PACK ##############################
            DT_Rets_HealthChecks_List = pd.read_excel(self.AlternativeStorageLocation+"HealthChecksPacks/plainedFactorDataDF_RetsLive.xlsx")
            DT_Rets_HealthChecks_ReportList = list(set(list(DT_Rets_HealthChecks_List.columns)))
            ###################################################################################
            IndicatorsDF = IndicatorsDF.loc[:, ~IndicatorsDF.columns.duplicated()].copy()
            IndicatorsDF[DT_Rets_HealthChecks_ReportList].tail(TailPointsStored).to_sql("Latest_Indicators_Values_TimeSeries", self.HealthChecksConn, if_exists='replace')
            IndicatorsDF[DT_Rets_HealthChecks_ReportList].iloc[-5:,:].T.to_sql("Latest_Indicators_Values", self.HealthChecksConn, if_exists='replace')
            ################################################################################################################################################
            DT_Roll_Mode = '_RollWindow_250_'
            ################################################################################################################################################
            print("Visual Checks : ", VisualCheckSettings['VisualCheck'])
            for c in DT_Rets_HealthChecks_ReportList:
                if True:#c == "FDTR Index":
                    try:
                        print("Health Checking DT_Rets_"+c)
                        DT_Data = pe.readPickleDF(self.AlternativeStorageLocation+"DecisionTrees/RetsLive/DecisionTrees_RV_Rets"+DT_Roll_Mode+c)#ExpWindow_25
                        DT_Data.columns = [x.split(",")[1] for x in DT_Data.columns]
                        ############################################################
                        DT_Data = DT_Data[DT_Rets_HealthChecks_List[c].dropna().tolist()]
                        DT_Data.tail(TailPointsStored).to_sql("DT_Rets_"+c, self.HealthChecksConn, if_exists='replace')
                        #########################################################################################################################
                        DT_Data_Signal = pe.sign(DT_Data.sub(IndicatorsDF[c],axis=0))
                        for a in DT_Data_Signal.columns:
                            directionDF = self.ActiveStrategiesFactorsControl.loc[a, ["SingleDecisionTreesControllers_Positive", "SingleDecisionTreesControllers_Negative"]]
                            directionDF = directionDF.to_frame()
                            directionDF[a] = directionDF[a].str.split("_").str[0]
                            directionStr = directionDF[directionDF==c].dropna().index[0].split("_")[-1]
                            if directionStr == "Negative":
                                DT_Data_Signal[a] = DT_Data_Signal[a] * (-1)
                        DT_Data_Signal.tail(TailPointsStored).to_sql("DT_Signal_Rets_"+c, self.HealthChecksConn, if_exists='replace')
                        ############################################################
                        if VisualCheckSettings['VisualCheck'] == 'YES':
                            for DT_Data_Col in tqdm(DT_Data.columns):
                                if True:#DT_Data_Col == "SF1 Curncy":#True
                                    try:
                                        #########################################################################################
                                        visualCheckDF = pd.concat([DT_Data[DT_Data_Col], IndicatorsDF[c],
                                                                   self.rets[DT_Data_Col], DT_Data_Signal[DT_Data_Col]], axis=1).sort_index()
                                        visualCheckDF.columns = ["DT_Thr", c, DT_Data_Col, "DT_Sig"]
                                        visualCheckDF.index = [x.split(" ")[0] for x in visualCheckDF.index]
                                        #########################################################################################
                                        for plotPack in [[visualCheckDF, "SinceInception"], [visualCheckDF.tail(500), "1Y"]]:
                                            fig, ax = plt.subplots(nrows=3, ncols=1)
                                            plotPack[0][["DT_Thr", c]].plot(ax=ax[0], legend=None)
                                            pe.cs(plotPack[0][DT_Data_Col]).plot(ax=ax[1])
                                            plotPack[0]["DT_Sig"].plot(ax=ax[2])
                                            plt.xticks(rotation=25)
                                            plt.savefig(VisualCheckPath+c+"_"+plotPack[1]+"__"+DT_Data_Col)
                                        #########################################################################################
                                    except Exception as e:
                                        print(e)
                            #########################################################################################
                            for DT_Data_Col in tqdm(DT_Data.columns):
                                f = open(VisualCheckPath + '/HTMLs/' + DT_Data_Col + '.html', 'w')
                                DT_html_template = '<html> <head> <title>Title</title></head><body><h2>DT Visualiser for '+DT_Data_Col+'</h2>'
                                DT_html_template += '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PyBloomyFlask/template/PyEurobankFlask_Home.html">Eurobank Flask Home</a><br>'
                                #####################
                                for name in glob.glob(VisualCheckPath+'/*'+DT_Data_Col+'*.png'):
                                    DT_html_template += '<img src="'+name+'" alt="DTVisualiser" width="1200" height="800"><hr>'
                                #####################
                                DT_html_template += '</body></html>'
                                f.write(DT_html_template)
                            f.close()
                        #########################################################################################################################
                        DEMAs = pe.readPickleDF(self.AlternativeStorageLocation + "DEMAs/_DefaultSingleLookBack_LookBacksPack_DEMAs").fillna(0)
                        MomentumSig = pe.sign(DEMAs)
                        #########################################################################################################################
                        DT_Data_With_Filter = DT_Data.copy()
                        for col in DT_Data_With_Filter.columns:
                            try:
                                DT_Data_With_Filter[col] = DT_Data_With_Filter[col].round(4).astype(str) + " :: " + IndicatorsDF[c].round(4).astype(str) + " :: " + DT_Data_Signal[col].round(4).astype(str)+ " :: " + MomentumSig[col].round(1).astype(str) + " ::: (total) = " + (DT_Data_Signal[col]*MomentumSig[col]).round(1).astype(str)
                            except Exception as e:
                                print(e)
                        DT_Data_With_Filter.columns = [x+"_"+"DtThr"+"_"+"CIndVal"+"_"+"DEMAsig" for x in DT_Data_With_Filter.columns]
                        DT_Data_With_Filter.tail(TailPointsStored).to_sql("DT_Rets_With_Filter_" + c, self.HealthChecksConn, if_exists='replace')
                    except Exception as e:
                        print(e)

####################################################################################################################

####################################################################################################################
############################################################################################################################################################
#### CALLERS ####
############################################################################################################################################################
"EXTRA FUNCTIONS"
def ExtraFunctions(mode, TargetAssets):
    LocalConn = sqlite3.connect("DataDeck.db")
    #retsDF = pd.read_sql('SELECT * FROM rets', LocalConn).set_index('date', drop=True)#.tail(50)
    retsDF = pd.read_sql('SELECT * FROM ValleyRV_ValleyRV_RVsDF', sqlite3.connect("ValleyRV.db")).set_index('date', drop=True).tail(50)
    IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', LocalConn).set_index('date', drop=True).tail(50)

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
"DATABASE HANDLERS"
def Transfer_Data(task):
    if task == 0:
        #SourceDB_conn = sqlite3.connect("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/DataDeck_Temp.db")
        SourceDB_conn = sqlite3.connect("DataDeck.db")
        TargetDB_conn = sqlite3.connect("DataDeck.db")

        #dfSource = pd.read_sql('SELECT * FROM LookBacksTest', SourceDB_conn).set_index('date', drop=True)
        dfTarget = pd.read_sql('SELECT * FROM LookBacks', TargetDB_conn).set_index('index', drop=True)
        dfTarget.index.names = ['date']
        #dfTarget = dfTarget.set_index('date', drop=True)
        dfTarget.to_sql("LookBacks", SourceDB_conn, if_exists='replace')
        """
        for c in dfSource.columns:
            dfTarget[c] = dfSource[c]
        dfTarget.ffill().bfill().to_sql("LookBacks", SourceDB_conn, if_exists='replace')
        """
    elif task == 1:
        SourceDB_conn = sqlite3.connect("DataDeck.db")
        sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
        cursor = SourceDB_conn.cursor()
        cursor.execute(sql_query)
        out = cursor.fetchall()
        print(out)
    elif task == 2:
        SourceDB_conn = sqlite3.connect("DataDeck.db")
        for pack in ["",
                     "Lumen_RV",
                     "ShoreDM_RV",
                     "ShoreEM_RV",
                     "Valley_RV",
                     "Endurance_RV_CF1 Index_Driver",
                     "Endurance_RV_DM1 Index_Driver",
                     "Endurance_RV_ES1 Index_Driver",
                     "Endurance_RV_GX1 Index_Driver",
                     "Endurance_RV_MES1 Index_Driver",
                     "Endurance_RV_MFS1 Index_Driver",
                     "Endurance_RV_NK1 Index_Driver",
                     "Endurance_RV_NQ1 Index_Driver",
                     "Endurance_RV_TP1 Index_Driver",
                     "Endurance_RV_VG1 Index_Driver",
                     "Endurance_RV_XP1 Index_Driver",
                     ]:
            try:
                df0 = pd.read_sql('SELECT * FROM LookBacks'+pack, SourceDB_conn).set_index('date', drop=True)
                df1 = pd.read_sql('SELECT * FROM LookBacksDirections'+pack, SourceDB_conn).set_index('date', drop=True)
            except Exception as e:
                print(e)
                df0 = pd.read_sql("SELECT * FROM 'LookBacks" + pack + "'", SourceDB_conn).set_index('date', drop=True)
                df1 = pd.read_sql("SELECT * FROM 'LookBacksDirections" + pack + "'", SourceDB_conn).set_index('date', drop=True)
            out = [df0, df1]
            DTPickle = open("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/LookbacksPacks/" + pack + "_Pack", 'wb')
            pickle.dump(out, DTPickle)
            DTPickle.close()
    elif task == 3:
        SourceDB_conn = sqlite3.connect("DataDeck.db")
        df = pd.read_sql("SELECT * FROM LookBackRets", SourceDB_conn).set_index('date', drop=True)
        ManifoldLearnerMethodsSpace = ["PCA", "DMAPs", "LLE"]
        for filename in glob.glob("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/LookbacksPacks/*_LookBacksPack"):
            Pickle = open(filename, 'rb')
            outPickle = pickle.load(Pickle)
            Pickle.close()
            if len(outPickle) > 3:
                out = outPickle[1:]
            else:
                out = outPickle
            print(len(outPickle), len(out))
            ################################################################################################################################
            #DT_Space_ID = filename.split("_RollMode_")[0].split("\\")[-1]
            #if DT_Space_ID.split("_")[0] in ManifoldLearnerMethodsSpace:
            #    ManifoldUnpacked = pe.readPickleDF("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/ManifoldLearners/" + DT_Space_ID)
            #    NativeSpace = ManifoldUnpacked[0]
            #    MPsList = ManSee.getManifoldProjections(ManifoldUnpacked, ProjectionShift=0)
            #    DT_Proj_Rets_List = []
            #    for Projection in MPsList:
            #        DT_Proj_Rets_List.append(pe.rs(Projection))
            #    LookbackTargets = pd.concat(DT_Proj_Rets_List, axis=1).sort_index()
            #    LookbackTargets.columns = ["Proj_" + str(x) for x in range(len(LookbackTargets.columns))]
            #else:
            #    LookbackTargets = df
            ################################################################################################################################
            #out = [LookbackTargets]
            #for item in outPickle:
            #    out.append(item)
            WritePickle = open(filename, 'wb')
            pickle.dump(out, WritePickle)
            WritePickle.close()
def Append_Data():
    print("Appending Data ...")
    #SourceDB_conn = sqlite3.connect("C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/DataDeck_Temp.db")
    SourceDB_conn = sqlite3.connect("DataDeck.db")
    TargetDB_conn = sqlite3.connect("DataDeck.db")

    for targetTable in ["LookBacks", "DEMAs"]:
        print(targetTable)
        dfTarget = pd.read_sql('SELECT * FROM '+targetTable, TargetDB_conn)
        #dfTarget = dfTarget.set_index(dfTarget.columns[0], drop=True)
        #dfTarget.index.names = ['date']
        #dfTarget.ffill().bfill().sort_index().to_sql(targetTable, TargetDB_conn, if_exists='replace')
        #dfTarget.ffill().bfill().sort_index().to_sql(targetTable+"_Latest", TargetDB_conn, if_exists='replace')
        #for table in [targetTable+"CarryDragons", targetTable+"CarryLumen", targetTable+"CarryValley"]:
        #    try:
        #        subDF = pd.read_sql('SELECT * FROM '+table, SourceDB_conn).set_index('date', drop=True)
        #        dfTarget = pd.concat([dfTarget, subDF], axis=1).sort_index()
        #    except Exception as e:
        #        print(e)
        #dfTarget.ffill().bfill().to_sql(targetTable, TargetDB_conn, if_exists='replace')
    print("Appending Done ...")
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
def DummyCaller():

    targetPath = "C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/LookbacksPacks/"
    out = pe.readPickleDF(targetPath + "_LookBacksList_LookBacksPack")
    print(out)
    """
    pack1 = pe.readPickleDF(targetPath+"_LookBacksList_LookBacksPack_Point_1")
    pack2 = pe.readPickleDF(targetPath + "_LookBacksList_LookBacksPack_Point_2")
    pack3 = pe.readPickleDF(targetPath + "_LookBacksList_LookBacksPack_Point_3")
    allPack = [
        pd.concat([pack1[0],pack2[0],pack3[0]],axis=1).sort_index(),
               pd.concat([pack1[1], pack2[1], pack3[1]], axis=1).sort_index(),
               pd.concat([pack1[2], pack2[2], pack3[2]], axis=1).sort_index()
               ]
    out = pe.savePickleDF(allPack,targetPath+"_LookBacksList_LookBacksPack")
    """

####################################################################################################################

if __name__ == '__main__':
    #"""
    "LIVE/PROD"
    RunSettings = {"RunDataMode": True,
                   "DataPatchers": True,
                   "AlternativeDataFields": [True,{"DataFetch":True,"AlternativeIndicators":True}],
                   "LookBacksDEMAsCalcMode": [True, "Update", True, ["Rets", "", "DefaultSingleLookBack", "EMA"]],
                   "Perform_TDA":[False, {"Wasserstein_Distances":True}],
                   "HedgeRatiosRVs": [False, {"Mode":"Setup"}],#Setup,Update
                   "DecisionTrees": [True, "Update", ["RetsLive", "RetsID"], 15],#Setup,Update,Break,DT_Space = "ManifoldLearner"; DT_Space_ID = "PCA_Spatial_ExpWindow_25_Endurance_LastValue_Spatial_Unpacked"
                   "RunEcoDataMode": True,
                   "RunHurst": "Break", #Setup, Update, Read, Break
                   "ManifoldLearnersMode": "Break", #Setup, Update, Read, ManifoldPackUnpack, Break, AlphaExtraction
                   "MacroConnectorsObservatory": [False, "Break", "RollMetric"],#"DataSetup", Break, "RollMetric", "ReadMetric"
                   "RunBenchmarkPortfolios": True,
                   "HealthChecks":[True, 50, {"VisualCheck":"YES"}]}
    #"""
    ###########################################################################
    """
    "DEV"
    RunSettings = {"RunDataMode": False,
                   "DataPatchers": False,
                   "AlternativeDataFields": [False,{"DataFetch":True,"AlternativeIndicators":True}],
                   "LookBacksDEMAsCalcMode": [False, "Update", True, ["Rets", "", "DefaultSingleLookBack", "EMA"]],#LookbacksDEMAsAll, Lookbacks(Setup,Update,Break), DEMAs, DatasetSel
                   "Perform_TDA":[False,{"Wasserstein_Distances":True}],
                   "HedgeRatiosRVs": [False,{"Mode":"Setup"}],#Setup,Update
                   "DecisionTrees": [False, "Update", ["RetsLive", "RetsID"], 125],#Setup,Update,Break,DT_Space = "ManifoldLearner"; DT_Space_ID = "PCA_Spatial_ExpWindow_25_Endurance_LastValue_Spatial_Unpacked"
                   "RunEcoDataMode": False,
                   "RunHurst": "Break", #Setup, Update, Read, Break
                   "ManifoldLearnersMode": "Break", #Setup, Update, Read, ManifoldPackUnpack, Break, AlphaExtraction
                   "MacroConnectorsObservatory": [False, "Break", "RollMetric"],#"DataSetup", Break, "RollMetric", "ReadMetric"
                   "RunBenchmarkPortfolios": False,
                   "HealthChecks":[True, 50, {"VisualCheck":"YES"}]}
    #"""
    ###########################################################################
    obj = DataDeck("DataDeck.db", RefDataFetch=RunSettings["RunDataMode"])
    #obj = DataDeck("DataDeckUnAdjusted.db", RefDataFetch=RunSettings["RunDataMode"])
    obj.Run(RunSettings)

    #ExtraFunctions("LookBacks", ["ES1 Index", "NQ1 Index"])

    #Transfer_Data(3)
    #Append_Data()
    #Insert_Tables_in_DB()
    #SyncIndicatorsOnAllDBs("DataDeck.db")
    #DummyCaller()

"""
##### NOTES #####
# obj = DataDeck("DataDeck_Staged.db", RefDataFetch=RunSettings["RunDataMode"])
# obj = DataDeck("DataDeck_Research.db", RefDataFetch=RunSettings["RunDataMode"])
# obj = DataDeck("DataDeck_Mega.db", RefDataFetch=RunSettings["RunDataMode"])
# obj = DataDeck("DataDeck_Temp.db", RefDataFetch=RunSettings["RunDataMode"])
# obj = DataDeck("DataDeck_1950.db", RefDataFetch=RunSettings["RunDataMode"])
"""