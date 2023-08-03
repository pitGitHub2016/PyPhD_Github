import time
try:
    from os.path import dirname, basename, isfile, join
    import glob, os, sys
    import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')

    from pyerb import pyerb as pe
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import DataDeck
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Endurance
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Coast
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Brotherhood
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Shore
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Valley
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Dragons
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import LiveEMSXHistory
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import StrategiesAggregator

    import warnings
    warnings.filterwarnings("ignore")
except Exception as e:
    print(e)
    time.sleep(10)

class TimeStories:

    def __init__(self, update, StrategiesStatus):
        self.StrategiesStatus = StrategiesStatus
        if self.StrategiesStatus == "ActiveStrategies":
            self.DataDB_Label = ""
        elif self.StrategiesStatus == "StagedStrategies":
            self.DataDB_Label = "_Staged"
        self.conn = sqlite3.connect("TimeStories"+self.DataDB_Label+".db")
        self.PyLiveTradingSystemsFolder = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems/"
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"
        self.KplusExposuresPath = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/Bloomberg_EMSX/"
        self.EMSX_Excel_Path = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/Bloomberg_EMSX/EMSX_Excel_Order_Routing.xlsx"
        self.PyTradingScriptsDF = pd.DataFrame(glob.glob(self.PyLiveTradingSystemsFolder+"*.py"), columns=["ScriptFile"])
        self.PyTradingScriptsDF["Name"] = self.PyTradingScriptsDF["ScriptFile"].str.split("PyLiveTradingSystems").str[1]
        self.PyTradingScriptsDF[["Name", "ScriptFile"]].to_sql("PyTradingScriptsDF", self.conn, if_exists='replace')

        if update == "Yes":
            RunSettings = {"RunDataMode": True,
                           "DataPatchers": True,
                           "LookBacksDEMAsCalcMode": [False, True, True],  # LookbacksDEMAsAll, Lookbacks, DEMAs
                           "RunEcoDataMode": True,
                           "ManifoldLearnersMode": "Break",  # Setup, Update, Read, Break
                           "MacroConnectorsObservatory": [False, "Break", "ReadMetric"],
                           # "DataSetup", "RollMetric", "ReadMetric"
                           "RunBenchmarkPortfolios": True}

            DataDeck.DataDeck("DataDeck"+self.DataDB_Label+".db").Run(RunSettings)

    def LiveStrategiesRun(self):

        # Start With Endurance which updates Latest Data (DataDeck) as well!
        Endurance.Endurance().Run()

        # Continue with the rest of the strategies - "No" means no need to update live data tables again
        Coast.Coast().Run()
        Brotherhood.Brotherhood().Run()
        Shore.Shore().Run()
        Valley.Valley().Run()
        Dragons.Dragons().Run()

    def ControlPanel(self, **kwargs):

        # Aggregate Live Strategies
        StrategiesAggregator.StrategiesAggregator("DataDeck"+self.DataDB_Label+".db", "Endurance", ["Coast", "Brotherhood", "Valley", "Shore", "Dragons"], [1,1,1,1,1]).LinearAggregation()

        def EMSX_Kondor_Dict(k):

            if k == "CME-ED":
                return "ED1 Comdty"
            elif k == 'F_3MEURIBOR':
                return 'ER1 Comdty'
            ##############################################
            elif k == "NE":
                return "NV1 Curncy"
            elif k == "F_MXN":
                return "PE1 Curncy"
            elif k == "F_EUR":
                return "EC1 Curncy"
            elif k == "F_JPY":
                return "JY1 Curncy"
            elif k == "FUT_BRL":
                return "BR1 Curncy"
            elif k == "F_ZAR":
                return "RA1 Curncy"
            elif k == "F_CAD":
                return "CD1 Curncy"
            elif k == "F_AUD":
                return "AD1 Curncy"
            elif k == "F_GBP":
                return "BP1 Curncy"
            elif k == "RUB_USD_FUT":
                return "RU1 Curncy"
            elif k == "F_CHF":
                return 'SF1 Curncy'
            elif k == "DOLLAR_FUT":
                return 'DX1 Curncy'
            ##############################################
            elif k == "E-MINI_FUTUR":
                return "ES1 Index"
            elif k == "YM":
                return "DM1 Index"
            elif k == "CAC40_FUTURE":
                return 'CF1 Index'
            elif k == "F_EURSTOXX50":
                return 'VG1 Index'
            elif k == "F_DAX":
                return 'GX1 Index'
            elif k == "ME":
                return 'FA1 Index'
            elif k == "OMXH25_FUT":
                return 'OT1 Index'
            elif k == "NAS_100_MINI":
                return 'NQ1 Index'
            elif k == "F_SMI":
                return "SM1 Index"
            ##############################################
            elif k == "F_2Y_T_NOTE":
                return "TU1 Comdty"
            elif k == "F_TY_T_NOTE":
                return "TY1 Comdty"
            elif k == "F_5Y_T_NOTE":
                return "FV1 Comdty"
            elif k == "EUREXSCHATZ":
                return 'DU1 Comdty'
            elif k == "EUREXBOBL":
                return 'OE1 Comdty'
            elif k == "EUREXBUND":
                return 'RX1 Comdty'
            ##############################################
            elif k == "----------------------":
                return 'FF1 Comdty'
            elif k == "----------------------":
                return 'FVS1 Index'
            elif k == "VIX_FUTURE":
                return 'UX1 Index'

        if "Stealth" in kwargs:
            Stealth = kwargs['Stealth']
        else:
            Stealth = "NO"

        if Stealth == "NO":
            ############# Live EMSX Position Repoting ##########
            obj = LiveEMSXHistory.LiveEMSXHistory("LiveEMSXHistory.db")
            obj.main()
            obj.ProcessFills()

        if "LiveFuturesMaturity" in kwargs:
            LiveFuturesMaturity = kwargs['LiveFuturesMaturity']
        else:
            LiveFuturesMaturity = ""

        "GET KONDOR EXPOSURES AND AGGREGATE!"
        self.KplusDF = pd.read_excel(self.KplusExposuresPath+"KondorFuturesDealsExport.xlsx")
        self.KplusDF["Trade Date"] = pd.to_datetime(self.KplusDF["Trade Date"])
        self.KplusDF["BinaryDirection"] = None
        self.KplusDF.loc[self.KplusDF["Type"] == "Sell", "BinaryDirection"] = -1
        self.KplusDF.loc[self.KplusDF["Type"] == "Buy", "BinaryDirection"] = 1
        self.KplusDF["Exposure"] = self.KplusDF["BinaryDirection"] * self.KplusDF["Quantity"]
        ExposuresList = []
        for asset in set(self.KplusDF["Security"].tolist()):
            subDF = self.KplusDF[self.KplusDF["Security"] == asset][["Trade Date", "Exposure"]].set_index("Trade Date",
                                                                                                drop=True)
            subDF = subDF.resample("D").sum()
            subDF.columns = [asset]
            subDF = subDF.sort_index()
            ExposuresList.append(subDF)
        self.dfAggr = pd.concat(ExposuresList, axis=1).fillna(0).cumsum()
        self.dfAggr.to_excel(self.KplusExposuresPath+"KondorExposureTracker.xlsx")
        self.KondorExposureTrackerDF = pd.read_excel(self.KplusExposuresPath+"KondorExposureTracker.xlsx").set_index("Trade Date", drop=True)
        for c in self.KondorExposureTrackerDF.columns:
            self.KondorExposureTrackerDF = self.KondorExposureTrackerDF.rename(columns={c: EMSX_Kondor_Dict(c)})
        self.KondorLatestExposures = self.KondorExposureTrackerDF.iloc[-1]

        "EMSX TRADES"
        tradesList = []
        ActiveStrategiesNames = pd.read_excel("AssetsDashboard.xlsx", sheet_name=self.StrategiesStatus, engine='openpyxl').dropna(axis=1, how='all').columns
        for strategy in ActiveStrategiesNames:
            strategyDf = pd.read_sql('SELECT * FROM '+strategy+'_LATEST_Trades', sqlite3.connect(strategy+".db"))
            try:
                strategyDf = strategyDf.set_index("index", drop=True)
            except Exception as e:
                #print(e)
                strategyDf = strategyDf.set_index("ticker", drop=True)
            strategyDf['Strategy'] = strategy
            tradesList.append(strategyDf)

        TargetTrades = pd.concat(tradesList)
        TargetTrades.to_sql("TargetTrades", self.conn, if_exists='replace')
        #print(TargetTrades)

        aggregatedFills = pd.read_sql('SELECT * FROM PPAPAIOANNO1_aggregatedFills', sqlite3.connect("LiveEMSXHistory.db"))
        aggregatedFills["ticker"] = aggregatedFills["index"].str.split(" ").str[0]
        aggregatedFills["index"] = aggregatedFills["ticker"].str[:2] + "1 " + aggregatedFills["index"].str.split(" ").str[1]
        aggregatedFills.loc[aggregatedFills["RollAction"] == "Expired", "exposureShares"] = 0
        aggregatedFills = aggregatedFills.groupby(["index"]).sum().reset_index().set_index("index", drop=True)
        #print(aggregatedFills["exposureShares"])

        controlled_TargetTrades = TargetTrades.reset_index().groupby(["index"]).sum().reset_index().set_index("index", drop=True).iloc[:,0]
        ControlPanel = pd.concat([controlled_TargetTrades,aggregatedFills["exposureShares"]], axis=1).fillna(0)
        ControlPanel["KondorExposures"] = 0
        for idx, row in ControlPanel.iterrows():
            try:
                ControlPanel.loc[idx, "KondorExposures"] = float(self.KondorLatestExposures[idx])
            except Exception as e:
                print(e)

        exposureSpaceIdentifier = ''
        ControlPanel["exposureSharesEMSX"] = None

        "IF YOU BASE THE POSITION ADJUSTMENT ON KONDOR INSTEAD OF BLOOMBERG EMSX ITSELF"
        exposureSpaceIdentifier = 'Kondor'
        ControlPanel["exposureSharesEMSX"] = ControlPanel["exposureShares"]
        ControlPanel["exposureShares"] = ControlPanel["KondorExposures"]

        ControlPanel["PositionAdjustment"] = ControlPanel.iloc[:, 0] - ControlPanel["exposureShares"]

        ### HTML REPORT ###
        ControlPanelToReport = ControlPanel.reset_index().rename(columns={'index': 'Asset', 'exposureShares': 'CurrentPosition'+exposureSpaceIdentifier})
        ControlPanelToReport = ControlPanelToReport[['Asset', ControlPanel.iloc[:,0].name, 'CurrentPosition'+exposureSpaceIdentifier, 'exposureSharesEMSX', 'KondorExposures', 'PositionAdjustment']].set_index('Asset',drop=True)
        ControlPanelToReport = ControlPanelToReport.sort_values(by="PositionAdjustment")
        ControlPanelToReport["Kondor_vs_EMSX"] = ControlPanel["exposureSharesEMSX"] - ControlPanel["KondorExposures"]
        ControlPanelToReport["Kondor_vs_EMSX_FLAG"] = "-"
        ControlPanelToReport.loc[(ControlPanelToReport["Kondor_vs_EMSX"] != 0)&(~ControlPanel["KondorExposures"].isna()),"Kondor_vs_EMSX_FLAG"] = "CHECK !!!"
        ControlPanelToReport.to_sql("ControlPanel", self.conn, if_exists='replace')
        ControlPanelToReport['CurrentPosition'+exposureSpaceIdentifier].to_excel(self.KplusExposuresPath+"EMSX_ControlPanel.xlsx")

        ControlPanelToReport = ControlPanelToReport.dropna().reset_index()
        pe.RefreshableFile([[ControlPanelToReport, 'QuantitativeStrategiesControlPanel']],
                        self.GreenBoxFolder + 'QuantitativeStrategies_ControlPanel.html',
                        5, cssID='QuantitativeStrategies',  addButtons="QuantStrategies")

        "EMSX Excel Trader"
        EMSX_Excel_Handler = ControlPanelToReport.copy().rename(columns={"Asset": "Ticker"})[["Ticker", "PositionAdjustment"]]
        EMSX_Excel_Handler = EMSX_Excel_Handler[EMSX_Excel_Handler["PositionAdjustment"] != 0]
        if EMSX_Excel_Handler.shape[0] > 0:
            EMSX_Excel_Handler.loc[EMSX_Excel_Handler["PositionAdjustment"] < 0, "Side"] = "SELL"
            EMSX_Excel_Handler.loc[EMSX_Excel_Handler["PositionAdjustment"] > 0, "Side"] = "BUY"

            EMSX_Excel_Handler["Action Trigger"] = "TRUE"
            EMSX_Excel_Handler["Action"] = "Route"
            EMSX_Excel_Handler["Request Status"] = None
            EMSX_Excel_Handler["Sequence Number"] = None
            EMSX_Excel_Handler["Route Number"] = None
            EMSX_Excel_Handler["Order Type"] = "MKT"
            EMSX_Excel_Handler["Amount"] = EMSX_Excel_Handler["PositionAdjustment"].abs()
            EMSX_Excel_Handler["TIF"] = "DAY"
            EMSX_Excel_Handler["Broker"] = "MLFE"
            EMSX_Excel_Handler["Handling instruction"] = "NONE"
            EMSX_Excel_Handler["Account"] = "2CC04485"
            EMSX_Excel_Handler["Strategy Name"] = "TargetTm"
            EMSX_Excel_Handler = EMSX_Excel_Handler[
                ["Action Trigger", "Action", "Request Status", "Sequence Number", "Route Number", "Ticker",
                 "Side", "Order Type", "Amount", "TIF", "Broker", "Handling instruction", "Account", "Strategy Name"]]
            LiveTickerHandler = EMSX_Excel_Handler["Ticker"].str.split(" ")
            EMSX_Excel_Handler["Ticker"] = LiveTickerHandler.str[0].str.replace("1", LiveFuturesMaturity) + " " + \
                                           LiveTickerHandler.str[1]
            NeedToWriteTrades = True
        else:
            print("NO PORTFOLIO ADJUSTMENTS NECESSARY! - skipping excel updating")
            NeedToWriteTrades = False

        from openpyxl import load_workbook
        print(self.EMSX_Excel_Path)
        writer = pd.ExcelWriter(self.EMSX_Excel_Path, engine='openpyxl')
        book = load_workbook(self.EMSX_Excel_Path)
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        if NeedToWriteTrades == True:
            EMSX_Excel_Handler.to_excel(writer, sheet_name="Staging Blotter", startrow=2, header=None, index=False)
            writer.save()
        else:
            import shutil
            shutil.copyfile(self.EMSX_Excel_Path.replace("EMSX_Excel_Order_Routing", "EMSX_Excel_Order_Routing-Fresh"),
                            self.EMSX_Excel_Path)

        print(EMSX_Excel_Handler)

mode = "Live"
#mode = "LiveNoDataUpdate"
#mode = "Stage"
#mode = "StageNoDataUpdate"
#mode = "CheckPositionsOnly_STEALTH_YES"
#mode = "CheckPositionsOnly_STEALTH_NO"

if mode == "Live":
    tradingObj = TimeStories("Yes", "ActiveStrategies")
elif (mode == "LiveNoDataUpdate")|("CheckPositionsOnly" in mode):
    tradingObj = TimeStories("No", "ActiveStrategies")
elif mode == "Stage":
    tradingObj = TimeStories("Yes", "StagedStrategies")
elif mode == "StageNoDataUpdate":
    tradingObj = TimeStories("No", "StagedStrategies")

if "CheckPositionsOnly" not in mode:
    tradingObj.LiveStrategiesRun()
    tradingObj.ControlPanel(LiveFuturesMaturity="M3", Stealth="NO") #Stealth="NO","YES"
else:
    print("Running CheckPositions with EMSX Stealth Mode = ", mode.split("_")[2])
    tradingObj.ControlPanel(LiveFuturesMaturity="M3", Stealth=mode.split("_")[2])  # Stealth="NO","YES"

if mode == "Live":
    #pass
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import RiskStatusIdentification
