import time
#try:
from os.path import dirname, basename, isfile, join
import glob, os, sys
import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
PyERBMainPath = 'F:/Dealing/Panagiotis Papaioannou/pyerb/'
sys.path.insert(0,PyERBMainPath)

from pyerb import pyerb as pe
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import DataDeck
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Endurance
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Coast
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Brotherhood
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import ShoreDM
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import ShoreEM
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Valley
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Dragons
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Lumen
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Fidei
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import LiveEMSXHistory
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import StrategiesAggregator

import warnings
warnings.filterwarnings("ignore")
#except Exception as e:
#    print(e)
#    time.sleep(10)

class TimeStories:

    def __init__(self, StrategiesStatus):
        self.StrategiesStatus = StrategiesStatus
        if self.StrategiesStatus == "ActiveStrategies":
            self.DataDB_Label = ""
        elif self.StrategiesStatus == "StagedStrategies":
            self.DataDB_Label = "_Staged"
        self.conn = sqlite3.connect("TimeStories"+self.DataDB_Label+".db")
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.PyLiveTradingSystemsFolder = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems/"
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"
        self.KplusExposuresPath = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/Bloomberg_EMSX/"
        self.EMSX_Excel_Path = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/Bloomberg_EMSX/EMSX_Excel_Order_Routing.xlsx"
        self.PyTradingScriptsDF = pd.DataFrame(glob.glob(self.PyLiveTradingSystemsFolder+"*.py"), columns=["ScriptFile"])
        self.PyTradingScriptsDF["Name"] = self.PyTradingScriptsDF["ScriptFile"].str.split("PyLiveTradingSystems").str[1]
        self.PyTradingScriptsDF[["Name", "ScriptFile"]].to_sql("PyTradingScriptsDF", self.conn, if_exists='replace')
        self.DataDeckExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
        self.Live_Strategies_Control_Panel = pd.read_excel(self.DataDeckExcel, sheet_name="Live Strategies Control Panel", engine='openpyxl').dropna(subset=["Strategy Name"]).dropna(axis=1)

    def LiveStrategiesRun(self):

        # Start With Endurance which updates Latest Data (DataDeck) as well!
        Endurance.Endurance().Run()

        # Continue with the rest of the strategies
        Coast.Coast().Run()
        Brotherhood.Brotherhood().Run()
        ShoreDM.ShoreDM().Run()
        ShoreEM.ShoreEM().Run()
        Valley.Valley().Run()
        Dragons.Dragons().Run()
        Lumen.Lumen().Run()
        Fidei.Fidei().Run()

    def ControlPanel(self, **kwargs):

        self.ActiveAssetsReferenceData = pd.read_sql('SELECT * FROM ActiveAssetsReferenceData', sqlite3.connect("DataDeck.db")).set_index('ticker', drop=True)

        # Aggregate Live Strategies
        StrategiesAggregatorObj = StrategiesAggregator.StrategiesAggregator("DataDeck"+self.DataDB_Label+".db", self.Live_Strategies_Control_Panel["Strategy Name"].iloc[0],
                                                  self.Live_Strategies_Control_Panel["Strategy Name"].iloc[1:].tolist(),
                                                  self.Live_Strategies_Control_Panel["CTA Allocation"].iloc[1:].tolist())
        StrategiesAggregatorObj.LinearAggregation()
        StrategiesAggregatorObj.PlotContributions()

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
        self.KplusDF = pd.read_excel(PyERBMainPath+"KPlus_FX_SPOT_POSITIONS.xlsx",sheet_name='DERIV_2_FUTURES')
        ##############################################################################################################
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
            self.KondorExposureTrackerDF = self.KondorExposureTrackerDF.rename(columns={c: pe.EMSX_Kondor_Dict(c)})
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


        ### HTML REPORT ###
        ControlPanelToReport = ControlPanel.reset_index().rename(columns={'index': 'Asset', 'exposureShares': 'CurrentPosition'+exposureSpaceIdentifier})
        ControlPanelToReport = ControlPanelToReport[['Asset', ControlPanel.iloc[:,0].name, 'CurrentPosition'+exposureSpaceIdentifier, 'KondorExposures',
                                                     'exposureSharesEMSX']].set_index('Asset',drop=True)

        "Adjusting Position for External EMSX exposures (e.g. Galileo)"
        ExternalStrategy_A_Name = 'Galileo'
        ExternalStrategy_A = pd.read_sql('SELECT * FROM NumContracts', sqlite3.connect(ExternalStrategy_A_Name+".db"))
        Exposures_ExternalStrategy_A = ExternalStrategy_A.iloc[-1,:]
        ControlPanelToReport["TargetPositions_"+ExternalStrategy_A_Name] = Exposures_ExternalStrategy_A
        ExternalStrategy_A_Fills = pd.read_sql('SELECT * FROM PPAPAIOANNO1_ExposuresDF_'+ExternalStrategy_A_Name, sqlite3.connect("LiveEMSXHistory.db"))
        ExternalStrategy_A_Fills = ExternalStrategy_A_Fills.set_index('BaseAsset',drop=True)
        ControlPanelToReport["Fills_"+ExternalStrategy_A_Name] = ExternalStrategy_A_Fills

        ControlPanelToReport["TotalSignal"] = 0#ControlPanelToReport.iloc[:, 0] + ControlPanelToReport["TargetPositions_"+ExternalStrategy_A_Name]
        ControlPanelToReport["TotalKondorExposures"] = 0#ControlPanelToReport["KondorExposures"] + ControlPanelToReport["Fills_"+ExternalStrategy_A_Name] * (-1)

        try:
            ControlPanelToReport = ControlPanelToReport.drop(["ED2 Comdty","ED3 Comdty"])
        except Exception as e:
            print(e)
        #ControlPanelToReport = ControlPanelToReport.sort_values(by='CurrentPosition'+exposureSpaceIdentifier)
        ControlPanelToReport["-B1-"] = "-"
        ControlPanelToReport = ControlPanelToReport[[ControlPanel.iloc[:,0].name, "TotalSignal",
                                                     'CurrentPosition'+exposureSpaceIdentifier, 'KondorExposures', "TotalKondorExposures",
                                                     "-B1-",
                                                     'Fills_'+ExternalStrategy_A_Name, 'TargetPositions_'+ExternalStrategy_A_Name,
                                                     'exposureSharesEMSX']].fillna(0)
        ########################################################################################################################

        ControlPanelToReport["-B2-"] = "-----"
        ControlPanelToReport["PositionAdjustment"] = ControlPanelToReport.iloc[:,0] - ControlPanelToReport["KondorExposures"]
        #ControlPanel["PositionAdjustment"] = ControlPanelToReport["TotalSignal"] - ControlPanelToReport["TotalKondorExposures"]
        print(ControlPanel)
        ########################################################################################################################
        ControlPanelToReport["Kondor_vs_EMSX"] = ControlPanel["exposureSharesEMSX"] - ControlPanel["KondorExposures"]
        ControlPanelToReport["Kondor_vs_EMSX_FLAG"] = "-"
        ControlPanelToReport.loc[(ControlPanelToReport["Kondor_vs_EMSX"] != 0)&(~ControlPanel["KondorExposures"].isna()),"Kondor_vs_EMSX_FLAG"] = "CHECK !!!"
        ControlPanelToReport.to_sql("ControlPanel", self.conn, if_exists='replace')
        ControlPanelToReport['CurrentPosition'+exposureSpaceIdentifier].to_excel(self.KplusExposuresPath+"EMSX_ControlPanel.xlsx")

        ControlPanelToReport = ControlPanelToReport.fillna(0).reset_index()
        ControlPanelToReport['HaveExposureFlag'] = pe.sign(ControlPanelToReport['KondorExposures']).abs()
        ControlPanelToReport = ControlPanelToReport.sort_values(by='HaveExposureFlag', ascending=False)
        ControlPanelToReport = ControlPanelToReport.drop(["HaveExposureFlag", "exposureSharesEMSX","Kondor_vs_EMSX", "Kondor_vs_EMSX_FLAG"],axis=1)

        pe.RefreshableFile([[ControlPanelToReport, 'QuantitativeStrategiesControlPanel']],
                        self.GreenBoxFolder + 'QuantitativeStrategies_ControlPanel.html',
                        5, cssID='QuantitativeStrategies',  addButtons="QuantStrategies")

        ##########################################################################################################################################################################
        #print(ControlPanelToReport)
        ControlPanelToReport_ActivePositions = ControlPanelToReport.loc[ControlPanelToReport.iloc[:,1] != 0,["Asset",ControlPanelToReport.columns[1],"CurrentPositionKondor","PositionAdjustment"]]
        ControlPanelToReport_ActivePositions = ControlPanelToReport_ActivePositions.set_index("Asset",drop=True)
        ControlPanelToReport_ActivePositions.index.names = ["ticker"]
        ControlPanelToReport_ActivePositions = pd.concat([ControlPanelToReport_ActivePositions,self.ActiveAssetsReferenceData.loc[ControlPanelToReport_ActivePositions.index, ["CRNCY","CURR_GENERIC_FUTURES_SHORT_NAME"]]], axis=1)
        Active_FUT_CONT_SIZES = pd.read_sql('SELECT * FROM HistContractValues', sqlite3.connect("DataDeck.db"))
        Active_FUT_CONT_SIZES = Active_FUT_CONT_SIZES.loc[Active_FUT_CONT_SIZES.index[-1],list(ControlPanelToReport_ActivePositions.index)]
        ControlPanelToReport_ActivePositions["CONT_SIZES"] = Active_FUT_CONT_SIZES
        ControlPanelToReport_ActivePositions["Notionals"] = ControlPanelToReport_ActivePositions.iloc[:,0] * ControlPanelToReport_ActivePositions["CONT_SIZES"]
        ControlPanelToReport_ActivePositions[["Notionals","CONT_SIZES"]] = ControlPanelToReport_ActivePositions[["Notionals","CONT_SIZES"]].astype(float).round()
        cta_lookbacks = pd.read_sql('SELECT * FROM HealthCheck_1_1', sqlite3.connect(self.AlternativeStorageLocation+"HealthChecks.db")).set_index("date",drop=True)
        cta_lookbacks = cta_lookbacks.loc[cta_lookbacks.index[-1], ControlPanelToReport_ActivePositions.index]
        cta_lookbacks.name = "Lookbacks"
        ControlPanelToReport_ActivePositions = pd.concat([ControlPanelToReport_ActivePositions, cta_lookbacks], axis=1)
        ControlPanelToReport_ActivePositions["Trading Style (Monthly Horizon)"] = "Trend Following"
        ControlPanelToReport_ActivePositions.loc[ControlPanelToReport_ActivePositions["Lookbacks"] <= 25, "Trading Style (Monthly Horizon)"] = "Mean Reversion"
        ControlPanelToReport_ActivePositions = ControlPanelToReport_ActivePositions.drop(["CurrentPositionKondor","PositionAdjustment","Lookbacks"],axis=1)
        ControlPanelToReport_ActivePositions = ControlPanelToReport_ActivePositions.reset_index()
        ControlPanelToReport_ActivePositions = ControlPanelToReport_ActivePositions.dropna(subset=["CONT_SIZES"])

        ##########################################################################################################################################################################

        "Get Credit Trader Signals in the Report"
        CreditTraderControlPanelExcel = "F:\Dealing\TRADING\High Yield Desk\Credit_Trader_Python\Credit_Trader_Control_Panel.xlsx"
        AssetsSpecsSheet = pd.read_excel(CreditTraderControlPanelExcel, sheet_name="AssetsSpecs",engine='openpyxl').set_index("Asset", drop=True).dropna()
        CreditTraderSignals = pd.read_sql('SELECT * FROM RedPill_LiveSignal',sqlite3.connect("F:/Dealing/TRADING/High Yield Desk/Credit_Trader_Python/CreditTrader.db")).set_index("index",drop=True)
        CreditTraderSignals = CreditTraderSignals.astype(float) * 10000000
        CreditTraderSignals["Risk On/Off"] = None
        CreditTraderSignals.loc[CreditTraderSignals[CreditTraderSignals.columns[0]] < 0, "Risk On/Off"] = "Risk On"
        CreditTraderSignals.loc[CreditTraderSignals[CreditTraderSignals.columns[0]] > 0, "Risk On/Off"] = "Risk Off"
        CreditTraderSignals.loc[CreditTraderSignals[CreditTraderSignals.columns[0]] == 0, "Risk On/Off"] = "Neutral"
        CreditTraderSignals = pd.concat([AssetsSpecsSheet["Description"], CreditTraderSignals],axis=1)
        CreditTraderSignals = CreditTraderSignals.reset_index()
        CreditTraderSignals.columns = ["Credit Trader : Tickers", "Name", "Position (Notional) : "+CreditTraderSignals.columns[1], "Risk On/Off"]
        CreditTraderSignals["Trading Style (Monthly Horizon)"] = "Mean Reversion"

        ##########################################################################################################################################################################

        "Get SOV Trader Signals in the Report"
        SOVTraderControlPanelExcel = "F:\Dealing\TRADING\Govies\QIS\DevQuantSystems\SOV_Trader_Control_Panel_Dev.xlsx"
        SOV_AssetsSpecsSheet = pd.read_excel(SOVTraderControlPanelExcel, sheet_name="ActiveStrategies",engine='openpyxl').set_index("Govies", drop=True)
        SOV_TraderConn = sqlite3.connect("F:\Dealing\TRADING\Govies\QIS\DevQuantSystems\SOV_TraderDev.db")
        SOVTraderSignals = pd.read_sql('SELECT * FROM Govies_LiveSignal',SOV_TraderConn).set_index("index",drop=True)
        SOVTraderSignals = (SOVTraderSignals.astype(float) * 10000000 * (-1)).fillna(0)
        SOVTraderSignals["Risk On/Off"] = "Neutral"
        SOVTraderSignals.loc[SOVTraderSignals[SOVTraderSignals.columns[0]] > 0, "Risk On/Off"] = "Risk On"
        SOVTraderSignals.loc[SOVTraderSignals[SOVTraderSignals.columns[0]] < 0, "Risk On/Off"] = "Risk Off"
        SOVTraderSignals.loc[SOVTraderSignals[SOVTraderSignals.columns[0]] == 0, "Risk On/Off"] = "Neutral"
        SOV_AssetsSpecsSheet = SOV_AssetsSpecsSheet.loc[SOVTraderSignals.index]
        SOVTraderSignals = pd.concat([SOV_AssetsSpecsSheet["Description"], SOVTraderSignals],axis=1)
        SOVTraderSignals = SOVTraderSignals.reset_index()
        SOVTraderSignals.columns = ["SOV Trader : Tickers", "Name", "Position (Notional) : "+SOVTraderSignals.columns[1], "Risk On/Off"]
        SOVTraderSignals = SOVTraderSignals.set_index("SOV Trader : Tickers",drop=True)
        SOVTrader_lookbacks = pd.read_sql('SELECT * FROM Govies_LookBacks', SOV_TraderConn).set_index("date", drop=True)
        SOVTrader_lookbacks = SOVTrader_lookbacks.loc[SOVTrader_lookbacks.index[-1], :]
        SOVTrader_lookbacks.name = "Lookbacks"
        SOVTraderSignals = pd.concat([SOVTraderSignals, SOVTrader_lookbacks], axis=1)
        SOVTraderSignals["Trading Style (Monthly Horizon)"] = "Trend Following"
        SOVTraderSignals.loc[SOVTraderSignals["Lookbacks"] <= 25, "Trading Style (Monthly Horizon)"] = "Mean Reversion"
        SOVTraderSignals = SOVTraderSignals.reset_index()
        SOVTraderSignals = SOVTraderSignals.drop(["Lookbacks"], axis=1)
        ###############################################################################################################################################################

        pe.RefreshableFile([[ControlPanelToReport_ActivePositions, 'QuantitativeStrategiesControlPanel'],
                            [CreditTraderSignals, 'QuantitativeStrategiesControlPanel'],
                            [SOVTraderSignals, 'QuantitativeStrategiesControlPanel']],
                        self.GreenBoxFolder + 'QuantitativeStrategies_ControlPanel_ActivePositions.html',
                        5, cssID='QuantitativeStrategies',  addButtons="QuantStrategiesHermes")

        ###############################################################################################################################################################

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
#mode = "Stage"
#mode = "StageNoDataUpdate"
#mode = "CheckPositionsOnly_STEALTH_YES"
#mode = "CheckPositionsOnly_STEALTH_NO"

if (mode == "Live")|("CheckPositionsOnly" in mode):
    tradingObj = TimeStories("ActiveStrategies")
elif mode == "Stage":
    tradingObj = TimeStories("StagedStrategies")
elif mode == "StageNoDataUpdate":
    tradingObj = TimeStories("StagedStrategies")

if "CheckPositionsOnly" not in mode:
    tradingObj.LiveStrategiesRun()
    tradingObj.ControlPanel(LiveFuturesMaturity="M4", Stealth="NO") #Stealth="NO","YES"
else:
    print("Running CheckPositions with EMSX Stealth Mode = ", mode.split("_")[2])
    tradingObj.ControlPanel(LiveFuturesMaturity="M4", Stealth=mode.split("_")[2])  # Stealth="NO","YES"

if "Live" in mode:
    #pass
    #from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import RiskStatusIdentification
    #################################################################################################
    print("Hermes is sending emails ... ")
    from PyEurobankBloomberg.PySystems.PyLiveTradingSystems import Hermes
