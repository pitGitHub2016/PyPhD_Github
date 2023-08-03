import pandas as pd, numpy as np, matplotlib.pyplot as plt, pdblp, sqlite3, os, time
import quantstats as qs
from pyerb import pyerb as pe
from PyEurobankBloomberg.PySystems.LiveOrderRoute import main as orderRoute
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems.StrategiesDeck import StrategiesDeck
import warnings
warnings.filterwarnings("ignore")

class Skeleton:

    def __init__(self, DB, SystemName, tradeMode, StrategySelection, ActiveAssets, AUM, AccCRNCY, Leverage, **kwargs):
        self.DB = DB
        self.SystemName = SystemName
        self.tradeMode = tradeMode
        self.StrategySelection = StrategySelection
        self.ActiveAssets = ActiveAssets
        self.AUM = AUM
        self.AccCRNCY = AccCRNCY
        self.Leverage = Leverage
        self.conn = sqlite3.connect(self.DB)
        if self.tradeMode == "PRODUCTION":
            self.DataDeckDB = "DataDeck.db"
        elif self.tradeMode == "STAGE":
            self.DataDeckDB = "DataDeck_Staged.db"
        elif self.tradeMode == "RESEARCH":
            self.DataDeckDB = "DataDeck_Research.db"
        self.DataDeckConn = sqlite3.connect(self.DataDeckDB)
        self.pyBloomyGetDataTable = "ActiveAssets_"+self.SystemName
        self.PositionsMonitorFolder = "C:/blp/data/"
        self.factSheetReportPath = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\StrategiesFactSheets/"
        self.assetsData = pd.read_excel("AssetsDashboard.xlsx", sheet_name="DataDeck",engine='openpyxl')

        """GET ASSETS REFERENCE DATA"""
        DataDeck_ActiveAssetsReferenceData = pd.read_sql('SELECT * FROM ActiveAssetsReferenceData', self.DataDeckConn).set_index("ticker", drop=True)
        self.ActiveAssetsReferenceData = DataDeck_ActiveAssetsReferenceData[DataDeck_ActiveAssetsReferenceData.index.isin(self.ActiveAssets)]
        self.ActiveAssetsReferenceData.to_sql("ActiveAssetsReferenceData", self.conn, if_exists='replace')
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable', self.DataDeckConn).set_index("index", drop=True)

        """GET forex DATA"""
        self.forexData = pd.read_sql('SELECT * FROM forexData', self.DataDeckConn).set_index("date", drop=True)
        self.forexData.to_sql("forexData", self.conn, if_exists='replace')

        """GET Historical Contract Values DATA"""
        self.HistContractValues = pd.read_sql('SELECT * FROM HistContractValues', self.DataDeckConn).set_index("date", drop=True)[self.ActiveAssets]
        self.HistContractValues.to_sql("HistContractValues", self.conn, if_exists='replace')

        self.HistContractValues_Diff = pe.d(self.HistContractValues).fillna(0)
        self.HistContractValues_Diff.to_sql("HistContractValues_Diff", self.conn, if_exists='replace')

        """GET ASSETS TIME SERIES"""
        DataDeck_df = pd.read_sql('SELECT * FROM DataDeck', self.DataDeckConn)
        DataDeck_df = DataDeck_df.set_index('date', drop=True)
        self.df = DataDeck_df[self.ActiveAssets]
        self.df.to_sql(self.pyBloomyGetDataTable, self.conn, if_exists='replace')

        self.rets = pe.dlog(self.df) #pe.r(self.df,calcMethod='Linear')
        self.rets.to_sql(self.SystemName + "_rets", self.conn, if_exists='replace')

        if 'Benchmark' in kwargs:
            self.BenchmarkRets = self.rets[kwargs['Benchmark']]
        else:
            self.BenchmarkRets = DataDeck_df.loc[:,"RX1 Comdty"]

        """GET Indicators DATA"""
        self.indicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', self.DataDeckConn).set_index("date", drop=True)
        self.indicatorsDF.to_sql("IndicatorsDeck", self.conn, if_exists='replace')

        print("Success!")

    def StrategyRun(self, signalShift, **kwargs):
        def getIndexes(dfObj, value):

            # Empty list
            listOfPos = []

            # isin() method will return a dataframe with
            # boolean values, True at the positions
            # where element exists
            result = dfObj.isin([value])

            # any() method will return
            # a boolean series
            seriesObj = result.any()

            # Get list of column names where
            # element exists
            columnNames = list(seriesObj[seriesObj == True].index)

            # Iterate over the list of columns and
            # extract the row index where element exists
            for col in columnNames:
                rows = list(result[col][result[col] == True].index)

                for row in rows:
                    listOfPos.append((row, col))

            # This list contains a list tuples with
            # the index of element in the dataframe
            return listOfPos
        print("======================================================================================================")
        print("RUNNING "+self.SystemName+" ... ")
        print("======================================================================================================")
        if "ShowPnL" in kwargs:
            ShowPnL = kwargs['ShowPnL']
        else:
            ShowPnL = "Yes"
        ############################################################################################################
        #trKernel = pe.fd(1/(100*pe.rollVol(self.rets, nIn=25))) ---> Risk Parity example
        "THE MAIN TRADING SIGNAL KERNEL OF THE STRATEGY"
        an_object = StrategiesDeck(self.SystemName, self.df, self.indicatorsDF, self.Leverage)
        class_method = getattr(StrategiesDeck, self.StrategySelection)
        trKernel = class_method(an_object)
        trKernel.to_sql(self.SystemName+"_trKernel", self.conn, if_exists='replace')
        ############################################################################################################

        print(self.SystemName+" : TARGET NOTIONALS AND NUMBER OF CONTRACTS")
        targetNotionals = (trKernel * self.AUM).sort_index()
        targetNotionals.to_sql(self.SystemName + "_targetNotionalsAccCrncy", self.conn, if_exists='replace')
        for c in targetNotionals.columns:
            assetCurrency = self.ActiveAssetsReferenceData[self.ActiveAssetsReferenceData.index == c]["CRNCY"]
            tnDF = pd.DataFrame(targetNotionals[c])
            #print(c, ", ", assetCurrency)
            if assetCurrency.values[0] != self.AccCRNCY:
                ConvertCurrency = self.forexData[self.AccCRNCY+assetCurrency+" Curncy"]
                targetNotionals[c] = tnDF * ConvertCurrency.rename(columns=dict(zip(ConvertCurrency.columns, tnDF.columns)))
        targetNotionals.to_sql(self.SystemName + "_targetNotionalsAssetCrncy", self.conn, if_exists='replace')
        numContractsDF = targetNotionals / self.HistContractValues
        numContractsDF.to_sql(self.SystemName + "_numContractsDF", self.conn, if_exists='replace')
        print(self.SystemName + " : ROUND THE NUMBER OF FUTURES CONTRACTS TRADED")
        roundContractsDF = numContractsDF.round(0)
        roundContractsDF.to_sql(self.SystemName + "_roundContractsDF", self.conn, if_exists='replace')
        roundContractsDF_shifted = pe.S(roundContractsDF, nperiods=signalShift)
        roundContractsDF_shifted.to_sql(self.SystemName + "_roundContractsDF_shifted", self.conn, if_exists='replace')
        roundContractsDFdiff = pe.d(roundContractsDF).fillna(0)
        roundContractsDFdiff.to_sql(self.SystemName + "_roundContractsDFdiff", self.conn, if_exists='replace')

        print(self.SystemName + " : CALCULATE TRANSACTIONS COSTS")
        cashPnlTCDF = roundContractsDFdiff.copy()
        for c in roundContractsDFdiff.columns:
            #print(c, getIndexes(self.FuturesTable, c)[0][0], self.FuturesTable["Point_1"].iloc[getIndexes(self.FuturesTable, c)[0][0]])
            ActiveContract = self.FuturesTable["Point_1"].iloc[getIndexes(self.FuturesTable, c)[0][0]]
            cashPnlTCDF[c] = roundContractsDFdiff[c].abs() * self.assetsData[self.assetsData["Asset"] == ActiveContract]["Total Cost"].values[0]
        cashPnlTCDF.to_sql(self.SystemName + "_cashPnlTCDF", self.conn, if_exists='replace')

        print(self.SystemName+" : TARGET NOTIONALS AFTER ROUNDING")
        targetNotionalsAfterRounding = roundContractsDF * self.HistContractValues
        targetNotionalsAfterRounding.to_sql(self.SystemName + "_targetNotionalsAfterRounding", self.conn, if_exists='replace')

        ############################################################################################################
        "SHIFT THE SIGNAL KERNEL OF THE STRATEGY !!!"
        targetNotionalsAfterRounding_shifted = pe.S(targetNotionalsAfterRounding, nperiods=signalShift)
        targetNotionalsAfterRounding_shifted.to_sql(self.SystemName + "_targetNotionalsAfterRounding_shifted", self.conn, if_exists='replace')
        ############################################################################################################

        print(self.SystemName+" : GROSS AND NET PNLs")
        #GrossCashPnl = targetNotionalsAfterRounding_shifted * self.rets # FIRST VERSION - BASED ON RETS
        GrossCashPnl = roundContractsDF_shifted * self.HistContractValues_Diff # BASED ON HISTORICAL CONTRACT VALUES
        GrossCashPnl.to_sql(self.SystemName+"_GrossCashPnl", self.conn, if_exists='replace')
        cumGrossCashPnl = pe.cs(GrossCashPnl)
        cumGrossCashPnl.to_sql(self.SystemName+"_cumGrossCashPnl", self.conn, if_exists='replace')

        NetCashPnl = GrossCashPnl - cashPnlTCDF
        NetCashPnl.to_sql(self.SystemName+"_NetCashPnl", self.conn, if_exists='replace')
        cumNetCashPnl = pe.cs(NetCashPnl)
        cumNetCashPnl.to_sql(self.SystemName+"_cumNetCashPnl", self.conn, if_exists='replace')

        print(self.SystemName+" : CONVERT NET CASH PNL INTO ACCOUNT CURRENCY TERMS")
        NetCashPnlAccCrncy = NetCashPnl.copy()
        for c in NetCashPnlAccCrncy.columns:
            assetCurrency = self.ActiveAssetsReferenceData[self.ActiveAssetsReferenceData.index == c]["CRNCY"]
            ncDF = pd.DataFrame(NetCashPnlAccCrncy[c])
            if assetCurrency.values[0] != self.AccCRNCY:
                ConvertCurrency = self.forexData[self.AccCRNCY+assetCurrency+" Curncy"]
                NetCashPnlAccCrncy[c] = ncDF / ConvertCurrency.rename(columns=dict(zip(ConvertCurrency.columns, ncDF.columns)))
        NetCashPnlAccCrncy.to_sql(self.SystemName + "_NetCashPnlAccCrncy", self.conn, if_exists='replace')

        print(self.SystemName+" : GET STRATEGY'S RETURNS")
        NetRets = NetCashPnlAccCrncy / self.AUM
        NetRets.to_sql(self.SystemName + "_NetRets", self.conn, if_exists='replace')
        csNetRets = pe.cs(NetRets)
        csNetRets.to_sql(self.SystemName + "_CumNetRets", self.conn, if_exists='replace')
        StrategyNetRets = pe.rs(NetRets)
        StrategyNetRets.name = self.SystemName
        StrategyNetRets.to_sql(self.SystemName + "_StrategyNetRets", self.conn, if_exists='replace')

        print(self.SystemName+" : QUANTSTATS HTML REPORT")
        StrategyNetRets.index = pd.DatetimeIndex(StrategyNetRets.index)
        StrategyNetRets = StrategyNetRets.sort_index()
        self.BenchmarkRets.index = pd.DatetimeIndex(self.BenchmarkRets.index)
        self.BenchmarkRets = self.BenchmarkRets.sort_index()
        qs.extend_pandas()
        reportTitle = self.SystemName + " " + "[" + ','.join(self.df.columns) + ']'
        try:
            qs.reports.html(StrategyNetRets, compounded=False, title=reportTitle, benchmark=self.BenchmarkRets, output=self.factSheetReportPath+self.SystemName+".html")
        except Exception as e:
            print(e)
            qs.reports.html(StrategyNetRets, compounded=False, title=reportTitle, output=self.factSheetReportPath+self.SystemName+".html")

        #########################################################################################################
        "GET THE SIGNAL TO BE ROUTED TODAY!"
        latestroundContractsDF = roundContractsDF.iloc[-1*signalShift]
        latestroundContractsDF.to_sql(self.SystemName + "_latestroundContractsDF", self.conn, if_exists='replace')
        #########################################################################################################

        print(self.SystemName+" : BUILD POSITIONS DASHBOARD")
        pos0DF = latestroundContractsDF.reset_index().rename(columns={"index": "ticker"}).set_index("ticker", drop=True)
        #print(pos0DF)
        pos1DF = self.df.iloc[-1].reset_index().rename(columns={"index": "ticker"}).set_index("ticker", drop=True)
        #print(pos1DF)
        pos2DF = self.ActiveAssetsReferenceData
        #print(pos2DF)
        positions = pd.concat([pos0DF, pos1DF, pos2DF], axis=1)
        positions["ConvertCurrency"] = self.AccCRNCY + positions["CRNCY"] + " Curncy"
        positions["Direction"] = ""
        positions["Direction"][positions[latestroundContractsDF.name] > 0] = "BUY"
        positions["Direction"][positions[latestroundContractsDF.name] < 0] = "SELL"
        positions.to_sql(self.SystemName + "_LATEST_Trades", self.conn, if_exists='replace')

        RollingSharpe = np.sqrt(252) * pe.roller(StrategyNetRets, pe.sharpe, 250)
        ExpandingSharpe = np.sqrt(252) * pe.expander(StrategyNetRets, pe.sharpe, 25)

        if ShowPnL == "Yes":
            fig, axes = plt.subplots(nrows=3, ncols=1)
            cumNetCashPnl.plot(ax=axes[0], title="cumNetCashPnl")
            pe.cs(pe.rs(NetCashPnl)).plot(ax=axes[1], title="E_NetCashPnl")
            RollingSharpe.plot(ax=axes[2], title="Sharpes")
            plt.show()

        print("%%%%%% RecentActivity Data %%%%%")
        print(self.df.tail(5))
        print("%%%%%% RecentActivity NetCashPnL %%%%%")
        print(NetCashPnl.tail(5))
        #print("%%%%%%%%%%%%%%%%%%%%%%%%% LATEST ROLLING SHARPE %%%%%%%%%%%%%%%%%%%%%%%%")
        #print(RollingSharpe.tail(5))
        #print("%%%%%%%%%%%%%%%%%%%%%%%%% LATEST EXPANDING SHARPE %%%%%%%%%%%%%%%%%%%%%%%%")
        #print(ExpandingSharpe.tail(5))
        print("%%%%%%%%%%%%%%%%%%%%%%%%% Dashboard %%%%%%%%%%%%%%%%%%%%%%%%")
        print(positions.T)

        TC = cumGrossCashPnl.iloc[-1] - cumNetCashPnl.iloc[-1]
        TC.to_sql(self.SystemName + "_TC", self.conn, if_exists='replace')

    def PositionsMonitor(self):
        def buildFileDF(positionsFolder):
            fileData = []
            for filename in os.listdir(positionsFolder):
                if filename.startswith("EUROBANK_EMSX_FILLS_EXPORT"):
                    fileData.append([positionsFolder+filename, os.path.getctime(positionsFolder+filename)])
            return pd.DataFrame(fileData, columns=["File", "Date"]).sort_values(by='Date', ascending=False).reset_index()

        fileDataDF = buildFileDF(self.PositionsMonitorFolder)
        fileDataDF.set_index("index", drop=True).to_sql(self.SystemName + "_fileDataDF_openPositions_EMSX", self.conn, if_exists='replace')
        if len(fileDataDF) > 1:
            print("MORE THAN ONE Positions-File present in the EUROBANK EMSX Positions Folder : " + self.PositionsMonitorFolder + " ---> Need to delete the old ones and leave the latest one standing ... ")
            for idx, row in fileDataDF.iterrows():
                if idx > 0:
                    os.remove(row['File'])
        else:
            print("Just one Positions File present in the EUROBANK EMSX Positions Folder : " + self.PositionsMonitorFolder)

        #RE-CHECK POSITIONS FILES : THERE SHOULD BE ONLY ONE POSITIONS FILE
        fileDataDF = buildFileDF(self.PositionsMonitorFolder)
        if len(fileDataDF) == 1:
            print("Just one Positions File present INDEED!")
            fillsDF = pd.read_csv(fileDataDF['File'].values[0])
            fillsDF.to_sql(self.SystemName + "_fills_EMSX", self.conn, if_exists='replace')
            openPositionsDF = fillsDF[fillsDF["Working Amount"] == 0]
            openPositionsDF.to_sql(self.SystemName + "_openPositions_EMSX", self.conn, if_exists='replace')
            openPositionsAggDF = openPositionsDF[["Ticker", "Fill Amount"]].groupby(by="Ticker").sum()
            openPositionsAggDF.to_sql(self.SystemName + "_openPositionsAggredated_EMSX", self.conn, if_exists='replace')
        else:
            print("Still multiple EUROBANK EMSX Positions Files Present!")

    def OrderHandler(self):
        positionsDF = pd.read_sql('SELECT * FROM ' + self.SystemName + "_LATEST_Trades", self.conn)
        print(positionsDF)
        orderRoute([self.SystemName, positionsDF])

