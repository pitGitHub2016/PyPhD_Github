import time
import subprocess
#try:
from os.path import dirname, basename, isfile, join
import glob, os, sys, pdblp, math
from datetime import timedelta
from scipy.stats import norm
import pickle
import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt, matplotlib as mpl
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')
from tqdm import tqdm
from pyerb import pyerb as pe
import quantstats as qs
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def BBG_Tenor_Handle(rawIn, mode):
    out = rawIn

    if mode == "FromBBGtoMonths":

        if rawIn == "1Y":
            out = "12M"
        elif rawIn == "2Y":
            out = "24M"
        elif rawIn == "3Y":
            out = "36M"

    elif mode == "FromMonthsToBBG":

        if rawIn == "12M":
            out = "1Y"
        elif rawIn == "24M":
            out = "2Y"
        elif rawIn == "36M":
            out = "3Y"

    return out

# //////////////////////////// STRATEGIES / FILTERS ////////////////////////////////////

def StrategiesTrader(Strategy, GenericDF, ID, **kwargs):
    RepoStoragepath = "C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/VolatilityScanner/"

    if 'thr' in kwargs:
        thr = kwargs['thr']
    else:
        thr = 50

    if 'GreeksCols' in kwargs:
        GreeksCols = kwargs['GreeksCols']
    else:
        GreeksCols = ["DeltaCalls","GammaCalls","ThetaCalls","VegaCalls","DeltaPuts","GammaPuts","ThetaPuts","VegaPuts"]

    if "KillOneLeg" in Strategy:
        "Keep The previous in the repo storage space"
        GenericDF.to_excel(RepoStoragepath+ID+"_PriorTo_"+Strategy+"_Strategy.xlsx")

        GenericDF["CallsPct"] = GenericDF["CallsPrices"] / GenericDF["StraddlePrices"]
        GenericDF["PutsPct"] = GenericDF["PutsPrices"] / GenericDF["StraddlePrices"]
        GenericDF.loc[GenericDF["CallsPct"] <= float(thr) / 100, "CallsPrices"] = None
        GenericDF.loc[GenericDF["PutsPct"] <= float(thr) / 100, "PutsPrices"] = None

        GenericDF["StraddlePrices"] = GenericDF["CallsPrices"] + GenericDF["PutsPrices"]
        "Correct Greeks"
        for c in GreeksCols:
            if "Call" in c:
                GenericDF[c] *= pe.sign(GenericDF["CallsPrices"])
            else:
                GenericDF[c] *= pe.sign(GenericDF["PutsPrices"])

        GenericDF["CallsPct"] = GenericDF["CallsPrices"] / GenericDF["StraddlePrices"]
        GenericDF["PutsPct"] = GenericDF["PutsPrices"] / GenericDF["StraddlePrices"]

        return GenericDF

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VolatilityScanner:

    def __init__(self, sheet, DB, **kwargs):
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        DB = self.AlternativeStorageLocation + DB
        self.HFT_PnL_DB_Path = "F:/Dealing/Panagiotis Papaioannou/MT5/HTML_Reports/LIVE/"
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"
        self.ExcelControlPanel = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="Control Panel").set_index("Parameter", drop=True)
        self.LiveStrategies = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="LiveStrategies")
        self.LiveStrategiesNames = list(self.LiveStrategies.columns)
        self.LiveStrategies_SubAllocations = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="LiveStrategies_SubAllocations")
        self.LiveStrategies_TCA = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="LiveStrategies_TCA").set_index("Generic",drop=True)
        self.VolPortfolioWeights = self.ExcelControlPanel.loc[self.LiveStrategiesNames]
        self.ExcelDF = pd.read_excel("AssetsDashboard.xlsx", sheet_name=sheet)
        self.Assets = self.ExcelDF["Assets"].dropna().tolist()
        self.tenorList = ["1W","2W","1M","2M","3M","6M","9M","1Y","2Y","3Y"]# ||||| "ON","1D","3W","2M","4M","5M","18M","2Y","3Y","4Y","5Y","6Y","7Y","10Y","15Y","20Y","25Y","30Y"
        #self.tenorList = ["1W"] #DEBUGGER
        self.DeltaSpanList = ["25"]#"5","10","35"
        self.workConn = sqlite3.connect(DB,detect_types=sqlite3.PARSE_DECLTYPES)
        self.PuntNotional = self.ExcelControlPanel.loc["Punt Notional", "Value"]
        self.TimeOverridesList = self.ExcelControlPanel.loc["TimeOverrideSettings", "Value"].split("|")
        self.AUM = self.ExcelControlPanel.loc["AUM", "Value"]
        self.LEVERAGE = self.ExcelControlPanel.loc["LEVERAGE", "Value"]
        self.WeeklyHorizon = self.ExcelControlPanel.loc["Weekly Horizon", "Value"]
        self.BiWeeklyHorizon = self.ExcelControlPanel.loc["BiWeekly Horizon", "Value"]
        self.MonthlyHorizon = self.ExcelControlPanel.loc["Monthly Horizon", "Value"]
        self.RatesSetup = 0
        self.FWDSetup = 0
        ##########################################################################################################################
        self.PySystemsPath = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/"
        self.IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', sqlite3.connect(self.PySystemsPath+"PyLiveTradingSystems/DataDeck.db")).set_index('date', drop=True)
        self.IndicatorsDF.index = pd.DatetimeIndex(self.IndicatorsDF.index)
        self.BenchmarkDF = pe.dlog(self.IndicatorsDF["NEIXCTA Index"]).fillna(0)#NEIXSTTI Index
        self.BenchmarkDF_Vol = self.BenchmarkDF.std()
        ############################################################################################################################################################
        self.SubAllocations = pd.concat([self.LiveStrategies['SingleGenerics'], self.LiveStrategies_SubAllocations['SingleGenerics']], axis=1)
        self.SubAllocations.columns = ['Strategy', 'Allocation']
        self.SubAllocations = self.SubAllocations.dropna(subset=['Strategy'])
        self.SubAllocations = self.SubAllocations.set_index('Strategy', drop=True)
        ############################################################################################################################################################
        self.ECO_Relevance_Path = 'F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/GMAA Observatory/ECO HistData/ToMT5/'

    def SetupRatesMatrix(self):
        self.Rates = pd.DataFrame(None, index=["USD","EUR","JPY","GBP","CHF","CAD", "AUD", "NZD"], columns=self.tenorList)
        for idx, row in self.Rates.iterrows():
            if idx == "USD":
                for c in row.index:
                    if c == "1W":
                        self.Rates.loc[idx,c] = "USOSFR1Z BGN Curncy"
                    elif c == "2W":
                        self.Rates.loc[idx, c] = "USOSFR2Z BGN Curncy"
                    elif c == "1M":
                        self.Rates.loc[idx, c] = "USOSFRA BGN Curncy"
                    elif c == "2M":
                        self.Rates.loc[idx, c] = "USOSFRB BGN Curncy"
                    elif c == "3M":
                        self.Rates.loc[idx, c] = "USOSFRC BGN Curncy"
                    elif c == "6M":
                        self.Rates.loc[idx, c] = "USOSFRF BGN Curncy"
                    elif c == "9M":
                        self.Rates.loc[idx, c] = "USOSFRI BGN Curncy"
                    elif c == "1Y":
                        self.Rates.loc[idx, c] = "USOSFR1 BGN Curncy"
                    elif c == "2Y":
                        self.Rates.loc[idx, c] = "USOSFR2 BGN Curncy"
                    elif c == "3Y":
                        self.Rates.loc[idx, c] = "USOSFR3 BGN Curncy"
            elif idx == "EUR":
                for c in row.index:
                    if c == "1W":
                        self.Rates.loc[idx,c] = "EESWE1Z Curncy"
                    elif c == "2W":
                        self.Rates.loc[idx, c] = "EESWE2Z Curncy"
                    elif c == "1M":
                        self.Rates.loc[idx, c] = "EUR001M Index"
                    elif c == "2M":
                        self.Rates.loc[idx, c] = "EESWEB Curncy"
                    elif c == "3M":
                        self.Rates.loc[idx, c] = "EUR003M Index"
                    elif c == "6M":
                        self.Rates.loc[idx, c] = "EUR006M Index"
                    elif c == "9M":
                        self.Rates.loc[idx, c] = "EESWEI Curncy"
                    elif c == "1Y":
                        self.Rates.loc[idx, c] = "EUSA1 Curncy"
                    elif c == "2Y":
                        self.Rates.loc[idx, c] = "EUSA2 Curncy"
                    elif c == "3Y":
                        self.Rates.loc[idx, c] = "EUSA3 Curncy"
            elif idx == "GBP":
                for c in row.index:
                    if c == "1W":
                        self.Rates.loc[idx,c] = "BPSWS1Z Curncy"
                    elif c == "2W":
                        self.Rates.loc[idx, c] = "BPSWS2Z Curncy"
                    elif c == "1M":
                        self.Rates.loc[idx, c] = "BP0001M Index"
                    elif c == "2M":
                        self.Rates.loc[idx, c] = "BPSWSB Curncy"
                    elif c == "3M":
                        self.Rates.loc[idx, c] = "BP0003M Index"
                    elif c == "6M":
                        self.Rates.loc[idx, c] = "BP0006M Index"
                    elif c == "9M":
                        self.Rates.loc[idx, c] = "BPSWSI Curncy"
                    elif c == "1Y":
                        self.Rates.loc[idx, c] = "BPSWS1 Curncy"
                    elif c == "2Y":
                        self.Rates.loc[idx, c] = "BPSWS2 Curncy"
                    elif c == "3Y":
                        self.Rates.loc[idx, c] = "BPSWS3 Curncy"
            elif idx == "CAD":
                for c in row.index:
                    if c == "1W":
                        self.Rates.loc[idx,c] = "CDSO1Z Curncy"
                    elif c == "2W":
                        self.Rates.loc[idx, c] = "CDSO2Z Curncy"
                    elif c == "1M":
                        self.Rates.loc[idx, c] = "CDSOA Curncy"
                    elif c == "2M":
                        self.Rates.loc[idx, c] = "CDSOB Curncy"
                    elif c == "3M":
                        self.Rates.loc[idx, c] = "CDSOC Curncy"
                    elif c == "6M":
                        self.Rates.loc[idx, c] = "CDSOF Curncy"
                    elif c == "9M":
                        self.Rates.loc[idx, c] = "CDSOI Curncy"
                    elif c == "1Y":
                        self.Rates.loc[idx, c] = "CDSW1 Curncy"
                    elif c == "2Y":
                        self.Rates.loc[idx, c] = "CDSW2 Curncy"
                    elif c == "3Y":
                        self.Rates.loc[idx, c] = "CDSW3 Curncy"

        self.Rates.to_sql("Rates", self.workConn, if_exists='replace')
        self.RatesSetup = 1

    def SetupFWDMatrix(self):
        self.FWD = pd.DataFrame(None, index=["EUR","JPY","GBP","CHF","CAD", "AUD", "NZD"], columns=self.tenorList)
        for idx, row in self.FWD.iterrows():
            for c in row.index:
                if c == "1Y":
                    cMask = "12M"
                else:
                    cMask = c
                self.FWD.loc[idx,c] = idx+cMask+" Curncy"

        self.FWD.to_sql("FWD", self.workConn, if_exists='replace')
        self.FWDSetup = 1

    def CreateVolatilityGenerics(self):
        self.genericsList = []
        for asset in self.Assets:
            assetSplit = asset.split(" ")
            for tenor in self.tenorList:
                volAssetTicker = assetSplit[0]+"V"+tenor+" "+assetSplit[1]
                self.genericsList.append(volAssetTicker)
                for delta in self.DeltaSpanList:
                    for iv in ["DC", "DP", "DR", "DB"]:
                        ImpliedVolTicker = assetSplit[0]+" " +tenor+" "+delta+iv+" VOL BVOL "+assetSplit[1]
                        self.genericsList.append(ImpliedVolTicker)
                    for volStrat in ["R", "B"]:
                        RiskReveralButterfliesTicker = assetSplit[0]+delta+volStrat+tenor+" "+assetSplit[1]
                        self.genericsList.append(RiskReveralButterfliesTicker)

        "CHECK WHICH ASSETS HAVE ACTIVE-FILLED-WITH-DATA TARGET VOL TICKERS"
        con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
        self.CleanGenericsList = []
        for ticker in self.genericsList:
            try:
                fetchedDF = con.ref(ticker, ["PX_LAST"])
                self.CleanGenericsList.append(ticker)
            except Exception as e:
                print(e)

        pd.DataFrame(self.CleanGenericsList, columns=["GenericNames"]).set_index("GenericNames", drop=True).to_sql("genericsList", self.workConn, if_exists='replace')

    def getData(self):
        field = "PX_LAST"
        self.Rates = pd.read_sql('SELECT * FROM Rates', self.workConn).set_index("index", drop=True).dropna()
        self.FWD = pd.read_sql('SELECT * FROM FWD', self.workConn).set_index("index", drop=True).dropna()

        for mode in ["TradingAssets", "AtTheMoneyVols", "RiskReversals", "Butterflies"]:
            try:
                self.customDataDF = pd.read_sql('SELECT * FROM DataDeck_'+mode, self.workConn).set_index('date', drop=True)
                startDate = self.customDataDF.index[-5].strftime("%Y-%m-%d").split(" ")[0].replace("-", "")
                print(startDate)
                self.updateStatus = 'update'
            except Exception as e:
                print(e)
                startDate = '20000101'
                self.updateStatus = 'fetchSinceInception'

            # Force Since Inception Update
            #startDate = '20000101'; self.updateStatus = 'fetchSinceInception'

            print(mode, ", self.updateStatus = ", self.updateStatus)
            time.sleep(1)
            if (self.RatesSetup == 1)|(self.FWDSetup == 1):
                print("NEED TO UPDATE DATA SINCE INCEPTION FROM SCRATCH!!!")
                startDate = '20000101'
                self.updateStatus = 'fetchSinceInception'

            self.genericsList = pd.read_sql('SELECT * FROM genericsList', self.workConn)['GenericNames'].tolist()

            if mode == "TradingAssets":
                self.AllTickers = self.Assets
                for c in self.Rates.columns:
                    for elem in self.Rates[c].tolist():
                        self.AllTickers.append(elem)
                for c in self.FWD.columns:
                    for elem in self.FWD[c].tolist():
                        self.AllTickers.append(elem)
            elif mode == "AtTheMoneyVols":
                self.AllTickers = [x for x in self.genericsList if "V" in x]
            elif mode == "RiskReversals":
                self.AllTickers = [x for x in self.genericsList if "R" in x]
            elif mode == "Butterflies":
                self.AllTickers = [x for x in self.genericsList if "B" in x]

            self.AllTickers = list(set(self.AllTickers))

            self.con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
            self.fetchData = self.con.bdh(self.AllTickers, [field], startDate, '21000630').ffill().bfill()
            self.fetchData.columns = [x[0].replace("'", "") for x in self.fetchData.columns]
            self.fetchData.to_sql("bloombergDataDF_"+mode, self.workConn, if_exists='replace')
            self.bloombergDataDF = pd.read_sql('SELECT * FROM bloombergDataDF_'+mode, self.workConn).set_index('date', drop=True)
            if self.updateStatus == 'fetchSinceInception':
                self.customDataDF = self.bloombergDataDF
            else:
                self.customDataDF = pd.concat([self.customDataDF, self.bloombergDataDF], axis=0)
                self.customDataDF = self.customDataDF[~self.customDataDF.index.duplicated(keep='last')]

            self.customDataDF.to_sql("DataDeck_"+mode, self.workConn, if_exists='replace')

        self.GGCDF = pd.concat([pd.read_sql('SELECT * FROM DataDeck_TradingAssets', self.workConn).set_index('date', drop=True),
                           pd.read_sql('SELECT * FROM DataDeck_AtTheMoneyVols', self.workConn).set_index('date', drop=True)],axis=1)

        self.GGCDF.to_sql("GGCDF", self.workConn, if_exists='replace')

    # ////////////////////// MAIN RUNNER ///////////////////////////////////////////

    def CreateOptionsGenerics(self, ConfigIn, plotOrNot):

        self.ID = ConfigIn['ID']

        # //////////////////////////////////////////////////////////////////////////////////////////////////////////

        self.Rates = pd.read_sql('SELECT * FROM Rates', self.workConn).set_index("index", drop=True).dropna()

        self.GGCDF = pd.read_sql('SELECT * FROM GGCDF', self.workConn).set_index('date', drop=True).sort_index()#.tail(150)
        self.GGCDF.index = pd.to_datetime(self.GGCDF.index)

        baseC = ConfigIn["Pair"][:3]
        quoteC = ConfigIn["Pair"][3:].replace(" Curncy", "")

        BaseDF = self.GGCDF[[ConfigIn["Pair"], ConfigIn["Pair"].split(" ")[0]+"V"+ConfigIn['Tenor']+" Curncy"]]

        for timeOverridePack in self.TimeOverridesList:
            timeOverrideSettings = [int(x) for x in timeOverridePack.split(",")]
            TimeOverrideDF = pe.TimeOverride(self.AlternativeStorageLocation+"MT5/"+ConfigIn['Pair'].split(" ")[0]+".eub_M1_Latest.csv", timeOverrideSettings[0], timeOverrideSettings[1], ConfigIn["Pair"])
            BaseDF.loc[TimeOverrideDF.index, ConfigIn["Pair"]] = TimeOverrideDF
        try:
            BaseDF['MarketFwd'] = ConfigIn["Pair"].split(" ")[0]+ConfigIn['Tenor'].replace("1Y","12M")+" Curncy"
            if "JPY" not in ConfigIn["Pair"]:
                BaseDF.loc[:,'MarketFwd'] = BaseDF.loc[:,ConfigIn["Pair"]] + BaseDF.loc[:,'MarketFwd'] / 10000
            else:
                BaseDF.loc[:,'MarketFwd'] = BaseDF.loc[:,ConfigIn["Pair"]] + BaseDF.loc[:,'MarketFwd'] / 100
        except Exception as e:
            pass
            #print(e)
        #####################################################################################################################
        BaseDF["r1"] = self.GGCDF[self.Rates.loc[baseC, ConfigIn['Tenor']]]
        BaseDF["r2"] = self.GGCDF[self.Rates.loc[quoteC, ConfigIn['Tenor']]]
        #####################################################################################################################
        "HITMAN HANDLER"
        if "W" in ConfigIn['Hitman']:
            toPeriod = 'W'
        elif "M" in ConfigIn['Hitman']:
            toPeriod = 'M'
        elif "Y" in ConfigIn['Hitman']:
            toPeriod = 'M'
            ConfigIn['Hitman'] = BBG_Tenor_Handle(ConfigIn['Hitman'], "FromBBGtoMonths")
        #####################################################################################################################
        "TENOR HANDLER"
        if ConfigIn['Tenor'] == "1W":
            T_Base = int(round(float(ConfigIn['Tenor'].replace("W", "")) * self.WeeklyHorizon))
        elif ConfigIn['Tenor'] == "2W":
            T_Base = int(round(float(ConfigIn['Tenor'].replace("W", "")) * self.BiWeeklyHorizon))
        elif "M" in ConfigIn['Tenor']:
            T_Base = int(round(float(ConfigIn['Tenor'].replace("M", "")) * self.MonthlyHorizon))
        elif "Y" in ConfigIn['Tenor']:
            ConfigIn['Tenor'] = BBG_Tenor_Handle(ConfigIn['Tenor'], "FromBBGtoMonths")
            T_Base = int(round(float(ConfigIn['Tenor'].replace("M", "")) * self.MonthlyHorizon))
        #####################################################################################################################
        print("Building "+self.ID)
        print("self.Optionality_UpdateStatus = ", self.Optionality_UpdateStatus)

        self.TimersCols = ["index_col", "index_Dt"]
        self.OptionsCols = ["CallsPrices", "PutsPrices", "StraddlePrices",
                            "DeltaCalls", "GammaCalls", "ThetaCalls", "VegaCalls", "RhoCalls",
                            "DeltaPuts", "GammaPuts", "ThetaPuts", "VegaPuts", "RhoPuts"]
        #####################################################################################################################
        if ConfigIn["HitmanStrike"] == "EMA_Tenor":
            HitBase = pe.sma(self.GGCDF[ConfigIn['Pair']], nperiods=T_Base).round(4)
        elif ConfigIn["HitmanStrike"] == "Latest_Spot":
            HitBase = self.GGCDF[ConfigIn['Pair']].round(4)
        elif "SMA_2STDs_Strangle" in ConfigIn["HitmanStrike"]:
            sma = pe.sma(self.GGCDF[ConfigIn['Pair']], nperiods=T_Base).round(4)
            RollStd = self.GGCDF[ConfigIn['Pair']].rolling(T_Base).std()
            HitBase = (sma + 2*RollStd).round(4).bfill().astype(str) + "," + (sma - 2*RollStd).round(4).bfill().astype(str)
        #####################################################################################################################
        if ConfigIn["Optionality_Type"] == "Barriers":
            VolMarketData = pd.read_sql('SELECT * FROM DataDeck_RiskReversals', self.workConn).set_index('date', drop=True)
            if "STDs" in ConfigIn["BarrierRule"]:
                sma = pe.sma(self.GGCDF[ConfigIn['Pair']], nperiods=T_Base).round(4)
                RollStd = self.GGCDF[ConfigIn['Pair']].rolling(T_Base).std()
                if ConfigIn["barrierType"] in ["UpIn", "UpOut"]:
                    BarrierLevel = (sma + float(ConfigIn["BarrierRule"].replace("STDs","")) * RollStd).round(4).bfill()
                elif ConfigIn["barrierType"] in ["DownIn", "DownOut"]:
                    BarrierLevel = (sma - float(ConfigIn["BarrierRule"].replace("STDs","")) * RollStd).round(4).bfill()
        #####################################################################################################################
        HitBase[HitBase == 0] = None
        HitBase = HitBase.ffill().bfill()
        HitBase = pd.DataFrame(HitBase[~HitBase.index.to_period(toPeriod).duplicated()].iloc[::int(ConfigIn['Hitman'].replace("W", "").replace("M", ""))])
        HitBase["HitFlag"] = 0
        StrikeColName = ConfigIn['Pair'] + "_" + ConfigIn['Hitman'] + "_" + ConfigIn['Tenor'] + "_" + " Strike"
        HitBase.columns = [StrikeColName, "HitFlag"]

        HitBase = pe.S(HitBase).bfill()
        if self.Optionality_UpdateStatus == 'update':
            self.GGCDF_Trade_Previous = pd.read_sql('SELECT * FROM GGCDF_Trade_'+self.ID, self.workConn).iloc[:-1,:]
            self.GGCDF_Trade_Previous = self.GGCDF_Trade_Previous.set_index(self.GGCDF_Trade_Previous.columns[0], drop=True)
            self.GGCDF_Trade_Previous = self.GGCDF_Trade_Previous.drop([x for x in self.GGCDF_Trade_Previous.columns if (x in BaseDF.columns)|(x in HitBase.columns)], axis=1)
            self.GGCDF_Trade_Previous = self.GGCDF_Trade_Previous.fillna(0)
            self.GGCDF_Trade = pd.concat([self.GGCDF_Trade_Previous, BaseDF, HitBase], axis=1)
            self.GGCDF_Trade = self.GGCDF_Trade[~self.GGCDF_Trade.index.duplicated(keep='last')].sort_index()
        else:
            self.GGCDF_Trade = pd.concat([BaseDF, HitBase], axis=1)
            self.GGCDF_Trade["T_Adj"] = 0
            self.GGCDF_Trade["fwd"] = None
            self.GGCDF_Trade["MaturityDate"] = None
            self.GGCDF_Trade[self.OptionsCols] = None
            self.GGCDF_Trade["AdjustedVol"] = None

        self.GGCDF_Trade.loc[self.GGCDF_Trade["HitFlag"].shift() == 0, "HitFlag"] = 1

        self.GGCDF_Trade[HitBase.columns] = self.GGCDF_Trade[HitBase.columns].ffill()
        self.GGCDF_Trade["index_col"] = self.GGCDF_Trade.index
        self.GGCDF_Trade["index_Dt"] = self.GGCDF_Trade["index_col"].shift()
        self.GGCDF_Trade["ThetaDecayDays"] = ((self.GGCDF_Trade["index_col"] - self.GGCDF_Trade["index_Dt"]).dt.days) * self.GGCDF_Trade["HitFlag"]
        self.GGCDF_Trade[self.OptionsCols] = self.GGCDF_Trade[self.OptionsCols].astype(float)
        self.GGCDF_Trade = self.GGCDF_Trade.dropna(subset=["index_Dt"])

        if self.Optionality_UpdateStatus == 'update':
            processIndexes = self.GGCDF_Trade[self.GGCDF_Trade["T_Adj"].isna()].index
            T = self.GGCDF_Trade.dropna().iloc[-1]["T_Adj"]
        else:
            processIndexes = self.GGCDF_Trade.index
            T = T_Base

        IVcol = [x for x in self.GGCDF_Trade.columns if (ConfigIn["Pair"].split(" ")[0]+"V" in x)]
        for idx, row in tqdm(self.GGCDF_Trade.iterrows()):
            if idx in processIndexes:
                ############################################################################################################################
                if row["ThetaDecayDays"] == 0:
                    T = T_Base
                    self.GGCDF_Trade.loc[idx, "MaturityDate"] = self.GGCDF_Trade.loc[idx, "index_col"] + timedelta(days=T)
                    if self.GGCDF_Trade.loc[idx, "MaturityDate"].dayofweek >= 5:
                        self.GGCDF_Trade.loc[idx, "MaturityDate"] -= timedelta(days=self.GGCDF_Trade.loc[idx, "MaturityDate"].dayofweek - 4)
                        T = (self.GGCDF_Trade.loc[idx, "MaturityDate"]-self.GGCDF_Trade.loc[idx, "index_col"]).days
                else:
                    T = T - row["ThetaDecayDays"]
                    self.GGCDF_Trade.loc[idx, "MaturityDate"] = self.GGCDF_Trade.loc[idx, "index_col"] + timedelta(days=T)
                ############################################################################################################################
                self.GGCDF_Trade.loc[idx, "T_Adj"] = T
                ############################################################################################################################
                r1 = row["r1"] / 100
                r2 = row["r2"] / 100
                ############################################################################################################################
                K = row[HitBase.columns[0]]
                if isinstance(K, str):
                    K = [float(x) for x in K.split(",")]
                if ConfigIn["HitmanStrike"] == "SMA_2STDs_Strangle_ITM":
                    Kc = K[1]
                    Kp = K[0]
                elif ConfigIn["HitmanStrike"] == "SMA_2STDs_Strangle_OTM":
                    Kc = K[0]
                    Kp = K[1]
                else:
                    Kc = K
                    Kp = K
                ############################################################################################################################
                sigma = row[IVcol].values[0] / 100
                ############################################################################################################################
                "Linear depreciation of vol"
                #sigma *= (T/(T_Base-1))
                ############################################################################################################################
                self.GGCDF_Trade.loc[idx, "AdjustedVol"] = sigma*100
                ############################################################################################################################
                if ConfigIn["FwdCalculator"] == "Market":
                    fwd = self.GGCDF_Trade.loc[idx, "MarketFwd"]
                else:
                    fwd = row[ConfigIn["Pair"]] * math.exp((r2 - r1) * (T / 365))
                ############################################################################################################################
                self.GGCDF_Trade.loc[idx, "fwd"] = fwd
                ############################################################################################################################
                if ConfigIn["Optionality_Type"] == "Vanillas":
                    self.GGCDF_Trade.loc[idx, "CallsPrices"] = pe.black_scholes(fwd, Kc, T / 365, r2, sigma, "call")
                    self.GGCDF_Trade.loc[idx, "PutsPrices"] = pe.black_scholes(fwd, Kp, T / 365, r2, sigma, "put")
                    self.GGCDF_Trade.loc[idx, "StraddlePrices"] = self.GGCDF_Trade.loc[idx, "CallsPrices"] + self.GGCDF_Trade.loc[idx, "PutsPrices"]
                    ########################################################################################################################
                    delta_call, gamma_call, theta_call, vega_call, rho_call = pe.black_scholes_greeks(fwd, Kc, T / 365, r2, sigma, "call")
                    self.GGCDF_Trade.loc[idx, "DeltaCalls"] = delta_call
                    self.GGCDF_Trade.loc[idx, "GammaCalls"] = gamma_call
                    self.GGCDF_Trade.loc[idx, "ThetaCalls"] = theta_call
                    self.GGCDF_Trade.loc[idx, "VegaCalls"] = vega_call
                    self.GGCDF_Trade.loc[idx, "RhoCalls"] = rho_call
                    ########################################################################################################################
                    delta_put, gamma_put, theta_put, vega_put, rho_put = pe.black_scholes_greeks(fwd, Kp, T / 365, r2, sigma, "put")
                    self.GGCDF_Trade.loc[idx, "DeltaPuts"] = delta_put
                    self.GGCDF_Trade.loc[idx, "GammaPuts"] = gamma_put
                    self.GGCDF_Trade.loc[idx, "ThetaPuts"] = theta_put
                    self.GGCDF_Trade.loc[idx, "VegaPuts"] = vega_put
                    self.GGCDF_Trade.loc[idx, "RhoPuts"] = rho_put
                    ########################################################################################################################
                    self.GGCDF_Trade.loc[idx, self.OptionsCols] *= self.PuntNotional
                    ########################################################################################################################
                elif ConfigIn["Optionality_Type"] == "Barriers":
                    ########################################################################################################################
                    try:
                        ########################################################################################################################
                        BarrierOptionPricerSpecs_Call = {
                            'today': pd.to_datetime(self.GGCDF_Trade.loc[idx, "index_col"]),
                            'option_type': 'call',
                            'strike': Kc,
                            'barrier_type': 'UpIn',
                            'barrier': BarrierLevel.loc[idx],
                            'payoff_amt': int(self.PuntNotional.round(0)),
                            'expiry_dt': pd.to_datetime(self.GGCDF_Trade.loc[idx, "MaturityDate"]),
                            'spot': row[ConfigIn["Pair"]],
                            'vol_atm': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "V" + ConfigIn['Tenor'] + " Curncy"],
                            'vol_rr': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "25R" + ConfigIn['Tenor'] + " Curncy"],
                            'vol_bf': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "25B" + ConfigIn['Tenor'] + " Curncy"],
                            'rd': r1 * 100,
                            'rf': r2 * 100,
                        }
                        ########################################################################################################################
                        BarrierOptionPricerSpecs_Put = {
                            'today': pd.to_datetime(self.GGCDF_Trade.loc[idx, "index_col"]),
                            'option_type': 'put',
                            'strike': Kp,
                            'barrier_type': 'UpIn',
                            'barrier': BarrierLevel.loc[idx],
                            'payoff_amt': int(self.PuntNotional.round(0)),
                            'expiry_dt': pd.to_datetime(self.GGCDF_Trade.loc[idx, "MaturityDate"]),
                            'spot': row[ConfigIn["Pair"]],
                            'vol_atm': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "V" + ConfigIn['Tenor'] + " Curncy"],
                            'vol_rr': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "25R" + ConfigIn['Tenor'] + " Curncy"],
                            'vol_bf': VolMarketData.loc[
                                idx, ConfigIn["Pair"].split(" ")[0] + "25B" + ConfigIn['Tenor'] + " Curncy"],
                            'rd': r1 * 100,
                            'rf': r2 * 100,
                        }
                        ########################################################################################################################
                        BarrierOptionPricerData_Call = pe.Barrier(BarrierOptionPricerSpecs_Call)
                        BarrierOptionPricerData_Put = pe.Barrier(BarrierOptionPricerSpecs_Put)
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "CallsPrices"] = BarrierOptionPricerData_Call['Price']
                        self.GGCDF_Trade.loc[idx, "PutsPrices"] = BarrierOptionPricerData_Put['Price']
                        self.GGCDF_Trade.loc[idx, "StraddlePrices"] = self.GGCDF_Trade.loc[idx, "CallsPrices"] + self.GGCDF_Trade.loc[idx, "PutsPrices"]
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "DeltaCalls"] = BarrierOptionPricerData_Call['delta']
                        self.GGCDF_Trade.loc[idx, "GammaCalls"] = BarrierOptionPricerData_Call['gamma']
                        self.GGCDF_Trade.loc[idx, "ThetaCalls"] = BarrierOptionPricerData_Call['theta']
                        self.GGCDF_Trade.loc[idx, "VegaCalls"] = BarrierOptionPricerData_Call['vega']
                        self.GGCDF_Trade.loc[idx, "RhoCalls"] = np.nan
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "DeltaPuts"] = BarrierOptionPricerData_Put['delta']
                        self.GGCDF_Trade.loc[idx, "GammaPuts"] = BarrierOptionPricerData_Put['gamma']
                        self.GGCDF_Trade.loc[idx, "ThetaPuts"] = BarrierOptionPricerData_Put['theta']
                        self.GGCDF_Trade.loc[idx, "VegaPuts"] = BarrierOptionPricerData_Put['vega']
                        self.GGCDF_Trade.loc[idx, "RhoPuts"] = np.nan
                    except Exception as e:
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "CallsPrices"] = np.nan
                        self.GGCDF_Trade.loc[idx, "PutsPrices"] = np.nan
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "DeltaCalls"] = np.nan
                        self.GGCDF_Trade.loc[idx, "GammaCalls"] = np.nan
                        self.GGCDF_Trade.loc[idx, "ThetaCalls"] = np.nan
                        self.GGCDF_Trade.loc[idx, "VegaCalls"] = np.nan
                        self.GGCDF_Trade.loc[idx, "RhoCalls"] = np.nan
                        ########################################################################################################################
                        self.GGCDF_Trade.loc[idx, "DeltaPuts"] = np.nan
                        self.GGCDF_Trade.loc[idx, "GammaPuts"] = np.nan
                        self.GGCDF_Trade.loc[idx, "ThetaPuts"] = np.nan
                        self.GGCDF_Trade.loc[idx, "VegaPuts"] = np.nan
                        self.GGCDF_Trade.loc[idx, "RhoPuts"] = np.nan
        # ////////////////////////////////////////////////////////////////////////////////////////////////////
        self.GGCDF_Trade["MaturityDate"] = pd.to_datetime(self.GGCDF_Trade["MaturityDate"])

        "Fixing the Columns Order"
        self.GGCDF_Trade = self.GGCDF_Trade[[ConfigIn["Pair"], 'fwd', 'T_Adj', 'MaturityDate', StrikeColName,
                                            'CallsPrices', 'PutsPrices', 'StraddlePrices',
                                            'ThetaDecayDays','HitFlag', IVcol[0], "AdjustedVol", 'r1', 'r2',
                                            'DeltaCalls', 'GammaCalls', 'ThetaCalls', 'VegaCalls', 'RhoCalls',
                                            'DeltaPuts', 'GammaPuts', 'ThetaPuts', 'VegaPuts', 'RhoPuts',
                                            'index_col', 'index_Dt']]

        ################################################ "STRATEGIES / FILTERS" ##########################################################
        "APPLY STRATEGIES/FILTERS SIGNALS"
        #Kill_Thr = 50
        #if (ConfigIn["Pair"] == "EURUSD Curncy")&("W" in ConfigIn['Tenor'])&("W" in ConfigIn['Hitman']):
        #    Kill_Thr = 50
        #RunStrategy = "KillOneLeg_Upper"
        #RunStrategy = "KillOneLeg_Lower"
        #self.GGCDF_Trade = StrategiesTrader(RunStrategy, self.GGCDF_Trade, self.ID, thr=Kill_Thr)
        ##################################################################################################################################
        "SAVE TO DB"
        #self.GGCDF_Trade = self.GGCDF_Trade.iloc[:-5,:] # DEBUGGING
        self.GGCDF_Trade.to_sql("GGCDF_Trade_"+self.ID, self.workConn, if_exists='replace')

        "LATEST TRADES INFO"
        lastLine = self.GGCDF_Trade.iloc[-1]
        self.LatestTradeInfo = [str(lastLine.name), StrikeColName, lastLine[ConfigIn["Pair"]], lastLine[StrikeColName],
                                float(lastLine["T_Adj"]), lastLine["MaturityDate"], float(lastLine["CallsPrices"]),
                                float(lastLine["PutsPrices"]), float(lastLine["StraddlePrices"]),
                                float(lastLine["DeltaCalls"])+float(lastLine["DeltaPuts"]),
                                float(lastLine["GammaCalls"])+float(lastLine["GammaPuts"]),
                                float(lastLine["ThetaCalls"])+float(lastLine["ThetaPuts"]),
                                float(lastLine["VegaCalls"])+float(lastLine["VegaPuts"])]

        if plotOrNot == 1:
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            self.GGCDF_Trade[["CallsPrices","PutsPrices"]].plot(ax=ax[0])
            self.GGCDF_Trade["StraddlePrices"].plot(ax=ax[1])
            plt.show()

    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def RunBuilder(self, mode, Optionality_UpdateStatus, **kwargs):

        if 'skipReadCalc' in kwargs:
            skipReadCalc = kwargs['skipReadCalc']
        else:
            skipReadCalc = 'NO'

        if 'DailyDeltaHedgeFlag' in kwargs:
            DailyDeltaHedgeFlag = kwargs['DailyDeltaHedgeFlag']
        else:
            DailyDeltaHedgeFlag = 'NO'

        if 'ECO_Relevance_Filter' in kwargs:
            ECO_Relevance_Filter = kwargs['ECO_Relevance_Filter']
        else:
            ECO_Relevance_Filter = ['YES',90]

        if 'TCA_Mode' in kwargs:
            self.TCA_Mode = kwargs['TCA_Mode']#'VegaCharge'
        else:
            self.TCA_Mode = 'PriceCharge'

        self.Optionality_UpdateStatus = Optionality_UpdateStatus

        if mode == "Run":
            NotYetProcessed = []
            for Pair in tqdm(["EURUSD Curncy","GBPUSD Curncy","USDCAD Curncy","EURGBP Curncy"]):#"EURUSD Curncy","GBPUSD Curncy",
                for TenorPack in [["1W", "1W"],
                                  #["1W", "2W"],["1W", "1M"],
                                  #["2W", "1W"],["2W", "2W"],["2W", "1M"],
                                  #["1M", "1W"],["1M", "2W"],["1M", "1M"],
                                  ]: #self.tenorList,"2W"
                    Hitman = TenorPack[0]
                    Tenor  = TenorPack[1]
                    for Optionality_Type in ["Vanillas"]:#"Barriers"
                        for HitManStrikeMode in ["Latest_Spot",
                                                 "SMA_2STDs_Strangle_OTM",
                                                 "SMA_2STDs_Strangle_ITM"
                                                 ]:
                            ########################################################################################################
                            if Optionality_Type == "Vanillas":
                                InputConfig = {"Pair": Pair, "Hitman": Hitman, "Tenor": Tenor,
                                               "Optionality_Type": Optionality_Type,
                                               "HitmanStrike": HitManStrikeMode,
                                               "FwdCalculator": "Native",
                                               }  # FwdCalculator=Market,Native
                                ################################################################################################
                                InputConfig['ID'] = '_'.join(InputConfig.values()).replace(" ", "_")
                                ################################################################################################
                                if self.Optionality_UpdateStatus == "":
                                    try:
                                        df = pd.read_sql('SELECT * FROM GGCDF_Trade_' + InputConfig['ID'], self.workConn)
                                    except Exception as e:
                                        print(e)
                                        NotYetProcessed.append(InputConfig)
                                else:
                                    NotYetProcessed.append(InputConfig)
                            ########################################################################################################
                            elif Optionality_Type == "Barriers":
                                for barrierType in ["UpIn", "UpOut", "DownIn", "DownOut"]:
                                    for BarrierRule in ["2STDs"]:
                                        InputConfig = {"Pair": Pair, "Hitman": Hitman, "Tenor": Tenor,
                                                       "Optionality_Type": Optionality_Type,
                                                       "barrierType": barrierType,
                                                       "BarrierRule": BarrierRule,
                                                       "HitmanStrike": HitManStrikeMode,
                                                       "FwdCalculator": "Native"}  # FwdCalculator=Market,Native
                                        ################################################################################################
                                        InputConfig['ID'] = '_'.join(InputConfig.values()).replace(" ", "_")
                                        ################################################################################################
                                        if self.Optionality_UpdateStatus == "":
                                            try:
                                                df = pd.read_sql('SELECT * FROM GGCDF_Trade_' + InputConfig['ID'],
                                                                 self.workConn)
                                            except Exception as e:
                                                print(e)
                                                NotYetProcessed.append(InputConfig)
                                        else:
                                            NotYetProcessed.append(InputConfig)
            ########################################################################################################
            print("len(NotYetProcessed) = ", len(NotYetProcessed))
            ########################################################################################################
            StraddlesList = []
            HitFlagList = []
            LatestTradeInfoList = []
            for subInputConfig in NotYetProcessed:
                try:
                    ################################################################################################
                    VolatilityScanner.CreateOptionsGenerics(self, subInputConfig, 0)
                    ################################################################################################
                    self.GGCDF_Trade = self.GGCDF_Trade[self.GGCDF_Trade["T_Adj"] >= 0]
                    subStraddle = self.GGCDF_Trade["StraddlePrices"]
                    subStraddle.name = self.ID
                    subHitFlag = self.GGCDF_Trade["HitFlag"]
                    subHitFlag.name = self.ID
                    StraddlesList.append(subStraddle)
                    HitFlagList.append(subHitFlag)
                    LatestTradeInfoList.append(self.LatestTradeInfo)
                except Exception as e:
                    print(e)
            ########################################################################################################
            OptionalityTradeDF = pd.concat(StraddlesList, axis=1).sort_index()
            OptionalityHitDF = pd.concat(HitFlagList, axis=1).sort_index()
            ########################################################################################################
            LatestTradeInfoDF = pd.DataFrame(LatestTradeInfoList, columns=["Last Date", "OptionID", "RefSpot", "Strike",
                                                                           "DaysToMaturity", "MaturityDate",
                                                                           "CallsPrices", "PutsPrices", "Straddles",
                                                                           "TotalDelta","TotalGamma","TotalTheta","TotalVega"]).set_index("OptionID", drop=True)
            ########################################################################################################
            OptionalityTradeDF.to_sql("OptionalityTradeDF", self.workConn, if_exists='replace')
            OptionalityHitDF.to_sql("OptionalityHitDF", self.workConn, if_exists='replace')
            LatestTradeInfoDF.to_sql("LatestTradeInfoDF", self.workConn, if_exists='replace')

        elif mode == "Trade":

            def DailyDeltaHedgePnL(GenericDF, **kwargs):

                if "HedgeList" in kwargs:
                    HedgeList = kwargs['HedgeList']
                else:
                    HedgeList = GenericDF.columns

                if "HedgeMode" in kwargs:
                    HedgeMode = kwargs['HedgeMode']
                else:
                    HedgeMode = "DailyDelta"

                deltaPnL_List = []
                for c in HedgeList:
                    ##########################################################################################################
                    subGeneric = pd.read_sql('SELECT * FROM GGCDF_Trade_'+c, self.workConn).set_index("date", drop=True)
                    ##########################################################################################################
                    if HedgeMode == "DailyDelta":
                        TotalDelta = subGeneric["DeltaCalls"]+subGeneric["DeltaPuts"]
                    ##########################################################################################################
                    deltaPnL = (-1) * pe.S(TotalDelta, nperiods=1).fillna(0) * pe.dlog(subGeneric[subGeneric.columns[0]])
                    deltaPnL.name = c
                    deltaPnL_List.append(deltaPnL)

                deltaPnL_DF = pd.concat(deltaPnL_List,axis=1).sort_index()

                return deltaPnL_DF

            OptionalityTradeDF = pd.read_sql('SELECT * FROM OptionalityTradeDF', self.workConn).set_index("date", drop=True)
            OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)
            ################################################################################################################################
            if self.TCA_Mode is not None:
                TCA_DF = pd.DataFrame(0,index=OptionalityHitDF.index,columns=OptionalityHitDF.columns)
                TotalVegaDF = pd.DataFrame(0,index=OptionalityHitDF.index,columns=OptionalityHitDF.columns)
                for c in TCA_DF.columns:
                    sub_GGCDF_Trade = pd.read_sql('SELECT * FROM GGCDF_Trade_' + c, self.workConn).set_index("date", drop=True)
                    ################################################################################################
                    EntryIndexes = sub_GGCDF_Trade[sub_GGCDF_Trade["ThetaDecayDays"] == 0].index
                    ExitIndexes = sub_GGCDF_Trade[sub_GGCDF_Trade["ThetaDecayDays"].shift(-1) == 0].index
                    TotalVegaDF[c] = (sub_GGCDF_Trade["VegaCalls"].abs() + sub_GGCDF_Trade["VegaPuts"]).fillna(0).abs()
                    if self.TCA_Mode == 'VegaCharge':
                        try:
                            VegaCharges = [float(x) for x in self.LiveStrategies_TCA.loc[c, "VegaChargeDirectional"].split(",")]
                            TCA_DF.loc[EntryIndexes,c] += VegaCharges[0] * TotalVegaDF[c]
                            TCA_DF.loc[ExitIndexes,c] += VegaCharges[1] * TotalVegaDF[c]
                        except Exception as e:
                            print(e)
                    elif 'PriceCharge' in self.TCA_Mode:
                        ################################################################################################
                        try:
                            PriceCharges = [float(x) for x in self.LiveStrategies_TCA.loc[c, "PriceChargeDirectional"].split(",")]
                            ################################################################################################
                            TCA_DF.loc[EntryIndexes,c] += (sub_GGCDF_Trade.loc[EntryIndexes, "StraddlePrices"] * PriceCharges[0]).abs()
                            TCA_DF.loc[ExitIndexes,c] += (sub_GGCDF_Trade.loc[ExitIndexes, "StraddlePrices"] * PriceCharges[1]).abs()
                        except Exception as e:
                            print(e)
                        ################################################################################################
                TCA_DF.to_sql("TCA_DF", self.workConn, if_exists='replace')
                TotalVegaDF.to_sql("TotalVegaDF", self.workConn, if_exists='replace')
                ####################################################################################################################
                try:
                    TCA_DF_RV = pe.RV(TCA_DF, mode="Baskets", RVspace="classicCombos")
                    TCA_DF_RV_Baskets = TCA_DF_RV.copy()
                    TCA_DF_RV_Baskets.columns = [x+"_Basket" for x in TCA_DF_RV_Baskets.columns]
                except Exception as e:
                    print(e)
                ####################################################################################################################
                TCA_Pct_DF = TCA_DF / self.AUM
                TCA_Pct_DF.to_sql("TCA_Pct_DF", self.workConn, if_exists='replace')
            ####################################################################################################################
            dTradeDF = pe.d(OptionalityTradeDF)
            dTradeDF.to_sql("dTradeDF", self.workConn, if_exists='replace')
            ####################################################################################################################
            if DailyDeltaHedgeFlag == 'YES':
                DailyDeltaHedgePnL_DF = DailyDeltaHedgePnL(dTradeDF, HedgeList=dTradeDF.columns, HedgeMode="DailyDelta")
                DailyDeltaHedgePnL_DF.to_sql("DailyDeltaHedgePnL_DF", self.workConn, if_exists='replace')
                for col in self.SubAllocations.index:#DailyDeltaHedgePnL_DF.columns:
                    try:
                        subDailyDeltaHedgePnL_DF = DailyDeltaHedgePnL_DF[col] / self.AUM
                        subDailyDeltaHedgePnL_DF *= self.SubAllocations.loc[col,'Allocation']
                        subDailyDeltaHedgePnL_DF.index = pd.to_datetime(subDailyDeltaHedgePnL_DF.index)
                        qs.reports.html(subDailyDeltaHedgePnL_DF, compounded=False, title="DailyDeltaHedgePnL_DF_"+col, output="StrategiesFactSheets/DailyDeltaHedgePnL_DF_"+col+".html")
                    except Exception as e:
                        print(e)
                "Neutralise Entry-Trade Day"
                GGC_Generic_Trader = (dTradeDF + (1)*DailyDeltaHedgePnL_DF) * OptionalityHitDF
            else:
                "Neutralise Entry-Trade Day"
                GGC_Generic_Trader = dTradeDF * OptionalityHitDF
            ####################################################################################################################
            GGC_Generic_Trader.to_sql("GGC_Generic_Trader_withDailyHedge", self.workConn, if_exists='replace')
            ####################################################################################################################
            if ECO_Relevance_Filter[0] == 'YES':
                EcoRelevance_Filter_Mega_DF = pd.DataFrame(None, index=GGC_Generic_Trader.index, columns=GGC_Generic_Trader.columns)
                for c in GGC_Generic_Trader.columns:
                    cSplit = c.split("_")
                    BaseCcy = cSplit[0][:3]
                    QuoteCcy = cSplit[0][3:]
                    ##############################################################
                    BaseCcy_EcoRelevance = pd.read_csv(self.ECO_Relevance_Path+BaseCcy+"_Daily.csv").set_index("Index",drop=True).sort_index()
                    BaseCcy_EcoRelevance.index = pd.to_datetime(BaseCcy_EcoRelevance.index)
                    ##############################################################
                    QuoteCcy_EcoRelevance = pd.read_csv(self.ECO_Relevance_Path+QuoteCcy+"_Daily.csv").set_index("Index",drop=True).sort_index()
                    QuoteCcy_EcoRelevance.index = pd.to_datetime(QuoteCcy_EcoRelevance.index)
                    ##############################################################
                    ReversOrNeutral = 0 #0,-1
                    BaseCcy_EcoRelevance[BaseCcy_EcoRelevance <= ECO_Relevance_Filter[1]] = ReversOrNeutral
                    QuoteCcy_EcoRelevance[QuoteCcy_EcoRelevance <= ECO_Relevance_Filter[1]] = ReversOrNeutral
                    ##############################################################
                    #EcoRelevance_Filter_Mega_DF[c] = pe.sign(pe.sign(BaseCcy_EcoRelevance) + pe.sign(QuoteCcy_EcoRelevance))
                    EcoRelevance_Filter_Mega_DF[c] = (-1) * (pe.sign(pe.sign(BaseCcy_EcoRelevance) + pe.sign(QuoteCcy_EcoRelevance))) + 1
                    "Short Only"
                    #EcoRelevance_Filter_Mega_DF[c] *= -1
                    ##############################################################
                    TempEcoDF = pd.concat([GGC_Generic_Trader[c],EcoRelevance_Filter_Mega_DF[c]],axis=1).dropna()
                    ##############################################################
                    GGC_Generic_Trader[c] = TempEcoDF.iloc[:,0]*TempEcoDF.iloc[:,1]
                    ##############################################################
                EcoRelevance_Filter_Mega_DF.to_sql("EcoRelevance_Filter_Mega_DF", self.workConn,if_exists='replace')
            ####################################################################################################################
            GGC_Generic_Trader.to_sql("GGC_Generic_Trader_withECO_Relevance_Filter", self.workConn, if_exists='replace')
            ####################################################################################################################
            """
            GGC_Generic_Trader_RV = pe.RV(GGC_Generic_Trader, RVspace="classicCombos")
            GGC_Generic_Trader_RV_Baskets = pe.RV(GGC_Generic_Trader, mode="Baskets", RVspace="classicCombos")
            GGC_Generic_Trader_RV_Baskets.columns = [x+"_Basket" for x in GGC_Generic_Trader_RV_Baskets.columns]
            if skipReadCalc == "NO":
                for dataPack in [[GGC_Generic_Trader, "GGC_Generic_Trader"], [GGC_Generic_Trader_RV, "GGC_Generic_Trader_RV"], [GGC_Generic_Trader_RV_Baskets, "GGC_Generic_Trader_RV_Baskets"]]:
                    print(dataPack[1])
                    if dataPack[1] == "GGC_Generic_Trader":
                    #try:
                    #    pass
                        dataPack[0].to_sql(dataPack[1], self.workConn, if_exists='replace')
                    #except Exception as e:
                    #    print(e)
                    #    filename = dataPack[1]
                    #    outfile = open(filename, 'wb')
                    #    pickle.dump(dataPack[0]+".p", outfile)
                    #    outfile.close()
                    sh_GGC_Generic_Trader = np.sqrt(252) * pe.sharpe(dataPack[0]).sort_values(ascending=False)
                    sh_GGC_Generic_Trader.to_sql("sh_"+dataPack[1], self.workConn, if_exists='replace')
            GGC_Generic_Trader = pd.concat([GGC_Generic_Trader, GGC_Generic_Trader_RV, GGC_Generic_Trader_RV_Baskets], axis=1).fillna(0)
            """
            VolStratsData = []
            for StratSpace in ["SingleGenerics"]: #"RVsSpreadsGenerics", "RVsBasketsGenerics"
                ActiveAllocations_Index = self.LiveStrategies_SubAllocations[StratSpace].dropna().loc[self.LiveStrategies_SubAllocations[StratSpace]!=0].index
                ###############################################################################################
                StratPortfolioWeight = self.VolPortfolioWeights.loc[StratSpace].values[0]
                StratContributions = self.LiveStrategies[StratSpace].iloc[ActiveAllocations_Index].dropna().tolist()
                StratContributions_SubAllocations = self.LiveStrategies_SubAllocations[StratSpace].iloc[ActiveAllocations_Index].fillna(0).tolist()
                StratContributionsList = []
                for StratContribution in StratContributions:
                    try:
                        subStratContributionsPnL = GGC_Generic_Trader[StratContribution].fillna(0)
                        StratContributionsList.append(subStratContributionsPnL)
                    except Exception as e:
                        print(e)
                StratContributionsPnL = pd.concat(StratContributionsList,axis=1)
                if StratSpace in ["SingleGenerics"]:
                    StratContributionsPnL.to_sql("StratContributionsPnL_"+StratSpace, self.workConn, if_exists='replace')
                ################################################################################################
                print("Active Optionality : ")
                k = 0
                for c in StratContributionsPnL.columns:
                    ########################################################################
                    StratContributionsPnL[c] *= StratContributions_SubAllocations[k]
                    ########################################################################
                    if self.TCA_Mode is not None:
                        print('TCA ...')
                        if StratSpace in ["SingleGenerics"]:
                            StratContributionsPnL[c] -= TCA_DF[c] * abs(StratContributions_SubAllocations[k])
                        elif StratSpace in ["RVsSpreadsGenerics"]:
                            StratContributionsPnL[c] -= TCA_DF_RV[c] * abs(StratContributions_SubAllocations[k])
                        elif StratSpace in ["RVsBasketsGenerics"]:
                            StratContributionsPnL[c] -= TCA_DF_RV_Baskets[c] * abs(StratContributions_SubAllocations[k])
                    ########################################################################
                    print("k=", k, ", c = ", c, ", ", StratContributions_SubAllocations[k], abs(StratContributions_SubAllocations[k]))
                    k += 1
                ################################################################################################
                if StratSpace in ["SingleGenerics"]:
                    StratContributionsPnL.to_sql("StratContributionsPnL_Net_"+StratSpace, self.workConn, if_exists='replace')
                ################################################################################################
                subDF = StratPortfolioWeight * pe.rs(StratContributionsPnL)
                subDF.name = StratSpace
                VolStratsData.append(subDF)
            ####################################################################################################
            VolatilityScannerPnL = pd.concat(VolStratsData, axis=1).loc['2008-01-01 00:00:00':,:]
            VolatilityScannerPnL *= self.LEVERAGE
            VolatilityScannerPnL.to_sql("VolatilityScannerPnL", self.workConn, if_exists='replace')
            ####################################################################################################
            VolatilityScannerLive = pe.rs(VolatilityScannerPnL)
            VolatilityScannerLive /= self.AUM # RETURNS
            shVolatilityScannerLive = np.sqrt(252) * pe.sharpe(VolatilityScannerLive)
            print("shVolatilityScannerLive = ", shVolatilityScannerLive)
            ####################################################################################################
            VolatilityScannerLive = pd.DataFrame(VolatilityScannerLive, columns=["VolatilityScannerLive"])
            VolatilityScannerLive.index = pd.to_datetime(VolatilityScannerLive.index)
            VolatilityScannerLive.to_sql("VolatilityScannerLive", self.workConn, if_exists='replace')
            try:
                qs.reports.html(VolatilityScannerLive["VolatilityScannerLive"], compounded=False, title="VolScanner",benchmark=self.BenchmarkDF*(VolatilityScannerLive["VolatilityScannerLive"].std()/self.BenchmarkDF.std()),
                                output="StrategiesFactSheets/VolatilityScannerPnL.html")
            except Exception as e:
                print(e)
                qs.reports.html(VolatilityScannerLive["VolatilityScannerLive"], compounded=False, title="VolScanner", output="StrategiesFactSheets/VolatilityScannerPnL.html")
            ###########################################################################################################
            try:
                # MERGE EVERYTHING WITH THE MAIN CTA & HFT COMPONENTS
                TOTAL_CTA = pd.read_sql('SELECT * FROM TotalPortfolioReturns', sqlite3.connect("StrategiesAggregator.db"))
                TOTAL_CTA = TOTAL_CTA.set_index(TOTAL_CTA.columns[0],drop=True)
                TOTAL_CTA.index.names = ['date']
                TOTAL_CTA.index = pd.to_datetime(TOTAL_CTA.index)
                TOTAL_CTA.columns = ["CTA"]
                ###########################################################################################################
                TOTAL_HFT = pd.read_sql("SELECT * FROM Currently_Running_System_DailyRetsDF", sqlite3.connect(self.HFT_PnL_DB_Path+"ERB_MT5_Reporter.db")).set_index('Time', drop=True)
                TOTAL_HFT.index.names = ['date']
                TOTAL_HFT.index = pd.to_datetime(TOTAL_HFT.index)
                TOTAL_HFT["HFT"] = pe.rs(TOTAL_HFT)
                #pe.cs(TOTAL_HFT).plot()
                #plt.show()
                ###########################################################################################################
                TOTAL_SYSTEMATIC = pd.concat([TOTAL_CTA, TOTAL_HFT["HFT"], VolatilityScannerLive], axis=1).sort_index().fillna(0)
                TOTAL_SYSTEMATIC["TOTAL"] = pe.rs(TOTAL_SYSTEMATIC)
                print("############################ TOTAL_SYSTEMATIC #########################")
                print("Contributions Vols ", TOTAL_SYSTEMATIC.std() * np.sqrt(252))
                print("Contributions Sharpes ", pe.sharpe(TOTAL_SYSTEMATIC) * np.sqrt(252))
                TOTAL_SYSTEMATIC.to_sql("TOTAL_SYSTEMATIC", self.workConn, if_exists='replace')
                qs.reports.html(TOTAL_SYSTEMATIC["TOTAL"], compounded=False, title="TOTAL SYSTEMATIC", benchmark=self.BenchmarkDF*(TOTAL_SYSTEMATIC["TOTAL"].std()/self.BenchmarkDF.std()), output="StrategiesFactSheets/TOTAL_SYSTEMATIC.html")

                ################################################################################################################################################################
                TOTAL_ShoreDM = pd.read_sql('SELECT * FROM ShoreDM', sqlite3.connect("StrategiesAggregator.db")).set_index('date',drop=True)
                TOTAL_ShoreDM.index = pd.to_datetime(TOTAL_ShoreDM.index)
                TOTAL_ShoreDM = (TOTAL_ShoreDM / self.AUM)

                TOTAL_HFT_VolScanner_ShoreDM = pd.concat([TOTAL_ShoreDM, TOTAL_HFT["HFT"], VolatilityScannerLive], axis=1).sort_index().fillna(0).loc[TOTAL_HFT.index[0]:,:]
                TOTAL_HFT_VolScanner_ShoreDM["TOTAL"] = pe.rs(TOTAL_HFT_VolScanner_ShoreDM)
                print("############################ TOTAL_HFT_VolScanner_ShoreDM #########################")
                print("Contributions Vols ", TOTAL_HFT_VolScanner_ShoreDM.std() * np.sqrt(252))
                print("Contributions Sharpes ", pe.sharpe(TOTAL_HFT_VolScanner_ShoreDM) * np.sqrt(252))
                TOTAL_HFT_VolScanner_ShoreDM.to_sql("TOTAL_HFT_VolScanner_ShoreDM", self.workConn, if_exists='replace')
                qs.reports.html(TOTAL_HFT_VolScanner_ShoreDM["TOTAL"], compounded=False, title="TOTAL HFT+VolScanner+ShoreDM", benchmark=self.BenchmarkDF*(TOTAL_HFT_VolScanner_ShoreDM["TOTAL"].std()/self.BenchmarkDF.std()),output="StrategiesFactSheets/TOTAL_HFT_VolScanner_ShoreDM.html")
            except Exception as e:
                print(e)

    # ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    def StrategyTester(self, StrategyConfigIn):

        strID = StrategyConfigIn["StrategyID"]
        if "KillOneLeg" in StrategyConfigIn["Strategy"]:
            shDF = pd.DataFrame(0, index=self.tenorList, columns=[str(x) for x in range(0,100,10)])
            #shDF = pd.DataFrame(0, index=["1W"], columns=[str(x) for x in range(40,60,10)])
            pnlList = []
            for space in tqdm(shDF.index):
                for thr in shDF.columns:
                    OptionalityTradeDF = pd.read_sql("SELECT * FROM GGCDF_Trade_EURUSD_Curncy_1W_"+space+strID, self.workConn).set_index("date", drop=True)
                    OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)["EURUSD_Curncy_1W_"+space+strID]

                    First_StraddleDF = OptionalityTradeDF["CallsPrices"] + OptionalityTradeDF["PutsPrices"]
                    First_dTradeDF = pe.d(First_StraddleDF)
                    First_GGC_Generic_Trader = First_dTradeDF * OptionalityHitDF

                    OptionalityTradeDF["CallsPct"] = OptionalityTradeDF["CallsPrices"] / OptionalityTradeDF["StraddlePrices"]
                    OptionalityTradeDF["PutsPct"] = OptionalityTradeDF["PutsPrices"] / OptionalityTradeDF["StraddlePrices"]
                    ########################################################################################################################
                    OptionalityTradeDF.loc[OptionalityTradeDF["CallsPct"].shift() <= float(thr)/100, "CallsPrices"] = 0
                    OptionalityTradeDF.loc[OptionalityTradeDF["PutsPct"].shift() <= float(thr)/100, "PutsPrices"] = 0
                    ID_SubSpace = "Upper"
                    ########################################################################################################################
                    #OptionalityTradeDF.loc[OptionalityTradeDF["CallsPct"].shift() >= float(thr) / 100, "CallsPrices"] = 0
                    #OptionalityTradeDF.loc[OptionalityTradeDF["PutsPct"].shift() >= float(thr) / 100, "PutsPrices"] = 0
                    #ID_SubSpace = "Lower"
                    ########################################################################################################################
                    StraddleDF = OptionalityTradeDF["CallsPrices"] + OptionalityTradeDF["PutsPrices"]
                    dTradeDF = pe.d(StraddleDF)
                    GGC_Generic_Trader = dTradeDF * OptionalityHitDF
                    sh = np.sqrt(252) * pe.sharpe(GGC_Generic_Trader)
                    shDF.loc[space,thr] = sh

                    pnlList.append([space, thr, First_GGC_Generic_Trader, GGC_Generic_Trader])
            shDF.to_sql("sh_KillOneLeg_"+ID_SubSpace, self.workConn, if_exists='replace')

            for item in pnlList:
                if (item[0] == "1W")&(item[1] == "50"):
                    print(np.sqrt(252) * pe.sharpe(pd.concat([item[2],item[3]],axis=1).sort_index()))
                    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
                    pe.cs(item[2]).plot(ax=ax[0])
                    pe.cs(item[3]).plot(ax=ax[1])
                    plt.show()
        elif StrategyConfigIn["Strategy"] == "Momentum_EMA":
            space = "1W"
            OptionalityTradeDF = pd.read_sql("SELECT * FROM GGCDF_Trade_EURUSD_Curncy_1W_" + space + strID,self.workConn).set_index("date", drop=True)
            OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)["EURUSD_Curncy_1W_" + space + strID]
            StraddleDF = OptionalityTradeDF["CallsPrices"] + OptionalityTradeDF["PutsPrices"]
            dTradeDF = pe.d(StraddleDF)
            GGC_Generic_Trader = pd.DataFrame(dTradeDF * OptionalityHitDF, columns=["Generic"])
            sh = np.sqrt(252) * pe.sharpe(GGC_Generic_Trader)
            print(sh)
            #sig = pe.sign(pe.ema(GGC_Generic_Trader,nperiods=25))
            sig = pe.sbb(GGC_Generic_Trader, nperiods=5, no_of_std=0.5)
            print(sig)
            sig[sig <= 0] = 0
            pnl_Mom = pe.S(sig) * GGC_Generic_Trader
            sh_Mom = np.sqrt(252) * pe.sharpe(pnl_Mom)
            print(sh_Mom)

            fig,ax=plt.subplots(nrows=2, ncols=1, sharex=True)
            pe.cs(GGC_Generic_Trader).plot(ax=ax[0])
            pe.cs(pnl_Mom).plot(ax=ax[1])
            plt.show()
        elif StrategyConfigIn["Strategy"] == "IV_HV":
            pair = "EURUSD"
            tnr0 = "1W"
            OptionalityTradeDF0 = pd.read_sql("SELECT * FROM GGCDF_Trade_" + pair + "_Curncy_1W_" + tnr0 + strID,self.workConn).set_index("date", drop=True)
            HistVol0 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF0[pair + " Curncy"]), 'Vol', nIn=7) * 100).bfill()
            IV0 = OptionalityTradeDF0[pair + "V1" + tnr0[1] + " Curncy"]
            IV_HV = pd.concat([HistVol0, IV0], axis=1)
            IV_HV.columns = ["HV", "IV"]
            IV_HV_Spread = (IV_HV["HV"] - IV_HV["IV"]).fillna(0)
            IV_HV_Spread.name = pair + "_" + tnr0

            fig,ax=plt.subplots(nrows=3, ncols=1, sharex=True)
            IV0.plot(ax=ax[0])
            HistVol0.plot(ax=ax[1])
            IV_HV_Spread.plot(ax=ax[2])
            plt.show()
        elif StrategyConfigIn["Strategy"] == "Plot_IV_HV_Spreads":
            DataDeck_TradingAssets = pd.read_sql("SELECT * FROM DataDeck_TradingAssets",self.workConn).set_index("date", drop=True)
            DataDeck_AtTheMoneyVols = pd.read_sql("SELECT * FROM DataDeck_AtTheMoneyVols",self.workConn).set_index("date", drop=True)

            for pair in ["EURUSD"]:
                PairData = DataDeck_TradingAssets[[x for x in DataDeck_TradingAssets if pair in x]]
                PairVolData = DataDeck_AtTheMoneyVols[[x for x in DataDeck_AtTheMoneyVols if pair in x]]
                V_List = []
                tenorList = ["1W", "2W", "1M", "3M", "6M", "9M"]
                for tnr0 in tenorList:
                    ###########################################################################################################################
                    if "W" in tnr0:
                        HistVol = np.sqrt(252) * (pe.rollStatistics(pe.dlog(PairData[pair + " Curncy"]), 'Vol', nIn=7*int(tnr0.replace("W",""))) * 100).bfill()
                    elif "M" in tnr0:
                        HistVol = np.sqrt(252) * (pe.rollStatistics(pe.dlog(PairData[pair + " Curncy"]), 'Vol', nIn=30*int(tnr0.replace("M",""))) * 100).bfill()
                    HistVol.name = pair + "HV" + tnr0 + " Curncy"
                    ###########################################################################################################################
                    V_List.append(HistVol)
                    V_List.append(PairVolData[pair + "V" + tnr0 + " Curncy"])
                    ###########################################################################################################################
                    spread = PairVolData[pair + "V" + tnr0 + " Curncy"] - HistVol
                    spread.name = pair+"_SpreadIvHv_"+tnr0
                    V_List.append(spread)
                    ###########################################################################################################################
                    if tnr0 == "1W":
                        bbSettings = {"nperiods":7, "no_of_std":1.5}
                    elif tnr0 == "2W":
                        bbSettings = {"nperiods":14, "no_of_std":1.5}
                    elif tnr0 == "1M":
                        bbSettings = {"nperiods":30, "no_of_std":1.5}
                    elif tnr0 == "3M":
                        bbSettings = {"nperiods":3*30, "no_of_std":1.5}
                    elif tnr0 == "6M":
                        bbSettings = {"nperiods":6*30, "no_of_std":1.5}
                    elif tnr0 == "9M":
                        bbSettings = {"nperiods":9*30, "no_of_std":1.5}
                    spreadBB = pe.bb(spread,nperiods=bbSettings['nperiods'],no_of_std=bbSettings['no_of_std'])
                    spreadBB = spreadBB.drop(spreadBB.columns[0],axis=1)
                    spreadBB.columns = [pair+"_SpreadIvHvBBupper_"+tnr0,pair+"_SpreadIvHvBBmiddle_"+tnr0,pair+"_SpreadIvHvBBlower_"+tnr0]
                    V_List.append(spreadBB)
                ###########################################################################################################################
                V_DF_0 = pd.concat(V_List, axis=1).sort_index()
                V_DF_EMA500 = pe.ema(V_DF_0, nperiods=500)
                V_DF_EMA500.columns = [x+"_EMA500" for x in V_DF_EMA500.columns]
                ###########################################################################################################################
                V_DF = pd.concat([V_DF_0, V_DF_EMA500],axis=1).sort_index()
                ###########################################################################################################################
                for tnrPlot in tenorList:
                    PlotDF = V_DF[[x for x in V_DF.columns if ("_"+tnrPlot in x)&("_EMA" not in x)&(("_SpreadIvHv_" in x)|("_SpreadIvHvBB" in x))]]
                    ###########################################################################################################################
                    if tnrPlot in ["3M", "6M", "9M"]:
                        PlotDF = PlotDF.loc["2010-01-01 00:00:00":,:]
                    elif tnrPlot in ["1W", "2W", "1M"]:
                        PlotDF = PlotDF.loc["2022-01-01 00:00:00":, :]
                    ###########################################################################################################################
                    fig, ax = plt.subplots(nrows=3,ncols=1,sharex=True)
                    PlotDF.plot(ax=ax[0], title="Spread")
                    ax[0].legend(loc="center left")
                    V_DF.loc[PlotDF.index, [pair + "V" + PlotDF.columns[0].split("_")[-1] + " Curncy",pair + "V" + PlotDF.columns[0].split("_")[-1] + " Curncy_EMA500"]].plot(ax=ax[1], title="IV")
                    ax[1].legend(loc="center left")
                    V_DF.loc[PlotDF.index, [pair + "HV" + PlotDF.columns[0].split("_")[-1] + " Curncy", pair + "HV" + PlotDF.columns[0].split("_")[-1] + " Curncy_EMA500"]].plot(ax=ax[2], title="HV")
                    ax[2].legend(loc="center left")
                    ###########################################################################################################################
                    plt.savefig(self.GreenBoxFolder+'IV_HV_Spreads_Plots/IV_HV_Spread_'+tnrPlot+'.jpg',dpi=100, figsize=(15, 10))
        elif StrategyConfigIn["Strategy"] == "GG0":
            pair = "EURUSD"
            #pair = "GBPUSD"
            #pair = "EURGBP"
            #pair = "USDCAD"

            tnr0 = "1W"
            #tnr0 = "2W"
            #tnr1 = "2W"
            tnr1 = "1M"
            deltaHedgeMul = [1, 1]

            #TCmethod = "VegaCharge"
            TCmethod = "StandardBidAsk"
            if TCmethod == "VegaCharge":
                vegaChargeMul = [0.1, 0.1]
            elif TCmethod == "StandardBidAsk":
                BidAskChargeMul = [0.02, 0.02]
            "LEG 1"
            OptionalityTradeDF0 = pd.read_sql("SELECT * FROM GGCDF_Trade_"+pair+"_Curncy_1W_"+tnr0 + strID,self.workConn).set_index("date", drop=True)
            if "M" in tnr0:
                HistVol0 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF0[pair + " Curncy"]), 'Vol', nIn=30)* 100).bfill()
            elif "W" in tnr0:
                HistVol0 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF0[pair + " Curncy"]), 'Vol', nIn=7)* 100).bfill()
            IV0 = OptionalityTradeDF0[pair+"V1"+tnr0[1]+" Curncy"]
            IV_HistVol_Spread0 = IV0-HistVol0

            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            pd.concat([HistVol0, IV0], axis=1).plot(ax=ax[0])#.tail(500)
            (IV0-HistVol0).plot(ax=ax[1])#.tail(500)
            (OptionalityTradeDF0["VegaCalls"]+OptionalityTradeDF0["VegaPuts"]).plot(ax=ax[2])#.tail(500)
            #plt.show()
            totalDelta0 = OptionalityTradeDF0["DeltaCalls"]+OptionalityTradeDF0["DeltaPuts"]
            OptionalityHitDF0 = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)[pair+"_Curncy_1W_" + tnr0 + strID]
            StraddleDF0 = OptionalityTradeDF0["CallsPrices"] + OptionalityTradeDF0["PutsPrices"]
            dTradeDF0 = pe.d(StraddleDF0)
            GGC_Generic_Trader0 = dTradeDF0 * OptionalityHitDF0
            TotalCost0 = pd.Series(0, index=GGC_Generic_Trader0.index)
            if TCmethod == "VegaCharge":
                TotalCost0.loc[OptionalityHitDF0.loc[OptionalityHitDF0 == 0].index] = vegaChargeMul[0] * (OptionalityTradeDF0["VegaCalls"] + OptionalityTradeDF0["VegaPuts"]).abs()
            elif TCmethod == "StandardBidAsk":
                TotalCost0.loc[OptionalityHitDF0.loc[OptionalityHitDF0 == 0].index] = BidAskChargeMul[0] * (OptionalityTradeDF0["CallsPrices"] + OptionalityTradeDF0["PutsPrices"]).abs()
            "LEG 2"
            OptionalityTradeDF1 = pd.read_sql("SELECT * FROM GGCDF_Trade_"+pair+"_Curncy_1W_"+tnr1 + strID,self.workConn).set_index("date", drop=True)
            if "M" in tnr1:
                HistVol1 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF1[pair + " Curncy"]), 'Vol', nIn=25) * 100).bfill()
            elif "W" in tnr1:
                HistVol1 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF1[pair + " Curncy"]), 'Vol', nIn=5) * 100).bfill()
            IV1 = OptionalityTradeDF1[pair+"V1"+tnr1[1]+" Curncy"]
            IV_HistVol_Spread1 = IV1 - HistVol1

            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            pd.concat([HistVol1, IV1], axis=1).plot(ax=ax[0])#.tail(500)
            (IV1-HistVol1).plot(ax=ax[1])#.tail(500)
            (OptionalityTradeDF1["VegaCalls"] + OptionalityTradeDF1["VegaPuts"]).plot(ax=ax[2])#.tail(500)

            totalDelta1 = OptionalityTradeDF1["DeltaCalls"] + OptionalityTradeDF1["DeltaPuts"]
            OptionalityHitDF1 = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)[pair+"_Curncy_1W_" + tnr1 + strID]
            StraddleDF1 = OptionalityTradeDF1["CallsPrices"] + OptionalityTradeDF1["PutsPrices"]
            dTradeDF1 = pe.d(StraddleDF1)
            GGC_Generic_Trader1 = dTradeDF1 * OptionalityHitDF1
            TotalCost1 = pd.Series(0, index=GGC_Generic_Trader1.index)
            if TCmethod == "VegaCharge":
                TotalCost1.loc[OptionalityHitDF1.loc[OptionalityHitDF1 == 0].index] = vegaChargeMul[1] * (OptionalityTradeDF1["VegaCalls"] + OptionalityTradeDF1["VegaPuts"]).abs()
            elif TCmethod == "StandardBidAsk":
                TotalCost1.loc[OptionalityHitDF1.loc[OptionalityHitDF1 == 0].index] = BidAskChargeMul[1] * (OptionalityTradeDF1["CallsPrices"] + OptionalityTradeDF1["PutsPrices"]).abs()

            deltaPnl0 = ((-1) * deltaHedgeMul[0] * pe.S(totalDelta0, nperiods=1) * pe.d(OptionalityTradeDF0[pair+" Curncy"])).fillna(0)
            deltaPnl1 = ((-1) * deltaHedgeMul[1] * pe.S(totalDelta1, nperiods=1) * pe.d(OptionalityTradeDF1[pair+" Curncy"])).fillna(0)
            deltaPnl = pd.concat([deltaPnl0, deltaPnl1], axis=1)
            deltaPnl.columns = [tnr0, tnr1]
            totalDeltaPnL = pe.rs(deltaPnl)

            checkVols = pd.concat([GGC_Generic_Trader0, GGC_Generic_Trader1, deltaPnl0, deltaPnl1],axis=1)/ self.AUM
            print(np.sqrt(252)*checkVols.std())

            sig_IV_HistVol_Spread0 = pe.sign(IV_HistVol_Spread0)
            sig_IV_HistVol_Spread0[sig_IV_HistVol_Spread0 < 0] = 0
            sig_IV_HistVol_Spread1 = pe.sign(IV_HistVol_Spread1)
            sig_IV_HistVol_Spread1[sig_IV_HistVol_Spread1 < 0] = 0

            Leg1_DailyDeltaHedged = (GGC_Generic_Trader0 + deltaPnl0) #* sig_IV_HistVol_Spread0.abs()
            Leg2_DailyDeltaHedged = (GGC_Generic_Trader1 + deltaPnl1) #* sig_IV_HistVol_Spread1.abs()
            print(np.sqrt(252)*pe.sharpe(Leg1_DailyDeltaHedged))
            print(np.sqrt(252)*pe.sharpe(Leg2_DailyDeltaHedged))
            figDH, axDH = plt.subplots(nrows=2,ncols=1, sharex=True)
            pe.cs(Leg1_DailyDeltaHedged).plot(ax=axDH[0])
            pe.cs(Leg2_DailyDeltaHedged).plot(ax=axDH[1])
            plt.show()

            totalDF = pd.concat([GGC_Generic_Trader0, (-1) * GGC_Generic_Trader1], axis=1)
            totalDF.columns = [tnr0, tnr1]
            #totalDF["TOTAL"] = pe.rs(totalDF)
            totalDF["TOTAL"] = totalDF[tnr0] + totalDF[tnr1] + totalDeltaPnL
            shBefore = np.sqrt(252) * pe.sharpe(totalDF)
            print("%%%%%% SHARPE (GROSS) %%%%%%")
            print(shBefore)
            totalDF[tnr0+"_Net"] = GGC_Generic_Trader0 - TotalCost0
            totalDF[tnr1+"_Net"] = (-1) * GGC_Generic_Trader1 - TotalCost1
            totalDF["TOTAL_Net"] = totalDF[tnr0+"_Net"] + totalDF[tnr1+"_Net"] + totalDeltaPnL
            sh = np.sqrt(252) * pe.sharpe(totalDF[[tnr0+"_Net", tnr1+"_Net", "TOTAL_Net"]])
            print("%%%%%%% SHARPE (NET) %%%%%")
            print(sh)

            fig, ax = plt.subplots(nrows=4, ncols=1)
            pe.cs(pd.concat([totalDF[tnr0], totalDF[tnr0+"_Net"]], axis=1).tail(500)).plot(ax=ax[0])
            pe.cs(pd.concat([totalDF[tnr1], totalDF[tnr1+"_Net"]], axis=1).tail(500)).plot(ax=ax[1])
            pe.cs(deltaPnl).plot(ax=ax[2])
            pe.cs(pd.concat([totalDF["TOTAL"], totalDF["TOTAL_Net"]], axis=1)).plot(ax=ax[3])
            plt.show()
        elif StrategyConfigIn["Strategy"] == "Plotter":
            OptionalityTradeDF = pd.read_sql("SELECT * FROM OptionalityTradeDF", self.workConn).set_index("date",drop=True)
            OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)
            TCA_DF = pd.read_sql('SELECT * FROM TCA_DF', self.workConn).set_index("date", drop=True)
            ####################################################################################################################################
            PnlDF = (OptionalityHitDF * pe.d(OptionalityTradeDF)) - TCA_DF
            ####################################################################################################################################
            sh = np.sqrt(252) * pe.sharpe(PnlDF).sort_values()
            print(sh)
            ####################################################################################################################################
            #fig, ax = plt.subplots(nrows=3,ncols=1,sharex=True)
            pe.cs(PnlDF).plot()
            plt.show()

#//////////////////////////////////// DATA PREPARE / GET ////////////////////////////////////////////
obj = VolatilityScanner("VolatilityScanner", "VolatilityScanner.db")
#obj.SetupRatesMatrix() #"Run once"
#obj.SetupFWDMatrix()
#obj.CreateVolatilityGenerics() #"Run once"
#//////////////////////////////////// GEKKO CHOOSES //////////////////////////////////////////////////
#"""#
obj.getData()
#RunMode = "forceReRunAll"
RunMode = "update"
obj.RunBuilder("Run", RunMode) #Optionality_UpdateStatus="",or "update"
obj.RunBuilder("Trade", RunMode, skipReadCalc="NO", DailyDeltaHedgeFlag="NO", TCA_Mode="PriceCharge", ECO_Relevance_Filter=["NO",90]) #skipReadCalc="YES", "NO |||| TCA_Mode = None, VegaCharge, PriceCharge
#obj.RunBuilder("Trade", RunMode, skipReadCalc="NO", DailyDeltaHedgeFlag="YES", TCA_Mode="PriceCharge") #skipReadCalc="YES", "NO
#obj.StrategyTester({"Strategy": "Plot_IV_HV_Spreads", "StrategyID":""})
#//////////////////////////////////// RUN LATEST HFT //////////////////////////////////////////////////
#os.chdir("F:/Dealing/Panagiotis Papaioannou/MT5/HTML_Reports/LIVE/")
#import ERB_Py_Reporter
#"""
#//////////////////////////////////////////// NOTES /////////////////////////////////////////////////
#obj.StrategyTester({"Strategy": "Plotter", "StrategyID":""})
#############################################################################################
#obj.StrategyTester({"Strategy": "KillOneLeg", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "IV_HV", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "Momentum_EMA", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "GG0", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "GG0", "StrategyID":"_SMA_2STDs_Strangle_ITM_Native"})
