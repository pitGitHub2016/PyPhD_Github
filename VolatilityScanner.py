import time
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

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class VolatilityScanner:

    def __init__(self, sheet, DB):
        self.AlternativeStorageLocation = "C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/"
        DB = self.AlternativeStorageLocation + DB
        self.ExcelControlPanel = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="Control Panel").set_index("Parameter", drop=True)
        self.LiveStrategies = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="LiveStrategies")
        self.LiveStrategiesNames = list(self.LiveStrategies.columns)
        self.LiveStrategies_SubAllocations = pd.read_excel("VolatilityTrader_ExecutionTracker.xlsx", sheet_name="LiveStrategies_SubAllocations")
        self.VolPortfolioWeights = self.ExcelControlPanel.loc[self.LiveStrategiesNames]
        self.ExcelDF = pd.read_excel("AssetsDashboard.xlsx", sheet_name=sheet)
        self.Assets = self.ExcelDF["Assets"].dropna().tolist()
        self.tenorList = ["1W","2W","1M","2M","3M","6M","9M","1Y","2Y","3Y"]# ||||| "ON","1D","3W","2M","4M","5M","18M","2Y","3Y","4Y","5Y","6Y","7Y","10Y","15Y","20Y","25Y","30Y"
        #self.tenorList = ["1W"] #DEBUGGER
        self.DeltaSpanList = ["25"]#"5","10","35"
        self.workConn = sqlite3.connect(DB,detect_types=sqlite3.PARSE_DECLTYPES)
        self.PuntNotional = self.ExcelControlPanel.loc["Punt Notional", "Value"]
        self.AUM = self.ExcelControlPanel.loc["AUM", "Value"]
        self.LEVERAGE = self.ExcelControlPanel.loc["LEVERAGE", "Value"]
        self.WeeklyHorizon = self.ExcelControlPanel.loc["Weekly Horizon", "Value"]
        self.BiWeeklyHorizon = self.ExcelControlPanel.loc["BiWeekly Horizon", "Value"]
        self.MonthlyHorizon = self.ExcelControlPanel.loc["Monthly Horizon", "Value"]
        self.RatesSetup = 0
        self.FWDSetup = 0

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
                        self.Rates.loc[idx, c] = "EUSWA Curncy"
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
            startDate = '20000101'; self.updateStatus = 'fetchSinceInception'

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
        try:
            BaseDF['MarketFwd'] = ConfigIn["Pair"].split(" ")[0]+ConfigIn['Tenor'].replace("1Y","12M")+" Curncy"
            if "JPY" not in ConfigIn["Pair"]:
                BaseDF.loc[:,'MarketFwd'] = BaseDF.loc[:,ConfigIn["Pair"]] + BaseDF.loc[:,'MarketFwd'] / 10000
            else:
                BaseDF.loc[:,'MarketFwd'] = BaseDF.loc[:,ConfigIn["Pair"]] + BaseDF.loc[:,'MarketFwd'] / 100
        except Exception as e:
            pass
            #print(e)

        BaseDF["r1"] = self.GGCDF[self.Rates.loc[baseC, ConfigIn['Tenor']]]
        BaseDF["r2"] = self.GGCDF[self.Rates.loc[quoteC, ConfigIn['Tenor']]]

        "HITMAN HANDLER"
        if "W" in ConfigIn['Hitman']:
            toPeriod = 'W'
        elif "M" in ConfigIn['Hitman']:
            toPeriod = 'M'
        elif "Y" in ConfigIn['Hitman']:
            toPeriod = 'M'
            ConfigIn['Hitman'] = BBG_Tenor_Handle(ConfigIn['Hitman'], "FromBBGtoMonths")

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

        print("Building "+self.ID)
        print("self.Optionality_UpdateStatus = ", self.Optionality_UpdateStatus)

        self.TimersCols = ["index_col", "index_Dt"]
        self.OptionsCols = ["CallsPrices", "PutsPrices", "StraddlePrices",
                            "DeltaCalls", "GammaCalls", "ThetaCalls", "VegaCalls", "RhoCalls",
                            "DeltaPuts", "GammaPuts", "ThetaPuts", "VegaPuts", "RhoPuts"]

        if ConfigIn["HitmanStrike"] == "EMA_Tenor":
            HitBase = pe.sma(self.GGCDF[ConfigIn['Pair']], nperiods=T_Base).round(4)
        elif ConfigIn["HitmanStrike"] == "Latest_Spot":
            HitBase = self.GGCDF[ConfigIn['Pair']].round(4)
        elif "SMA_2STDs_Strangle" in ConfigIn["HitmanStrike"]:
            sma = pe.sma(self.GGCDF[ConfigIn['Pair']], nperiods=T_Base).round(4)
            RollStd = self.GGCDF[ConfigIn['Pair']].rolling(T_Base).std()
            HitBase = (sma + 2*RollStd).round(4).bfill().astype(str) + "," + (sma - 2*RollStd).round(4).bfill().astype(str)

        HitBase[HitBase == 0] = None
        HitBase = HitBase.ffill().bfill()
        HitBase = pd.DataFrame(HitBase[~HitBase.index.to_period(toPeriod).duplicated()].iloc[::int(ConfigIn['Hitman'].replace("W", "").replace("M", ""))])
        HitBase["HitFlag"] = 0
        StrikeColName = ConfigIn['Pair'] + "_" + ConfigIn['Hitman'] + "_" + ConfigIn['Tenor'] + "_" + " Strike"
        HitBase.columns = [StrikeColName, "HitFlag"]

        HitBase = pe.S(HitBase).bfill()

        if self.Optionality_UpdateStatus == 'update':
            self.GGCDF_Trade_Previous = pd.read_sql('SELECT * FROM GGCDF_Trade_'+self.ID, self.workConn)
            self.GGCDF_Trade_Previous = self.GGCDF_Trade_Previous.set_index(self.GGCDF_Trade_Previous.columns[0], drop=True)
            self.GGCDF_Trade_Previous = self.GGCDF_Trade_Previous.drop([x for x in self.GGCDF_Trade_Previous.columns if (x in BaseDF.columns)|(x in HitBase.columns)], axis=1)
            self.GGCDF_Trade = pd.concat([self.GGCDF_Trade_Previous, BaseDF, HitBase], axis=1)
            self.GGCDF_Trade = self.GGCDF_Trade[~self.GGCDF_Trade.index.duplicated(keep='last')].sort_index()
        else:
            self.GGCDF_Trade = pd.concat([BaseDF, HitBase], axis=1)
            self.GGCDF_Trade["T_Adj"] = 0
            self.GGCDF_Trade["fwd"] = None
            self.GGCDF_Trade["MaturityDate"] = None
            self.GGCDF_Trade[self.OptionsCols] = None

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
                if row["ThetaDecayDays"] == 0:
                    T = T_Base
                    self.GGCDF_Trade.loc[idx, "MaturityDate"] = self.GGCDF_Trade.loc[idx, "index_col"] + timedelta(days=T)
                    if self.GGCDF_Trade.loc[idx, "MaturityDate"].dayofweek >= 5:
                        self.GGCDF_Trade.loc[idx, "MaturityDate"] -= timedelta(days=self.GGCDF_Trade.loc[idx, "MaturityDate"].dayofweek - 4)
                        T = (self.GGCDF_Trade.loc[idx, "MaturityDate"]-self.GGCDF_Trade.loc[idx, "index_col"]).days
                else:
                    T = T - row["ThetaDecayDays"]
                    self.GGCDF_Trade.loc[idx, "MaturityDate"] = self.GGCDF_Trade.loc[idx, "index_col"] + timedelta(days=T)
                self.GGCDF_Trade.loc[idx, "T_Adj"] = T
                r1 = row["r1"] / 100
                r2 = row["r2"] / 100
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
                sigma = row[IVcol].values[0] / 100
                if ConfigIn["FwdCalculator"] == "Market":
                    fwd = self.GGCDF_Trade.loc[idx, "MarketFwd"]
                else:
                    fwd = row[ConfigIn["Pair"]] * math.exp((r1 - r2) * (T / 365))
                self.GGCDF_Trade.loc[idx, "fwd"] = fwd
                # ////////////////////////////////////////////////////////////////////////////////////////////////////
                self.GGCDF_Trade.loc[idx,"CallsPrices"] = pe.black_scholes(fwd, Kc, T / 365, r2, r1 - r2, sigma, "call")
                self.GGCDF_Trade.loc[idx,"PutsPrices"] = pe.black_scholes(fwd, Kp, T / 365, r2, r1 - r2, sigma, "put")
                self.GGCDF_Trade.loc[idx,"StraddlePrices"] = self.GGCDF_Trade.loc[idx,"CallsPrices"] + self.GGCDF_Trade.loc[idx,"PutsPrices"]
                # ////////////////////////////////////////////////////////////////////////////////////////////////////
                delta_call, gamma_call, theta_call, vega_call, rho_call = pe.black_scholes_greeks(fwd, Kc, T / 365, r2,
                                                                                               r1 - r2, sigma, "call")
                self.GGCDF_Trade.loc[idx, "DeltaCalls"] = delta_call
                self.GGCDF_Trade.loc[idx, "GammaCalls"] = gamma_call
                self.GGCDF_Trade.loc[idx, "ThetaCalls"] = theta_call
                self.GGCDF_Trade.loc[idx, "VegaCalls"] = vega_call
                self.GGCDF_Trade.loc[idx, "RhoCalls"] = rho_call
                # ////////////////////////////////////////////////////////////////////////////////////////////////////
                delta_put, gamma_put, theta_put, vega_put, rho_put = pe.black_scholes_greeks(fwd, Kp, T / 365, r2, r1 - r2,
                                                                                          sigma, "put")
                self.GGCDF_Trade.loc[idx, "DeltaPuts"] = delta_put
                self.GGCDF_Trade.loc[idx, "GammaPuts"] = gamma_put
                self.GGCDF_Trade.loc[idx, "ThetaPuts"] = theta_put
                self.GGCDF_Trade.loc[idx, "VegaPuts"] = vega_put
                self.GGCDF_Trade.loc[idx, "RhoPuts"] = rho_put

        self.GGCDF_Trade.loc[processIndexes, self.OptionsCols] *= self.PuntNotional
        self.GGCDF_Trade["MaturityDate"] = pd.to_datetime(self.GGCDF_Trade["MaturityDate"])

        "Fixing the Columns Order"
        self.GGCDF_Trade = self.GGCDF_Trade[[ConfigIn["Pair"], 'fwd', 'T_Adj', 'MaturityDate', StrikeColName,
                                            'CallsPrices', 'PutsPrices', 'StraddlePrices',
                                            'ThetaDecayDays','HitFlag', IVcol[0], 'r1', 'r2',
                                            'DeltaCalls', 'GammaCalls', 'ThetaCalls', 'VegaCalls', 'RhoCalls',
                                            'DeltaPuts', 'GammaPuts', 'ThetaPuts', 'VegaPuts', 'RhoPuts',
                                            'index_col', 'index_Dt']]

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

        self.Optionality_UpdateStatus = Optionality_UpdateStatus

        if mode == "Run":
            NotYetProcessed = []
            for Pair in tqdm(["EURUSD Curncy", "GBPUSD Curncy", "USDCAD Curncy", "EURGBP Curncy"]):#["EURUSD Curncy"]:
                for Hitman in ["1W", "2W"]: #self.tenorList
                    for Tenor in self.tenorList: #["1W"], self.tenorList
                        for HitManStrikeMode in ["Latest_Spot", "SMA_2STDs_Strangle_OTM", "SMA_2STDs_Strangle_ITM"]: #
                            InputConfig = {"Pair": Pair, "Hitman": Hitman, "Tenor": Tenor, "HitmanStrike": HitManStrikeMode, "FwdCalculator": "Native"}  # FwdCalculator=Market,Native
                            InputConfig['ID'] = '_'.join(InputConfig.values()).replace(" ", "_")
                            if self.Optionality_UpdateStatus == "":
                                try:
                                    df = pd.read_sql('SELECT * FROM GGCDF_Trade_'+InputConfig['ID'], self.workConn)
                                except Exception as e:
                                    print(e)
                                    NotYetProcessed.append(InputConfig)
                            else:
                                NotYetProcessed.append(InputConfig)

            print("len(NotYetProcessed) = ", len(NotYetProcessed))

            StraddlesList = []
            HitFlagList = []
            LatestTradeInfoList = []
            for subInputConfig in NotYetProcessed:
                VolatilityScanner.CreateOptionsGenerics(self, subInputConfig, 0)
                self.GGCDF_Trade = self.GGCDF_Trade[self.GGCDF_Trade["T_Adj"] >= 0]
                subStraddle = self.GGCDF_Trade["StraddlePrices"]
                subStraddle.name = self.ID
                subHitFlag = self.GGCDF_Trade["HitFlag"]
                subHitFlag.name = self.ID
                StraddlesList.append(subStraddle)
                HitFlagList.append(subHitFlag)
                LatestTradeInfoList.append(self.LatestTradeInfo)

            OptionalityTradeDF = pd.concat(StraddlesList, axis=1).sort_index()
            OptionalityHitDF = pd.concat(HitFlagList, axis=1).sort_index()

            LatestTradeInfoDF = pd.DataFrame(LatestTradeInfoList, columns=["Last Date", "OptionID", "RefSpot", "Strike",
                                                                           "DaysToMaturity", "MaturityDate",
                                                                           "CallsPrices", "PutsPrices", "Straddles",
                                                                           "TotalDelta","TotalGamma","TotalTheta","TotalVega"]).set_index("OptionID", drop=True)

            OptionalityTradeDF.to_sql("OptionalityTradeDF", self.workConn, if_exists='replace')
            OptionalityHitDF.to_sql("OptionalityHitDF", self.workConn, if_exists='replace')
            LatestTradeInfoDF.to_sql("LatestTradeInfoDF", self.workConn, if_exists='replace')

        elif mode == "Read":

            OptionalityTradeDF = pd.read_sql('SELECT * FROM OptionalityTradeDF', self.workConn).set_index("date", drop=True)
            OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)
            LatestTradeInfoDF = pd.read_sql('SELECT * FROM LatestTradeInfoDF', self.workConn).set_index("OptionID", drop=True)

            dTradeDF = pe.d(OptionalityTradeDF)
            dTradeDF.to_sql("dTradeDF", self.workConn, if_exists='replace')

            GGC_Generic_Trader = dTradeDF * OptionalityHitDF
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

            VolStratsData = []
            for StratSpace in ["SingleGenerics", "RVsSpreadsGenerics", "RVsBasketsGenerics"]:
                StratPortfolioWeight = self.VolPortfolioWeights.loc[StratSpace].values[0]
                StratContributions = self.LiveStrategies[StratSpace].dropna().tolist()
                StratContributions_SubAllocations = self.LiveStrategies_SubAllocations[StratSpace].fillna(0).tolist()
                StratContributionsRets = GGC_Generic_Trader[StratContributions].fillna(0)
                print("Active Optionality : ")
                k = 0
                for c in StratContributionsRets.columns:
                    StratContributionsRets[c] *= StratContributions_SubAllocations[k]
                    print("k=", k, ", c = ", c, ", ", StratContributions_SubAllocations[k])
                    k += 1
                subDF = StratPortfolioWeight*pe.rs(StratContributionsRets)
                subDF.name = StratSpace
                VolStratsData.append(subDF)
            ####################################################################################################
            VolatilityScannerPnL = pd.concat(VolStratsData, axis=1)
            VolatilityScannerPnL.to_sql("VolatilityScannerPnL", self.workConn, if_exists='replace')
            ####################################################################################################
            VolatilityScannerLive = pe.rs(VolatilityScannerPnL)
            VolatilityScannerLive *= self.LEVERAGE # LEVERAGE
            VolatilityScannerLive /= self.AUM # RETURNS
            shVolatilityScannerLive = np.sqrt(252) * pe.sharpe(VolatilityScannerLive)
            print("shVolatilityScannerLive = ", shVolatilityScannerLive)
            ####################################################################################################
            VolatilityScannerLive = pd.DataFrame(VolatilityScannerLive, columns=["VolatilityScannerLive"])
            VolatilityScannerLive.index = pd.to_datetime(VolatilityScannerLive.index)
            VolatilityScannerLive.to_sql("VolatilityScannerLive", self.workConn, if_exists='replace')
            qs.reports.html(VolatilityScannerLive["VolatilityScannerLive"], compounded=False, title="VolScanner",
                            output="StrategiesFactSheets/VolatilityScannerPnL.html")

            # MERGE EVERYTHING WITH THE MAIN CTA
            TOTAL_CTA = pd.read_sql('SELECT * FROM TotalPortfolioReturns', sqlite3.connect("StrategiesAggregator.db"))
            TOTAL_CTA = TOTAL_CTA.set_index(TOTAL_CTA.columns[0],drop=True)
            TOTAL_CTA.index.names = ['date']
            TOTAL_CTA.index = pd.to_datetime(TOTAL_CTA.index)
            TOTAL_CTA.columns = ["CTA"]
            TOTAL_SYSTEMATIC = pd.concat([TOTAL_CTA, VolatilityScannerLive], axis=1).sort_index().fillna(0)
            TOTAL_SYSTEMATIC["TOTAL"] = pe.rs(TOTAL_SYSTEMATIC)
            print("Contributions Vols ", TOTAL_SYSTEMATIC.std() * np.sqrt(252))
            print("Contributions Sharpes ", pe.sharpe(TOTAL_SYSTEMATIC) * np.sqrt(252))
            TOTAL_SYSTEMATIC.to_sql("TOTAL_SYSTEMATIC", self.workConn, if_exists='replace')
            qs.reports.html(TOTAL_SYSTEMATIC["TOTAL"], compounded=False, title="TOTAL SYSTEMATIC", output="StrategiesFactSheets/TOTAL_SYSTEMATIC.html")

            #plt.rcParams["figure.figsize"] = (20,20)
            #fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
            #pe.cs(VolatilityScannerLive).plot(ax=ax[0])#"1M","2M","3M","1Y","2Y","3Y"
            pe.cs(VolatilityScannerLive).plot()#"1M","2M","3M","1Y","2Y","3Y"
            plt.savefig('VolatilityScannerPnL.jpg')

    def StrategyTester(self, StrategyConfigIn):

        strID = StrategyConfigIn["StrategyID"]
        if StrategyConfigIn["Strategy"] == "KillTheWinners":
            shDF = pd.DataFrame(0, index=self.tenorList, columns=[str(x) for x in range(0,100,10)])
            for space in tqdm(shDF.index):
                for thr in shDF.columns:
                    OptionalityTradeDF = pd.read_sql("SELECT * FROM GGCDF_Trade_EURUSD_Curncy_1W_"+space+strID, self.workConn).set_index("date", drop=True)
                    OptionalityHitDF = pd.read_sql('SELECT * FROM OptionalityHitDF', self.workConn).set_index("date", drop=True)["EURUSD_Curncy_1W_"+space+strID]

                    OptionalityTradeDF["CallsPct"] = OptionalityTradeDF["CallsPrices"] / OptionalityTradeDF["StraddlePrices"]
                    OptionalityTradeDF["PutsPct"] = OptionalityTradeDF["PutsPrices"] / OptionalityTradeDF["StraddlePrices"]

                    #print(OptionalityTradeDF)
                    #time.sleep(30000)
                    OptionalityTradeDF.loc[OptionalityTradeDF["CallsPct"].shift() <= float(thr)/100, "CallsPrices"] = 0
                    OptionalityTradeDF.loc[OptionalityTradeDF["PutsPct"].shift() <= float(thr)/100, "PutsPrices"] = 0

                    StraddleDF = OptionalityTradeDF["CallsPrices"] + OptionalityTradeDF["PutsPrices"]

                    dTradeDF = pe.d(StraddleDF)
                    GGC_Generic_Trader = dTradeDF * OptionalityHitDF
                    sh = np.sqrt(252) * pe.sharpe(GGC_Generic_Trader)
                    shDF.loc[space,thr] = sh

            shDF.to_sql("sh_KillTheWinners", self.workConn, if_exists='replace')
        elif StrategyConfigIn["Strategy"] == "Plot_IV_HV_Spreads":

            IV_HV_Spreads_List = []
            Total_Gammas_List = []
            Total_Thetas_List = []
            Total_Vegas_List = []
            E_PnL_List = []
            for pair in ["EURUSD", "GBPUSD", "EURGBP"]:

                for tnr0 in ["1W", "1M"]:

                    tnr1 = "1M"

                    "LEG 1"
                    OptionalityTradeDF0 = pd.read_sql("SELECT * FROM GGCDF_Trade_" + pair + "_Curncy_1W_" + tnr0 + strID,self.workConn).set_index("date", drop=True)
                    TotalVega = OptionalityTradeDF0["VegaCalls"] + OptionalityTradeDF0["VegaPuts"]
                    TotalGamma = OptionalityTradeDF0["GammaCalls"] + OptionalityTradeDF0["GammaPuts"]
                    TotalTheta = OptionalityTradeDF0["ThetaCalls"] + OptionalityTradeDF0["ThetaPuts"]
                    DS = pe.d(OptionalityTradeDF0[pair + " Curncy"]).fillna(0)
                    if "M" in tnr0:
                        HistVol0 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF0[pair + " Curncy"]), 'Vol',nIn=30) * 100).bfill()
                    elif "W" in tnr0:
                        HistVol0 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF0[pair + " Curncy"]), 'Vol',nIn=7) * 100).bfill()
                    IV0 = OptionalityTradeDF0[pair + "V1" + tnr0[1] + " Curncy"]

                    IV_HV_Spread = np.sqrt(HistVol0) - np.sqrt(IV0.fillna(0))
                    IV_HV_Spread.name = pair+"_"+tnr0
                    IV_HV_Spreads_List.append(IV_HV_Spread)

                    TotalVega.name = pair + "_" + tnr0
                    Total_Vegas_List.append(TotalVega)
                    TotalGamma.name = pair + "_" + tnr0
                    Total_Gammas_List.append(TotalGamma)
                    TotalTheta.name = pair + "_" + tnr0
                    Total_Thetas_List.append(TotalTheta)

                    subPnL = (1/2) * TotalGamma * (DS.pow(DS)) + TotalTheta
                    #subPnL = TotalVega * IV_HV_Spread
                    subPnL.name = pair + "_" + tnr0
                    E_PnL_List.append(subPnL)

            IV_HV_Spreads_DF = pd.concat(IV_HV_Spreads_List, axis=1)
            IV_HV_Spreads_DF.to_sql("IV_HV_Spreads_DF", self.workConn, if_exists='replace')
            VolPnL = pd.read_sql("SELECT * FROM VolatilityScannerPnL", self.workConn).set_index("date", drop=True)
            VolPnL_vs_IV_HV_Spreads_DF = pd.concat([VolPnL, IV_HV_Spreads_DF], axis=1)
            VolPnL_vs_IV_HV_Spreads_DF.to_sql("VolPnL_vs_IV_HV_Spreads_DF", self.workConn, if_exists='replace')
            Total_Gammas_DF = pd.concat(Total_Gammas_List, axis=1).fillna(0)
            Total_Vegas_DF = pd.concat(Total_Vegas_List, axis=1).fillna(0)
            Total_Thetas_DF = pd.concat(Total_Vegas_List, axis=1).fillna(0)
            E_PnL_DF = pd.concat(E_PnL_List, axis=1)

            fig, ax = plt.subplots(nrows=2, ncols=1)
            #IV_HV_Spreads_DF.plot(ax=ax[0])
            #pe.cs(IV_HV_Spreads_DF).plot(ax=ax[1])
            IV_HV_Spreads_DF[[x for x in IV_HV_Spreads_DF.columns if "1W" in x]].tail(250).plot(ax=ax[0], title="IV_HV_Spreads")
            IV_HV_Spreads_DF[[x for x in IV_HV_Spreads_DF.columns if "1M" in x]].tail(250).plot(ax=ax[1])
            #pe.cs(IV_HV_Spreads_DF.tail(250)).plot(ax=ax[1])

            fig0, ax0 = plt.subplots(nrows=2, ncols=1)
            Total_Vegas_DF[[x for x in Total_Vegas_DF.columns if "1W" in x]].tail(250).plot(ax=ax0[0], title="Total_Vegas")
            Total_Vegas_DF[[x for x in Total_Vegas_DF.columns if "1M" in x]].tail(250).plot(ax=ax0[1])

            fig01, ax01 = plt.subplots(nrows=2, ncols=1)
            Total_Gammas_DF[[x for x in Total_Gammas_DF.columns if "1W" in x]].tail(250).plot(ax=ax01[0], title="Total_Gammas")
            Total_Gammas_DF[[x for x in Total_Gammas_DF.columns if "1M" in x]].tail(250).plot(ax=ax01[1])

            #fig02, ax02 = plt.subplots(nrows=2, ncols=1)
            #Total_Thetas_DF[[x for x in Total_Thetas_DF.columns if "1W" in x]].tail(250).plot(ax=ax02[0], title="Total_Thetas")
            #Total_Thetas_DF[[x for x in Total_Thetas_DF.columns if "1M" in x]].tail(250).plot(ax=ax02[1])

            #fig1, ax1 = plt.subplots(nrows=4, ncols=1)
            #E_PnL_DF[[x for x in E_PnL_DF.columns if "1W" in x]].tail(250).plot(ax=ax1[0], title="E_PnL")
            #pe.cs(E_PnL_DF[[x for x in E_PnL_DF.columns if "1W" in x]].tail(250)).plot(ax=ax1[1])
            #E_PnL_DF[[x for x in E_PnL_DF.columns if "1M" in x]].tail(250).plot(ax=ax1[2])
            #pe.cs(E_PnL_DF[[x for x in E_PnL_DF.columns if "1M" in x]].tail(250)).plot(ax=ax1[3])

            plt.show()

        elif StrategyConfigIn["Strategy"] == "GG0":
            pair = "EURUSD"
            #pair = "GBPUSD"
            #pair = "EURGBP"
            #pair = "USDCAD"

            tnr0 = "1W"
            #tnr0 = "2W"
            #tnr1 = "2W"
            tnr1 = "1M"
            deltaHedgeMul = [0.35, 0]

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
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            pd.concat([HistVol0, IV0], axis=1).tail(500).plot(ax=ax[0])
            (IV0-HistVol0).tail(500).plot(ax=ax[1])
            (OptionalityTradeDF0["VegaCalls"]+OptionalityTradeDF0["VegaPuts"]).tail(500).plot(ax=ax[2])
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
                HistVol1 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF1[pair + " Curncy"]), 'Vol', nIn=25)* 100).bfill()
            elif "W" in tnr1:
                HistVol1 = np.sqrt(252) * (pe.rollStatistics(pe.dlog(OptionalityTradeDF1[pair + " Curncy"]), 'Vol', nIn=5)* 100).bfill()
            IV1 = OptionalityTradeDF1[pair+"V1"+tnr1[1]+" Curncy"]
            fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
            pd.concat([HistVol1, IV1], axis=1).tail(500).plot(ax=ax[0])
            (IV1-HistVol1).tail(500).plot(ax=ax[1])
            (OptionalityTradeDF1["VegaCalls"] + OptionalityTradeDF1["VegaPuts"]).tail(500).plot(ax=ax[2])

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

#//////////////////////////////////// DATA PREPARE / GET ////////////////////////////////////////////
obj = VolatilityScanner("VolatilityScanner", "VolatilityScanner.db")
#obj.SetupRatesMatrix() #"Run once"
#obj.SetupFWDMatrix()
#obj.CreateVolatilityGenerics() #"Run once"
#//////////////////////////////////// GEKKO CHOOSES /////////////////////////////////////////////////
#"""
obj.getData()
#RunMode = ""
#RunMode = "forceReRunAll"
RunMode = "update"
obj.RunBuilder("Run", RunMode) #Optionality_UpdateStatus="",or "update"
obj.RunBuilder("Read", RunMode, skipReadCalc="NO") #skipReadCalc="YES", "NO
#"""
#//////////////////////////////////// META STRATEGIES /////////////////////////////////////////////////
#obj.StrategyTester({"Strategy": "KillTheWinners", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "GG0", "StrategyID":"_Latest_Spot_Native"})
obj.StrategyTester({"Strategy": "Plot_IV_HV_Spreads", "StrategyID":"_Latest_Spot_Native"})
#obj.StrategyTester({"Strategy": "GG0", "StrategyID":"_SMA_2STDs_Strangle_ITM_Native"})
#obj.StrategyTester({"Strategy": "GG0", "StrategyID":"_SMA_2STDs_Strangle_OTM_Native"})
