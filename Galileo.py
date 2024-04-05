import time, pickle
try:
    from os.path import dirname, basename, isfile, join
    import glob, os, sys, pdblp,csv
    import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt, matplotlib as mpl
    import streamlit as st
    import quantstats as qs
    from itertools import permutations
    from tqdm import tqdm
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')

    from pyerb import pyerb as pe
    from pyerbML import ML, ManSee
    from ta import add_all_ta_features
    from ta.utils import dropna

    import warnings
    warnings.filterwarnings("ignore")
except Exception as e:
    print(e)
    time.sleep(10)

MaturitiesList = ["1M","1Y","2Y"] #, "10Y"

def DecisionTrees_RV_Runner(argDict):
    AllDF = pe.fd(
        pd.concat([argDict["DT_RetsDF"], argDict["IndicatorsDF"][argDict["MacroSplitter"]]], axis=1).sort_index())
    AllDF = AllDF.loc[:, ~AllDF.columns.duplicated()].copy()
    cc = [x for x in list(permutations(AllDF.columns, 2)) if x[0] == argDict["MacroSplitter"]]
    print(AllDF.columns)
    print(cc)

    if argDict["Mode"] == "Update":
        startLevel = AllDF.shape[0] - argDict['DT_Update_Depth']
    else:
        startLevel = argDict["st"]
        # startLevel = AllDF.shape[0] - 10 ### THIS IS A DEBUGGER !!!

    DecisionTrees_RV = pd.concat(
        [ML.RollDecisionTree(ML.binarize(AllDF, targetColumns=[c[1]]), X=[c[0]], Y=[c[1]], RollMode=argDict["RollMode"],
                             st=startLevel) for
         c in tqdm(cc)], axis=1, keys=cc)
    DecisionTrees_RV.columns = DecisionTrees_RV.columns.map(','.join)
    DecisionTrees_RV.columns = [x.replace("_TreeThreshold", "") for x in DecisionTrees_RV.columns]
    DecisionTrees_RV.index.names = ['date']
    ###########################################################################################
    if argDict["Mode"] == "Update":
        # try:
        DecisionTrees_RV = pe.updatePickleDF(DecisionTrees_RV, argDict["WriteFilePath"])
        # except Exception as e:
        #    print(e)
    ###########################################################################################
    DTPickle = open(argDict["WriteFilePath"], 'wb')
    pickle.dump(DecisionTrees_RV, DTPickle)
    DTPickle.close()
    ###########################################################################################
    return "Success"

def GalileoMapper(GalileoMapperDF, fromSpace, ToSpace, tickerIn):
    GalileoMapperDF = GalileoMapperDF.set_index(fromSpace,drop=True)
    return GalileoMapperDF.loc[tickerIn,ToSpace]

class Galileo:

    def __init__(self):
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.workConn = sqlite3.connect("Galileo.db")
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"
        self.AssetsDashboardExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
        self.DataDeckConn = sqlite3.connect("DataDeck.db")
        self.AUM = 1000000
        self.ECO_IndicatorsConn = sqlite3.connect("ECO_Indicators.db")
        self.ECO_Indicators_Main_Excel = "Bloomberg_Key_ECO_Indicators.xlsx"
        self.AllDF = pd.read_sql('SELECT * FROM DataDeck', self.DataDeckConn).set_index('date', drop=True)
        self.AllRetsDF = pd.read_sql('SELECT * FROM rets', self.DataDeckConn).set_index('date', drop=True)
        self.IndicatorsDF_Raw = pd.read_sql('SELECT * FROM IndicatorsDeck_Raw',self.DataDeckConn).set_index('date', drop=True)
        self.IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck',self.DataDeckConn).set_index('date', drop=True)
        self.IndicatorsData = pd.read_excel("IndicatorsDashboard.xlsx",engine='openpyxl').dropna(subset=['Selector'])
        self.IndicatorsDF = pe.IndicatorsHandler(self.IndicatorsDF, self.IndicatorsData, SelectorHandler=[[0, "exclude"], [2, "diff"]])
        self.AlternativeDataFieldsIndicators = pd.read_sql('SELECT * FROM AlternativeDataFieldsIndicators', self.DataDeckConn).set_index('date', drop=True)
        self.IndicatorsDF = pd.concat([self.IndicatorsDF, self.AlternativeDataFieldsIndicators], axis=1).sort_index()
        self.IndicatorsDF.index = pd.to_datetime(self.IndicatorsDF.index)
        self.GalileoDataDF = pd.read_excel(self.AssetsDashboardExcel, sheet_name="Galileo")
        self.GlobalYields = self.GalileoDataDF["Yields"].dropna().tolist()
        self.GalileoMapperDF = self.GalileoDataDF[["Yields", "FuturesMarkets"]]

    def getData(self):
        #########################################################################################################################
        DataList = []
        for fld in ["PX_LAST", "YLD_YTM_MID"]:
            self.AllTickers = self.GalileoDataDF[self.GalileoDataDF['Fields'] == fld]["Yields"].dropna().tolist()
            self.con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
            self.fetchData = self.con.bdh(self.AllTickers, [fld], "20000101", '21000630').ffill().bfill()
            self.fetchData.columns = [x[0].replace("'", "") for x in self.fetchData.columns]
            DataList.append(self.fetchData)
        AllDataDF = pd.concat(DataList, axis=1)
        AllDataDF.to_sql("GalileoDataDeck", self.workConn, if_exists='replace')

    def CarryTrader(self, yieldBaseMaturity, topCarries):

        self.FuturesDF = self.AllRetsDF[list(set(self.GalileoMapperDF["FuturesMarkets"].tolist()))]
        self.FuturesDF.index = pd.to_datetime(self.FuturesDF.index)
        self.Futures_RVs = pe.RV(self.FuturesDF, HedgeRatioConnectionMode="Spreads", mode="Linear", n=25)

        self.AllRetsDF.index = pd.to_datetime(self.AllRetsDF.index)
        self.AllRetsDF = self.AllRetsDF.loc[[x for x in self.Futures_RVs.index]]

        self.GlobalYieldsDF = pd.read_sql('SELECT * FROM GalileoDataDeck', self.workConn).set_index('date', drop=True)[self.GlobalYields]
        self.GlobalYieldsDF.index = pd.to_datetime(self.GlobalYieldsDF.index)

        self.Group = pd.read_excel(self.AssetsDashboardExcel, sheet_name="Galileo").set_index("Yields")
        self.GroupList = list(self.Group[self.Group['Group']==yieldBaseMaturity].index)

        self.GlobalYieldsDF_ToRV = self.GlobalYieldsDF[self.GroupList]
        print(self.GlobalYieldsDF_ToRV)
        print([GalileoMapper(self.GalileoMapperDF, "Yields", "FuturesMarkets", x) for x in self.GlobalYieldsDF_ToRV.columns])
        self.GlobalYieldsDF_ToRV.columns = [GalileoMapper(self.GalileoMapperDF, "Yields", "FuturesMarkets", x) for x in self.GlobalYieldsDF_ToRV.columns]

        self.GlobalYields_RVs = pe.RV(self.GlobalYieldsDF_ToRV, HedgeRatioConnectionMode="Spreads", mode="Linear", n=25)
        self.GlobalYields_RVs_Top = self.GlobalYields_RVs.apply(lambda row: sorted(row.nlargest(topCarries).index.tolist()), axis=1).astype(str)
        self.GlobalYields_RVs_Top.to_sql("GlobalYields_RVs_Top_" + yieldBaseMaturity, self.workConn, if_exists='replace')

        ###############################################################################################################
        #fig, ax = plt.subplots(nrows=4, ncols=1)
        #pe.cs(self.Futures_RVs).plot(ax=ax[0], legend=None)

        self.Sig_Futures_RVs = pd.DataFrame(0, index=self.Futures_RVs.index, columns=self.Futures_RVs.columns)
        for c in self.Futures_RVs.columns:
            "Indexes That Contain The Pair into the top List"
            self.Sig_Futures_RVs[c] = self.GlobalYields_RVs_Top.str.contains(c).astype(int)
            self.Futures_RVs[c] *= self.Sig_Futures_RVs[c]

        self.Futures_RVs.to_sql("Futures_RVs_"+yieldBaseMaturity, self.workConn, if_exists='replace')
        self.Sig_Futures_RVs.to_sql("Sig_Futures_RVs_"+yieldBaseMaturity, self.workConn, if_exists='replace')
        ###############################################################################################################
        self.UnderlyingsSignalsDF = pe.RVSignalHandler(self.Sig_Futures_RVs, HedgeRatioMul=-1).fillna(0)
        self.UnderlyingsSignalsDF.to_sql("UnderlyingsSignalsDF_" + yieldBaseMaturity, self.workConn, if_exists='replace')

        self.Futures_RVs["TOTAL"] = pe.rs(self.Futures_RVs)

        self.Futures_RVs["TOTAL"].to_sql("PnL_Galileo_CarryTrader_" + yieldBaseMaturity, self.workConn, if_exists='replace')

        #pe.cs(self.Futures_RVs).plot(ax=ax[1], legend=None)
        #pe.cs(self.Futures_RVs["TOTAL"]).plot(ax=ax[2])
        #self.IndicatorsDF[['JPMVXYEM Index', 'JPMVXYG7 Index']].plot(ax=ax[3])
        #plt.show()

    def DT_Filter(self, RunMode, **kwargs):

        if 'Leverage' in kwargs:
            Leverage = kwargs['Leverage']
        else:
            Leverage = 10

        if 'RiskParity' in kwargs:
            RiskParity = kwargs['RiskParity']
        else:
            RiskParity = True

        if 'ActivateDTFilter' in kwargs:
            ActivateDTFilter = kwargs['ActivateDTFilter']
        else:
            ActivateDTFilter = True

        if 'RoundSignalDecimals' in kwargs:
            RoundSignalDecimals = kwargs['RoundSignalDecimals']
        else:
            RoundSignalDecimals = 2

        if 'Add_Premia' in kwargs:
            Add_Premia = kwargs['Add_Premia']
        else:
            Add_Premia = ['NQ1 Index', "TY1 Comdty",'VG1 Index', "RX1 Comdty"]

        self.CarryTraderPnLs_List = []
        for maturity in MaturitiesList:
            subPnl = pd.read_sql('SELECT * FROM PnL_Galileo_CarryTrader_'+maturity, self.workConn).set_index('date', drop=True)
            subPnl.columns = ["Galileo_CarryTrader_"+maturity]
            self.CarryTraderPnLs_List.append(subPnl)
        self.CarryTraderPnLs_DF = pd.concat(self.CarryTraderPnLs_List,axis=1).sort_index()
        self.CarryTraderPnLs_DF.index = pd.to_datetime(self.CarryTraderPnLs_DF.index)

        #RunMode = "Update"
        RollModeScanList = [["RollWindow", 25]]#,["ExpWindow", 25],["RollWindow", 250]]
        DT_Sig_List = []
        DT_Filtered_PnL_List = []
        for MacroSplitter in ['JPMVXYEM Index']:#,'JPMVXYG7 Index']:
            for RollModeIn in RollModeScanList:
                #############################################################################################################################
                DT_ID = "Galileo_" + RollModeIn[0] + "_" + str(RollModeIn[1]) + "_" + MacroSplitter
                print("Running " + DT_ID)
                argDict = {
                    "Mode": RunMode,
                    "DT_RetsDF": self.CarryTraderPnLs_DF,
                    "IndicatorsDF": self.IndicatorsDF,
                    "Raws": self.IndicatorsDF_Raw,
                    "RollMode": RollModeIn[0],
                    "st": RollModeIn[1],
                    "MacroSplitter": MacroSplitter,
                    "DT_Update_Depth": 25,
                    "WriteFilePath": self.AlternativeStorageLocation + "DecisionTrees/Galileo/" + DT_ID
                }
                DecisionTrees_RV_Runner(argDict)
                #############################################################################################################################
                if RunMode == "Update":
                    DT_Pack = pe.readPickleDF(argDict["WriteFilePath"])
                    DT_Pack.index = pd.to_datetime(DT_Pack.index)

                    for maturity in MaturitiesList: #
                        sub_DT_Pack = DT_Pack[[x for x in DT_Pack.columns if maturity in x]]
                        sub_DT_Pack = sub_DT_Pack[sub_DT_Pack.columns[0]]

                        DT_Sig = pe.sign(self.IndicatorsDF[MacroSplitter] - sub_DT_Pack)
                        DT_Sig[DT_Sig > 0] = 0
                        DT_Sig = DT_Sig.abs()
                        DT_Sig.to_sql("DT_Sig_" + maturity, self.workConn, if_exists='replace')

                        DT_Sig_Level_Report = self.IndicatorsDF[MacroSplitter].astype(str) + " :: " + sub_DT_Pack.astype(str)
                        DT_Sig_Level_Report.to_sql("DT_Sig_Level_Report_" + maturity, self.workConn, if_exists='replace')

                        UnderlyingsSignalsDF = pd.read_sql('SELECT * FROM UnderlyingsSignalsDF_'+maturity, self.workConn).set_index('date', drop=True)
                        if ActivateDTFilter == True:
                            UnderlyingsSignalsFilteredDF = UnderlyingsSignalsDF.mul(DT_Sig,axis=0).fillna(0)
                            DT_Filtered_PnL = self.CarryTraderPnLs_DF['Galileo_CarryTrader_' + maturity] * pe.S(DT_Sig,nperiods=1)
                        else:
                            UnderlyingsSignalsFilteredDF = UnderlyingsSignalsDF.copy()
                            DT_Filtered_PnL = self.CarryTraderPnLs_DF['Galileo_CarryTrader_' + maturity].copy()

                        if RiskParity == True:
                            RollVol = (np.sqrt(252) * pe.roller(DT_Filtered_PnL, np.std, n=250) * 100).ffill().bfill()
                            RollVol.name = 'Galileo_CarryTrader_' + maturity + "_" + DT_ID
                            RollVol.to_sql("RollVolDF_" + maturity, self.workConn,if_exists='replace')

                            UnderlyingsSignalsFilteredDF = UnderlyingsSignalsFilteredDF.div(RollVol,axis=0).round(RoundSignalDecimals)
                            UnderlyingsSignalsFilteredDF *= Leverage

                            DT_Filtered_PnL = pe.rs((UnderlyingsSignalsFilteredDF.shift().bfill() * self.AllRetsDF[UnderlyingsSignalsFilteredDF.columns]).fillna(0))

                        DT_Filtered_PnL.name = 'Galileo_CarryTrader_' + maturity + "_" + DT_ID

                        if Add_Premia == True:
                            pass

                        DT_Filtered_PnL.to_sql("DT_Filtered_PnL_" + maturity, self.workConn,if_exists='replace')
                        UnderlyingsSignalsFilteredDF.to_sql("UnderlyingsSignalsDF_Filtered_" + maturity, self.workConn,if_exists='replace')

                        "Append Lists"
                        DT_Filtered_PnL_List.append(DT_Filtered_PnL)
                        DT_Sig_List.append(UnderlyingsSignalsFilteredDF)
                        qs.reports.html(DT_Filtered_PnL, compounded=False,title=DT_Filtered_PnL.name,output="StrategiesFactSheets/"+DT_Filtered_PnL.name + ".html")

                        AllPnL = pd.concat([self.CarryTraderPnLs_DF['Galileo_CarryTrader_'+maturity], DT_Filtered_PnL],axis=1).sort_index()
                        fig,ax = plt.subplots(nrows=2, ncols=1)
                        pe.cs(AllPnL).tail(250).plot(ax=ax[0])
                        sub_DT_Pack.tail(250).plot(ax=ax[1])
                        self.IndicatorsDF[MacroSplitter].tail(250).plot(ax=ax[1])
                        plt.savefig(self.AlternativeStorageLocation+"\HealthChecksPacks\Galileo_DT_VisualCheck/"+DT_Filtered_PnL.name)
        #############################################################################################################################
        if RunMode == "Setup":
            print("DT Setup done ... !")
        elif RunMode == "Update":
            #########################################################################################################################
            Total_UnderlyingsSignalsFilteredDF = DT_Sig_List[0]
            for sig in DT_Sig_List[1:]:
                Total_UnderlyingsSignalsFilteredDF += sig
            Total_UnderlyingsSignalsFilteredDF.to_sql("Total_UnderlyingsSignals_FilteredDF", self.workConn,if_exists='replace')
            "Convert To Num of Futures"
            ExposuresNotionals = Total_UnderlyingsSignalsFilteredDF * self.AUM
            ExposuresNotionals.to_sql("ExposuresNotionals", self.workConn,if_exists='replace')
            #########################################################################################################################
            HistContractValues = pd.read_sql('SELECT * FROM HistContractValues', self.DataDeckConn).set_index('date', drop=True)[Total_UnderlyingsSignalsFilteredDF.columns]
            NumContracts = (ExposuresNotionals / HistContractValues).round(0)
            NumContracts.to_sql("NumContracts", self.workConn,if_exists='replace')
            #########################################################################################################################
            DT_Filtered_PnL_DF = pd.concat(DT_Filtered_PnL_List,axis=1).sort_index()
            DT_Filtered_PnL_DF['TOTAL'] = pe.rs(DT_Filtered_PnL_DF)
            #########################################################################################################################
            qs.reports.html(DT_Filtered_PnL_DF['TOTAL'], compounded=False, title="CarryTrader_Total",
                            output="StrategiesFactSheets/Galileo_Total.html")
            
    #############################################################################################################################

    def Run_ECO_Trader(self):

        self.SurveyShiftDays = 1

        self.Active_ECO_Indicators = []
        for country in ["USA"]:
            country_Active_ECO_IndicatorsDF = pd.read_excel(self.ECO_Indicators_Main_Excel, sheet_name=country)
            country_Active_ECO_IndicatorsDF = country_Active_ECO_IndicatorsDF[country_Active_ECO_IndicatorsDF["GG_Selector"]>0][['Ticker', "AssetsToTrade"]]
            country_Active_ECO_IndicatorsDF = country_Active_ECO_IndicatorsDF.set_index("Ticker",drop=True)
            self.Active_ECO_Indicators.append(country_Active_ECO_IndicatorsDF)

        self.Active_ECO_IndicatorsDF = pd.concat(self.Active_ECO_Indicators).dropna()
        print(self.Active_ECO_IndicatorsDF)

        self.ECO_Indicators_PX_LAST = pd.read_sql('SELECT * FROM ECO_Indicators_PX_LAST', self.ECO_IndicatorsConn).set_index('date', drop=True)
        self.ECO_Indicators_B_SURVEY_MEDIAN = pd.read_sql('SELECT * FROM ECO_Indicators_BN_SURVEY_MEDIAN', self.ECO_IndicatorsConn).set_index('date', drop=True)
        self.ECO_Indicators_Spread = pd.read_sql('SELECT * FROM ECO_Indicators_Spread', self.ECO_IndicatorsConn).set_index('date', drop=True)

        for c in self.Active_ECO_IndicatorsDF.index:

            AssetsToTradeList = country_Active_ECO_IndicatorsDF.loc[c,"AssetsToTrade"].split(",")

            plotDF = pd.concat([self.ECO_Indicators_PX_LAST[c], self.ECO_Indicators_B_SURVEY_MEDIAN[c].shift(self.SurveyShiftDays)],axis=1).sort_index()
            plotDF.columns = [c,c+"_Survey_Median"]

            ECO_Sig = pe.S(pe.sign(plotDF[c]-plotDF[c+"_Survey_Median"]), nperiods=1)
            ECO_PnL = self.AllRetsDF[AssetsToTradeList].mul(ECO_Sig,axis=0)
            ECO_Sh = np.sqrt(252) * pe.sharpe(ECO_PnL)
            print(ECO_Sh)

            fig,ax = plt.subplots(nrows=3,ncols=1)
            plotDF.plot(ax=ax[0])
            self.ECO_Indicators_Spread[c].plot(ax=ax[1])
            pe.cs(ECO_PnL).plot(ax=ax[2])
            plt.show()

obj = Galileo()
obj.getData()
for maturity in MaturitiesList: obj.CarryTrader(maturity, 10)
#obj.DT_Filter("Setup")
obj.DT_Filter("Update", ActivateDTFilter=True, RiskParity=True, Leverage=10, RoundSignalDecimals=2)
#obj.DT_Filter("Update", ActivateDTFilter=False, RiskParity=True, Leverage=10, RoundSignalDecimals=2)
##########################################################################################
#obj.Run_ECO_Trader()
