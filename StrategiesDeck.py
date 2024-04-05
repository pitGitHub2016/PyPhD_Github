from itertools import combinations, permutations
import pandas as pd, numpy as np, sqlite3, time, inspect, pickle
import matplotlib.pyplot as plt
from pyerb import pyerb as pe
from pyerbML import ML,ManSee
from hurst import compute_Hc

class StrategiesDeck:

    ############################# INITIALISERS & GENERAL FUNCTIONS #############################

    def __init__(self, SystemName, df, IndicatorsDF, Leverage):
        self.SystemName = SystemName
        self.df = df
        self.LocalDataConn = sqlite3.connect("DataDeck.db")
        self.AllDF = pd.read_sql('SELECT * FROM DataDeck', self.LocalDataConn).set_index('date', drop=True)
        self.IndicatorsDF = IndicatorsDF
        AlternativeDataFieldsIndicators = pd.read_sql('SELECT * FROM AlternativeDataFieldsIndicators', self.LocalDataConn).set_index('date', drop=True)
        self.IndicatorsDF = pd.concat([self.IndicatorsDF, AlternativeDataFieldsIndicators], axis=1).sort_index()
        self.Leverage = Leverage
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable', self.LocalDataConn).set_index('index', drop=True)
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.ActiveStrategiesFactorsControl = pd.read_excel("AssetsDashboard.xlsx",sheet_name="ActiveStrategiesFactorsControl").set_index("Asset", drop=True)
        "GET VOLUMES"
        try:
            self.VolumeDeck = pd.read_sql('SELECT * FROM VolumeDeck', self.LocalDataConn).set_index('date', drop=True)
        except Exception as e:
            print(e)
        try:
            self.DailyTurnoverDF = pd.read_sql('SELECT * FROM DailyTurnoverDF', self.LocalDataConn).set_index('date',drop=True)
        except Exception as e:
            print(e)
    def VolumeFilter(signal, **kwargs):

        if 'VolumeDF' in kwargs:
            VolumeDF = kwargs['VolumeDF']
        else:
            VolumeDF = pd.read_sql('SELECT * FROM VolumeDeck', sqlite3.connect("DataDeck.db")).set_index('date',drop=True)[signal.columns]

        if 'DailyTurnoverDF' in kwargs:
            DailyTurnoverDF = kwargs['DailyTurnoverDF']
        else:
            DailyTurnoverDF = pd.read_sql('SELECT * FROM DailyTurnoverDF', sqlite3.connect("DataDeck.db")).set_index('date',drop=True)[signal.columns]

        if "FilterSet" in kwargs:
            FilterSet = kwargs['FilterSet']
        else:
            FilterSet = {'VolumeLowerThr': [True,100],'DailyTurnoverDFLowerThr': [True,1000000]}

        out = signal

        if FilterSet['VolumeLowerThr'][0]:
            VolumeDF[VolumeDF < FilterSet['VolumeLowerThr'][1]] = 0
        if FilterSet['DailyTurnoverLowerThr'][0]:
            VolumeDF[DailyTurnoverDF < FilterSet['DailyTurnoverLowerThr'][1]] = 0

        out *= pe.sign(VolumeDF.fillna(0))

        return out
    def ExPostRebase(signal, rets, **kwargs):
        if 'exPostRebaseCompounds' in kwargs:
            exPostRebaseCompoundsFlag = kwargs['exPostRebaseCompounds']
        else:
            exPostRebaseCompoundsFlag = "NO"

        if 'exPostRebaseCompounds_nIn' in kwargs:
            exPostRebaseCompounds_nIn = kwargs['exPostRebaseCompounds_nIn']
        else:
            exPostRebaseCompounds_nIn = 250

        if 'exPostVol' in kwargs:
            exPostVolFlag = kwargs['exPostVol']
        else:
            exPostVolFlag = "NO"

        if 'exPostVolMode' in kwargs:
            exPostVolMode = kwargs['exPostVolMode']
        else:
            exPostVolMode = "RollWindow"

        if 'exPostVolLag' in kwargs:
            exPostVolLag = kwargs['exPostVolLag']
        else:
            exPostVolLag = 1000

        if 'exPostHurstControl' in kwargs:
            exPostHurstControlFlag = kwargs['exPostHurstControl']
        else:
            exPostHurstControlFlag = "NO"

        "ExPost Strategy PnL"
        rsStrat = pe.rs((signal * rets).fillna(0))

        propsOut = dict()
        if exPostRebaseCompoundsFlag == 'YES':
            "Strategy ExPost Rebasing on Ann. Compounding"
            compoundScaler = 1 + rsStrat.rolling(exPostRebaseCompounds_nIn).sum()
            propsOut['compoundScaler'] = compoundScaler
            signal = signal.mul(compoundScaler, axis=0)
        else:
            propsOut['compoundScaler'] = "No compounding Done..."
        if exPostVolFlag == "YES":
            "Strategy ExPost Risk Parity & Rebasing on Ann. Compounding"
            if exPostVolMode == "RollWindow":
                volExpander = (np.sqrt(252) * pe.roller(rsStrat, np.std, n=exPostVolLag) * 100).ffill().bfill()
            else:
                volExpander = (np.sqrt(252) * pe.expander(rsStrat, np.std, n=25) * 100).ffill().bfill()
            #############################################################################################
            propsOut['volExpander'] = volExpander
            signal = signal.div(volExpander, axis=0).fillna(0)
        else:
            propsOut['volExpander'] = "No Vol Adjustment Done..."
        if exPostHurstControlFlag == "YES":
            H = lambda x: compute_Hc(x)[0]
            HurstDF = rsStrat.rolling(500).apply(H) - 0.5
            HurstFilter = pe.sign(HurstDF)
            HurstFilter[HurstFilter < 0] = 0
            signal = signal.div(HurstFilter, axis=0).fillna(0)
        else:
            propsOut['HurstFilter'] = "No Hurst Filter Done..."

        return [signal, propsOut]

    ##################################### STRATEGIES CORES #####################################

    def EMA_RP(self, **kwargs):
        if "DEMA_ID" in kwargs:
            DEMA_ID = kwargs["DEMA_ID"]
        else:
            DEMA_ID = "_DefaultSingleLookBack_LookBacksPack_DEMAs"
        ###################################################################################################################
        if self.StrategySettings['EMA_Mode'][0] == "DEMA":
            self.DEMAs = pe.readPickleDF(self.AlternativeStorageLocation + "DEMAs/" + DEMA_ID)[self.localRets.columns]
            self.LookbacksDirections = 1
            #self.LookbacksPacks = pe.readPickleDF(self.AlternativeStorageLocation + "LookbacksPacks/" + DEMA_ID.replace("_DEMAs",""))
            #print(self.LookbacksPacks[0]["NQ1 Index"])
            #print(self.LookbacksPacks[1]["NQ1 Index"])
            #print(self.LookbacksPacks[2]["NQ1 Index"])
            #time.sleep(3000)
            #self.LookbacksDirections = self.LookbacksPacks[1][self.localRets.columns]
            #self.LookbacksDirections[self.LookbacksDirections < 0] = 0
            self.sigRaw = pe.sign(self.DEMAs.fillna(0)) * self.LookbacksDirections
        elif self.StrategySettings['EMA_Mode'][0] == "EMA":
            self.sigRaw = pe.sign(pe.ema(self.localRets, nperiods=self.StrategySettings['EMA_Mode'][1]))

        self.RollingVolatilities = 1
        if self.StrategySettings['RP_Mode'][0] == "Standard":
            self.RollingVolatilities = np.sqrt(252) * pe.rollStatistics(self.localRets, 'Vol', nIn=self.StrategySettings['RP_Mode'][1]) * 100
        elif self.StrategySettings['RP_Mode'][0] == "DEMA":
            self.RollingVolatilities = pe.dema(self.localRets, self.LookBacks[self.localRets.columns],mode="AnnVol").ffill().bfill() * 100

        if self.StrategySettings['DirectionBias'] == "LO":
            "Long Only"
            self.sigRaw[self.sigRaw < 0] = 0
        elif self.StrategySettings['DirectionBias'] == "SO":
            "Short Only"
            self.sigRaw[self.sigRaw > 0] = 0
        "Handle Stabilisers"
        if self.StrategySettings['IndicatorsStabilisers'] is not None:
            if self.StrategySettings['IndicatorsStabilisers'] not in ["DT"]:
                for stabiliser in self.StrategySettings['IndicatorsStabilisers']:
                    Smoothed_IndicatorsStabiliser = self.IndicatorsDF[stabiliser]
                    "Smooth Stabiliser"
                    if self.StrategySettings['SmoothStabilisers'][0] == "EMA":
                        Smoothed_IndicatorsStabiliser = pe.ema(Smoothed_IndicatorsStabiliser,
                                                               nperiods=self.StrategySettings['SmoothStabilisers'][1])
                    sig_IndicatorsStabiliser = pe.sign(Smoothed_IndicatorsStabiliser)
                    "Restrict Stabiliser"
                    if self.StrategySettings['SmoothStabilisers'][2] == "LO":
                        sig_IndicatorsStabiliser[sig_IndicatorsStabiliser < 0] = 0
                    elif self.StrategySettings['SmoothStabilisers'][2] == "SO":
                        sig_IndicatorsStabiliser[sig_IndicatorsStabiliser > 0] = 0
                    "Additive or Multiplicative Stabilisers"
                    if self.StrategySettings['SmoothStabilisers'][3] == "Additive":
                        self.sigRaw = pe.sign(self.sigRaw.add(sig_IndicatorsStabiliser, axis=0)).sort_index()
                    elif self.StrategySettings['SmoothStabilisers'][3] == "Multiplicative":
                        self.sigRaw = self.sigRaw.mul(sig_IndicatorsStabiliser, axis=0).sort_index()
            elif self.StrategySettings['IndicatorsStabilisers'] in ["DT"]:
                ############################################################################################
                for c in self.sigRaw.columns:
                    ActiveContract = c
                    #ActiveContract = self.FuturesTable["Point_1"].iloc[pe.getIndexes(self.FuturesTable, c)[0][0]]
                    if c in list(self.ActiveStrategiesFactorsControl.index):#["NQ1 Index"]
                        print(c)
                        Combined_DT_Sig_List = []
                        ############################################################################################
                        for DT_Roll_Mode in ['_RollWindow_250_']:#'_RollWindow_250_','_ExpWindow_25_'
                        #for DT_Roll_Mode in ['_ExpWindow_25_']:#'_RollWindow_250_','_ExpWindow_25_'
                            ############################################################################################
                            sub_DT_Map = self.ActiveStrategiesFactorsControl.loc[ActiveContract, :]
                            ############################################################################################
                            DT_Map_Pack_List = [x for x in sub_DT_Map["SingleDecisionTreesControllers_GG"].split(":") if x != '']
                            for item_DT_Map_Pack_List in DT_Map_Pack_List:
                                DT_Map_Pack = item_DT_Map_Pack_List.split("_")
                                DT_Thr = pe.readPickleDF(self.AlternativeStorageLocation + "DecisionTrees" + self.StrategySettings['SmoothStabilisers'][0] + "/RetsLive/DecisionTrees_RV_Rets"+ DT_Roll_Mode + DT_Map_Pack[0])
                                try:
                                    DT_Sig = pe.sign(DT_Thr[DT_Map_Pack[0] + "," + c + ",TreeThreshold"] - self.IndicatorsDF[DT_Map_Pack[0]])  # .fillna(0)
                                except:
                                    print(DT_Thr.columns)
                                    print(DT_Thr[DT_Map_Pack[0] + "," + c + ",TreeThreshold"])
                                    print(self.IndicatorsDF[DT_Map_Pack[0]])
                                ########################################
                                if DT_Map_Pack[1] == "Upper":
                                    DT_Sig[DT_Sig > 0] = 0
                                elif DT_Map_Pack[1] == "Lower":
                                    DT_Sig[DT_Sig < 0] = 0
                                ############################################################################################
                                print("DT_Roll_Mode = ", DT_Roll_Mode, " | DT_Map_Pack : ", DT_Map_Pack)
                                if DT_Map_Pack[2] == "Positive":
                                    FilterSig = DT_Sig
                                elif DT_Map_Pack[2] == "Negative":
                                    FilterSig = DT_Sig * (-1)
                                ############################################################################################
                                Combined_DT_Sig_List.append(FilterSig)
                        ############################################################################################
                        MomentumSig = self.sigRaw[c]
                        ############################################################################################
                        FinalSig = Combined_DT_Sig_List[0] * MomentumSig
                        for fSig in Combined_DT_Sig_List[1:]:
                            FinalSig += fSig * MomentumSig
                            #FinalSig[FinalSig.abs() == 1] = 0
                            #FinalSig = pe.sign(FinalSig)
                        ############################################################################################
                        #FinalSig /= len(Combined_DT_Sig_List)
                        ############################################################################################
                        #self.sigRaw[c] = FinalSig
                        self.sigRaw[c] = pe.sign(FinalSig)
                ############################################################################################
        self.sigRaw = self.sigRaw / self.RollingVolatilities
        #try:
        #    pe.cs((pe.S(self.sigRaw[["NQ1 Index", "ES1 Index"]], nperiods=2)*self.localRets[["NQ1 Index", "ES1 Index"]]).fillna(0)).plot()
        #    plt.show()
        #except Exception as e:
        #    print(e)
        return self

    ##################################### SPECIFIC CALLERS #####################################
    def Endurance(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn, if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Coast(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Brotherhood(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self,
                              DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets,
                                                exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'],
                                                exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],
                                                          'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def ShoreDM(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        #sDF["DX1 Curncy"] *= 0
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def ShoreEM(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'],
                                                exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        #sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Valley(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")
        ##########################################################################################
        "Neutralise first point"
        self.sigRaw[[x for x in self.sigRaw.columns if x in self.FuturesTable["Point_1"].tolist()]] *= 0
        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Dragons(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Lumen(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "SO",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'],
                                                exPostVol=self.StrategySettings['exPostVolIn'],
                                                exPostVolMode="ExpWindow"
                                                )
        sDF = RebaseOut[0]
        #################################### NEUTRALISE ##########################################
        sDF["FVS1 Index"] *= 0
        ##########################################################################################
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage
    ############################################################################################
    def Fidei(self):

        self.StrategyName = inspect.stack()[0][3]
        self.stategyDBconn = sqlite3.connect(self.StrategyName + ".db")
        self.localRets = pe.dlog(self.df).fillna(0)

        self.StrategySettings = {
            "EMA_Mode": ["DEMA", 250],
            "RP_Mode": [None],  # ["Standard", 250]
            "DirectionBias": "NoBias",  # "LO", "SO"
            "IndicatorsStabilisers": "DT",  # None,[Ind1, Ind2],"DT"
            "SmoothStabilisers": [""],  # ["EMA", 250, "", "Additive"],"Positive","Negative","Both"
            "exPostRebaseCompoundsIn": "NO",
            "exPostVolIn": "YES"
        }
        StrategiesDeck.EMA_RP(self, DEMA_ID="_DefaultSingleLookBack_LookBacksPack_DEMAs")

        ##########################################################################################
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(self.sigRaw, self.localRets, exPostRebaseCompounds=self.StrategySettings['exPostRebaseCompoundsIn'], exPostVol=self.StrategySettings['exPostVolIn'])
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, self.stategyDBconn, if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        pe.fd(sDF).to_sql(self.SystemName + "_" + self.StrategyName + "_sDF_PreVolumeFiltered", self.stategyDBconn,if_exists='replace')
        ##############################################################################################################
        sDF = StrategiesDeck.VolumeFilter(sDF, FilterSet={'VolumeLowerThr': [True, 100],'DailyTurnoverLowerThr': [True, 1000000]})
        ##############################################################################################################
        print("self.Leverage = ", self.Leverage)
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + self.StrategyName + "_sDF", self.stategyDBconn, if_exists='replace')
        return sDF * self.Leverage

#test = StrategiesDeck()
