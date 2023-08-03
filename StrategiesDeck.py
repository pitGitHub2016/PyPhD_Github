from itertools import combinations, permutations
import pandas as pd, numpy as np, sqlite3, time, inspect
import matplotlib.pyplot as plt
from pyerb import pyerb as pe
from pyerbML import ML,ManSee
from hurst import compute_Hc

class StrategiesDeck:

    def __init__(self, SystemName, df, IndicatorsDF, Leverage):
        self.SystemName = SystemName
        self.df = df
        self.IndicatorsDF = IndicatorsDF
        self.Leverage = Leverage
        try:
            self.LookBacks = pd.read_sql("SELECT * FROM LookBacks", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
            self.DEMAs = pd.read_sql("SELECT * FROM DEMAs", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        except Exception as e:
            print(e)

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
            volExpander = (np.sqrt(252) * pe.roller(rsStrat, np.std, n=1000) * 100).ffill().bfill()
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

    def Endurance(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        RollingVolatilities = pe.dema(localRets, self.LookBacks[localRets.columns], mode="AnnVol").ffill().bfill()
        RollingVolatilities.to_sql(self.SystemName + "_" + strategyName + "_RollingVolatilities",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        ###############################################################################################################
        #EMAs = pe.ema(localRets, nperiods=250)
        #EMAs.to_sql(self.SystemName+"_"+strategyName+"_EMAs", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        #driver = pe.sign(EMAs)
        ###############################################################################################################
        driver = pe.sign(self.DEMAs[localRets.columns])
        ###############################################################################################################
        driver[driver < 0] = 0
        "FINANCIAL CONDITIONS INDEXES"
        #FCIs = self.IndicatorsDF[["BFCIUS Index", "BFCIEU Index"]]
        ################################ STATIC FCI FILTER #########################################
        #kernelFCIs = pd.DataFrame(1, index=FCIs.index,columns=FCIs.columns)
        #kernelFCIs[FCIs <= -1] = 0
        #for c in driver.columns:
        #    driver[c] *= kernelFCIs["BFCIUS Index"]
        ################################ DECISION TREE FCI FILTER ########################################
        DecisionTrees_RV = pd.read_sql("SELECT * FROM DecisionTrees_RV", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        [DT_FilterDF, DT_Aggregated_SigDF] = ML.DecisionTrees_Signal_Filter(localRets, self.IndicatorsDF, DecisionTrees_RV, ["VIX Index", "MOVE Index"])
        DT_FilterDF.to_sql(self.SystemName + "_" + strategyName + "_DT_FilterDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DT_Aggregated_SigDF.to_sql(self.SystemName + "_" + strategyName + "_DT_Aggregated_SigDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        "DT Filter Out Final Driver (Signal)"
        driver *= DT_Aggregated_SigDF

        driver.to_sql(self.SystemName + "_"+strategyName+"_driver_Filtered", sqlite3.connect("SkeletonDB.db"), if_exists='replace')

        ######################################## RISK PARITY #############################################
        driver["NQ1 Index"] = driver["NQ1 Index"].div(pd.concat([self.IndicatorsDF["VXN Index"], RollingVolatilities["NQ1 Index"]], axis=1).max(axis=1), axis=0)
        driver["ES1 Index"] = driver["ES1 Index"].div(pd.concat([self.IndicatorsDF["VIX Index"], RollingVolatilities["ES1 Index"]], axis=1).max(axis=1), axis=0)
        driver["DM1 Index"] = driver["DM1 Index"].div(pd.concat([self.IndicatorsDF["VXD Index"], RollingVolatilities["DM1 Index"]], axis=1).max(axis=1), axis=0)
        driver["GX1 Index"] = driver["GX1 Index"].div(pd.concat([self.IndicatorsDF["VDAX Index"], RollingVolatilities["GX1 Index"]], axis=1).max(axis=1), axis=0)
        driver["VG1 Index"] = driver["VG1 Index"].div(pd.concat([self.IndicatorsDF["V2X Index"], RollingVolatilities["VG1 Index"]], axis=1).max(axis=1), axis=0)
        driver["CF1 Index"] = driver["CF1 Index"].div(pd.concat([self.IndicatorsDF["V2X Index"], RollingVolatilities["CF1 Index"]], axis=1).max(axis=1), axis=0)

        sDF = driver
        #sDF = driver / RollingVolatilities
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES") #, exPostHurstControl="YES"
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']: #'HurstFilter'
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"), if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.loc[["2002-04-10 00:00:00", "2002-04-11 00:00:00"], "DM1 Index"] = 0
        sDF.to_sql(self.SystemName+"_"+strategyName+"_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def Coast(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        #################################################################################################################################
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        RollingVolatilities = pe.dema(localRets, self.LookBacks[localRets.columns], mode="AnnVol").ffill().bfill()
        RollingVolatilities.to_sql(self.SystemName + "_" + strategyName + "_RollingVolatilities", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        #################################################################################################################################
        #EMAs = pe.ema(localRets, nperiods=250)
        #EMAs.to_sql(self.SystemName + "_" + strategyName + "_EMAs", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        #driver = pe.sign(EMAs)
        #################################################################################################################################
        driver = pe.sign(self.DEMAs[localRets.columns])
        #################################################################################################################################
        "FINANCIAL CONDITIONS INDEXES"
        #FCIs = self.IndicatorsDF[["BFCIUS Index", "BFCIEU Index", "BFCIGB Index"]]
        #kernelFCIs = pd.DataFrame(1, index=FCIs.index, columns=FCIs.columns)
        #kernelFCIs[FCIs <= -1.5] = 0
        #for c in driver.columns:
        #    driver[c] *= kernelFCIs["BFCIUS Index"]
        #################### DECISION TREES ###################################################
        DecisionTrees_RV = pd.read_sql("SELECT * FROM DecisionTrees_RV", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        [DT_FilterDF, DT_Aggregated_SigDF] = ML.DecisionTrees_Signal_Filter(localRets, self.IndicatorsDF,DecisionTrees_RV,["VIX Index", "MOVE Index"])
        DT_FilterDF.to_sql(self.SystemName + "_" + strategyName + "_DT_FilterDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DT_Aggregated_SigDF.to_sql(self.SystemName + "_" + strategyName + "_DT_Aggregated_SigDF",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        "DT Filter Out Final Driver (Signal)"
        driver *= DT_Aggregated_SigDF

        driver.to_sql(self.SystemName + "_" + strategyName + "_driver_Filtered", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        ######################### RISK PARITY #####################################################
        for c in driver.columns:
            driver[c] = driver[c].div(pd.concat([self.IndicatorsDF["MOVE Index"], RollingVolatilities[c]], axis=1).max(axis=1), axis=0)

        sDF = driver
        #sDF = driver / RollingVolatilities
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + strategyName + "_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def Brotherhood(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        ####################################################################################################################
        #driver = pe.sign(pe.ema(localRets, nperiods=25))
        ####################################################################################################################
        driver = pe.sign(self.DEMAs[localRets.columns])
        ####################################################################################################################
        RollingVolatilities = pe.dema(localRets, self.LookBacks[localRets.columns], mode="AnnVol").ffill().bfill()
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250).ffill().bfill() * 100

        for c in driver.columns:
            driver[c] = driver["DX1 Curncy"]
        driver["DX1 Curncy"] = 0
        driver["EC1 Curncy"] *= 0.576
        driver["JY1 Curncy"] *= 0.136
        driver["BP1 Curncy"] *= 0.119
        driver["CD1 Curncy"] *= 0.091
        driver["SF1 Curncy"] *= 0.042 + 0.036  # "Getting SEK exposure into Swiss Franc Futures"
        #################### DECISION TREES ###################################################
        DecisionTrees_RV = pd.read_sql("SELECT * FROM DecisionTrees_RV", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        [DT_FilterDF, DT_Aggregated_SigDF] = ML.DecisionTrees_Signal_Filter(localRets, self.IndicatorsDF, DecisionTrees_RV,["JPMVXYG7 Index"], BarrierDirections=["UpperReverse"])
        DT_FilterDF.to_sql(self.SystemName + "_" + strategyName + "_DT_FilterDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DT_Aggregated_SigDF.to_sql(self.SystemName + "_" + strategyName + "_DT_Aggregated_SigDF",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        "DT Filter Out Final Driver (Signal)"
        #driver = pe.sign(driver+DT_Aggregated_SigDF)
        driver *= DT_Aggregated_SigDF

        driver.to_sql(self.SystemName + "_"+strategyName+"_driver", sqlite3.connect("SkeletonDB.db"), if_exists='replace')

        for c in driver.columns:
            driver[c] = driver[c].div(pd.concat([self.IndicatorsDF["JPMVXYG7 Index"], RollingVolatilities[c]], axis=1).max(axis=1), axis=0)

        sDF = driver
        #sDF = driver / RollingVolatilities
        #sDF = driver.div(RollingVolatilities["DX1 Curncy"],axis=0)

        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        #sDF[sDF.abs() > 1] = 0
        sDF.to_sql(self.SystemName + "_"+strategyName+"_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def Shore(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        IRDs = self.IndicatorsDF[[x for x in self.IndicatorsDF.columns if "IS Curncy" in x]]
        DEMAsIRDs = self.DEMAs[IRDs.columns]
        IRDs.columns = [(pe.getFutureTicker(x.replace("IS Curncy", "")[:3]) + "_" + pe.getFutureTicker(
            x.replace("IS Curncy", "")[-3:])).replace("_USD", "") for x in IRDs.columns]
        DEMAsIRDs.columns = [(pe.getFutureTicker(x.replace("IS Curncy", "")[:3]) + "_" + pe.getFutureTicker(
            x.replace("IS Curncy", "")[-3:])).replace("_USD", "") for x in DEMAsIRDs.columns]
        #dIRDs = pe.d(IRDs)
        LookBacksRV_DF = pd.read_sql("SELECT * FROM "+self.SystemName + "_" + strategyName + "_LookBacksRV_DF", sqlite3.connect("SkeletonDB.db")).set_index('date', drop=True)
        #DEMAsRV_DF = pd.read_sql("SELECT * FROM "+self.SystemName + "_" + strategyName + "_DEMAsRV_DF", sqlite3.connect("SkeletonDB.db")).set_index('date', drop=True)

        #DEMAsRVCalculateStatus = "Setup"
        DEMAsRVCalculateStatus = "Update"
        #DEMAsRVCalculateStatus = "Read"

        nHR = 250; nEMA = 250; nIRDs = 250
        targetPairs = [c for c in list(permutations(localRets.columns, 2)) if c[0] == "BR1 Curncy"]
        "Enumerate Asset Occurencies"
        trAssetList = []
        for combo in targetPairs:
            trAssetList.append(combo[0])
            trAssetList.append(combo[1])
        trAssetDF = pd.Series(trAssetList).value_counts()
        "Strategy's Core"
        sRawList = []
        HedgeRatiosList = []
        RVsList = []
        LookBacksRVList = []
        DEMAsRVList = []
        for pair in targetPairs:
            subPairName = pair[0] + "_" + pair[1]
            assetX = localRets[pair[0]]
            assetY = localRets[pair[1]]
            HedgeRatio = pe.S(assetX.rolling(nHR).corr(assetY) * (
                    pe.roller(assetX, np.std, n=nHR) / pe.roller(assetY, np.std, n=nHR)))
            RV = assetX + HedgeRatio * assetY
            RV = pd.DataFrame(RV, columns=[subPairName])

            if DEMAsRVCalculateStatus == "Setup":
                LookBacksRV = pe.DynamicSelectLookBack(RV, RollMode="ExpWindow")
            elif DEMAsRVCalculateStatus == "Update":
                OldLookBacksRV = LookBacksRV_DF[subPairName]
                NewLookBacksRV = pe.DynamicSelectLookBack(RV, RollMode="ExpWindow", st=RV.shape[0]-10).dropna()[subPairName]
                LookBacksRV = pd.concat([OldLookBacksRV, NewLookBacksRV], axis=0)
                LookBacksRV = LookBacksRV[~LookBacksRV.index.duplicated(keep='last')]
                LookBacksRV = pd.DataFrame(LookBacksRV, columns=[subPairName])
                LookBacksRV.columns = [subPairName]
            elif DEMAsRVCalculateStatus == "Read":
                LookBacksRV = pd.DataFrame(LookBacksRV_DF[subPairName], columns=[subPairName])

            DEMAsRV = pe.dema(RV, LookBacksRV).fillna(0)
            DEMAsRV.columns = [subPairName]

            LookBacksRVList.append(LookBacksRV)
            DEMAsRVList.append(DEMAsRV)

            HedgeRatiosList.append(pd.DataFrame(HedgeRatio, index=RV.index, columns=[subPairName]))
            RVsList.append(RV)
            #subSig = pd.DataFrame(pe.sign(pe.sign(pe.ema(RV, nperiods=nEMA)) + pe.sign(pe.ema(dIRDs[subPairName], nperiods=nIRDs))),index=RV.index, columns=[subPairName])
            if isinstance(DEMAsRV, pd.DataFrame):
                DemaSubSig = DEMAsRV[subPairName]
            else:
                DemaSubSig = DEMAsRV

            subSig = pe.sign(pe.sign(DemaSubSig) + pe.sign(DEMAsIRDs[subPairName]))
            sRawList.append(subSig)

        LookBacksRV_DF = pd.concat(LookBacksRVList, axis=1).sort_index()
        DEMAsRV_DF = pd.concat(DEMAsRVList, axis=1).sort_index()
        LookBacksRV_DF.to_sql(self.SystemName + "_" + strategyName + "_LookBacksRV_DF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DEMAsRV_DF.to_sql(self.SystemName + "_" + strategyName + "_DEMAsRV_DF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')

        HedgeRatioDF = pd.concat(HedgeRatiosList, axis=1)
        RVsDF = pd.concat(RVsList, axis=1)
        sRaw = pd.concat(sRawList, axis=1)
        #RVsDF.columns = sRaw.columns

        sRaw.to_sql(self.SystemName + "_" + strategyName + "_sRaw", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        HedgeRatioDF.to_sql(self.SystemName + "_Shore_HedgeRatioDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        RVsDF.to_sql(self.SystemName + "_" + strategyName + "_RVsDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        sRaw.to_sql(self.SystemName + "_" + strategyName + "_sRollSharpeFiltered", sqlite3.connect("SkeletonDB.db"),if_exists='replace')

        "First Signal"
        sDF = pe.RVSignalHandler(sRaw, HedgeRatioDF=HedgeRatioDF)

        "Rebase each asset exposure on frequency per pair"
        for c in sDF.columns:
            sDF[c] /= trAssetDF[c]

        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + strategyName + "_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def Valley(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df).fillna(0)
        localRets.to_sql(self.SystemName + "_" + strategyName + "_localRets", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        RollingVolatilities = pe.dema(localRets, self.LookBacks[localRets.columns], mode="AnnVol").ffill().bfill()
        RollingVolatilities.to_sql(self.SystemName + "_" + strategyName + "_RollingVolatilities",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        ##########################################################################################################
        #driverEMA = pe.sign(pe.ema(localRets, nperiods=500)) # LIVE TRADED
        ##########################################################################################################
        driverEMA = pe.sign(self.DEMAs[localRets.columns])
        ##########################################################################################################
        driver = pd.DataFrame(0,index=localRets.index, columns=localRets.columns)
        for betaMap in [["TU1 Comdty","FF1 Comdty"],["DU1 Comdty","ER1 Comdty"]]:
            driver[betaMap[0]] = pe.beta(localRets, betaMap[0], betaMap[1], n=250) * driverEMA[betaMap[1]]
            #driver[betaMap[0]] = 1 * driverEMA[betaMap[1]]
        driver.to_sql(self.SystemName + "_"+strategyName+"_driver", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        "FINANCIAL CONDITION INDEXES"
        #FCIs = self.IndicatorsDF[["BFCIUS Index", "BFCIEU Index"]]
        #kernelFCIs = pd.DataFrame(1, index=FCIs.index, columns=FCIs.columns)
        #kernelFCIs[FCIs <= -1] = 0
        #for c in ["TU1 Comdty", "DU1 Comdty"]:
        #    driver[c] *= kernelFCIs["BFCIUS Index"]
        ######################## DECISION TREES ###############################################
        DecisionTrees_RV = pd.read_sql("SELECT * FROM DecisionTrees_RV", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        [DT_FilterDF, DT_Aggregated_SigDF] = ML.DecisionTrees_Signal_Filter(localRets, self.IndicatorsDF,DecisionTrees_RV,
                                                                            ["MOVE Index"], BarrierDirections=["Upper"])
        DT_FilterDF.to_sql(self.SystemName + "_" + strategyName + "_DT_FilterDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DT_Aggregated_SigDF.to_sql(self.SystemName + "_" + strategyName + "_DT_Aggregated_SigDF",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        "DT Filter Out Final Driver (Signal)"
        driver *= DT_Aggregated_SigDF
        # --------------------------------------------------------
        driver[[x for x in driver.columns if (x != "TU1 Comdty")&(x != "DU1 Comdty")]] = driverEMA[[x for x in driver.columns if (x != "TU1 Comdty") & (x != "DU1 Comdty")]]
        # --------------------------------------------------------
        #driver[["FF1 Comdty", "ZB1 Comdty", "SFR1 Comdty", "ER1 Comdty", "IR1 Comdty", "BA1 Comdty", ]] *= 0
        # --------------------------------------------------------
        driver.to_sql(self.SystemName + "_" + strategyName + "_driver_Filtered", sqlite3.connect("SkeletonDB.db"),if_exists='replace')

        for c in driver.columns:
            driver[c] = driver[c].div(pd.concat([self.IndicatorsDF["EUNS01 Curncy"]/100, RollingVolatilities[c]], axis=1).max(axis=1), axis=0)

        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(driver, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_"+strategyName+"_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def Dragons(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        #################################################################################################################################
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        RollingVolatilities = pe.dema(localRets, self.LookBacks[localRets.columns], mode="AnnVol").ffill().bfill()
        RollingVolatilities.to_sql(self.SystemName + "_" + strategyName + "_RollingVolatilities",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        #################################################################################################################################
        #EMAs = pe.ema(localRets, nperiods=250)
        #EMAs.to_sql(self.SystemName + "_" + strategyName + "_EMAs", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        #driver = pe.sign(EMAs)
        #################################################################################################################################
        driver = pe.sign(self.DEMAs[localRets.columns])
        #################################################################################################################################
        "FINANCIAL CONDITIONS INDEXES"
        # FCIs = self.IndicatorsDF[["BFCIUS Index", "BFCIEU Index", "BFCIGB Index"]]
        # kernelFCIs = pd.DataFrame(1, index=FCIs.index, columns=FCIs.columns)
        # kernelFCIs[FCIs <= -1.5] = 0
        # for c in driver.columns:
        #    driver[c] *= kernelFCIs["BFCIUS Index"]
        #################### DECISION TREES ###################################################
        DecisionTrees_RV = pd.read_sql("SELECT * FROM DecisionTrees_RV", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        [DT_FilterDF, DT_Aggregated_SigDF] = ML.DecisionTrees_Signal_Filter(localRets, self.IndicatorsDF,DecisionTrees_RV,
                                                                            ["GVZ Index","OVX Index"],
                                                                            BarrierDirections=["Upper", "Upper"]
                                                                            )
        DT_FilterDF.to_sql(self.SystemName + "_" + strategyName + "_DT_FilterDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        DT_Aggregated_SigDF.to_sql(self.SystemName + "_" + strategyName + "_DT_Aggregated_SigDF",sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        "DT Filter Out Final Driver (Signal)"
        #driver *= DT_Aggregated_SigDF

        driver.to_sql(self.SystemName + "_" + strategyName + "_driver_Filtered", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        ######################### RISK PARITY #####################################################
        #for c in driver.columns:
        #    driver[c] = driver[c].div(pd.concat([self.IndicatorsDF["GVZ Index"], self.IndicatorsDF["OVX Index"], RollingVolatilities[c]], axis=1).max(axis=1), axis=0)

        #sDF = driver
        sDF = driver / RollingVolatilities
        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_" + strategyName + "_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    """R&D"""

    def BrotherhoodRnD(self):
        strategyName = inspect.stack()[0][3]
        localRets = pe.dlog(self.df)
        targetPairs = [

        ]
        "Enumerate Asset Occurencies"
        trAssetList = []
        for combo in targetPairs:
            trAssetList.append(combo[0])
            trAssetList.append(combo[1])
        trAssetDF = pd.Series(trAssetList).value_counts()
        "Strategy's Core"
        sRawList = []
        HedgeRatiosList = []
        RVsList = []
        for pair in targetPairs:
            subPairName = pair[0] + "_" + pair[1]
            assetX = localRets[pair[0]]
            assetY = localRets[pair[1]]
            nHR = 250
            nEMA = 250
            nIRDs = 250
            HedgeRatio = pe.S(assetX.rolling(nHR).corr(assetY) * (
                    pe.roller(assetX, np.std, n=nHR) / pe.roller(assetY, np.std, n=nHR)))
            RV = assetX - HedgeRatio * assetY
            HedgeRatiosList.append(pd.DataFrame(HedgeRatio, index=RV.index, columns=[subPairName]))
            RVsList.append(RV)
            subSig = pd.DataFrame(pe.sign(pe.ema(RV, nperiods=nEMA)), index=RV.index, columns=[subPairName])
            #subSig = pd.DataFrame(pe.sign(pe.sign(pe.ema(RV, nperiods=nEMA)) + pe.sign(pe.ema(pe.d(IRDs[subPairName]), nperiods=nIRDs))),index=RV.index, columns=[subPairName])
            sRawList.append(subSig)
        HedgeRatioDF = pd.concat(HedgeRatiosList, axis=1)
        RVsDF = pd.concat(RVsList, axis=1)
        sRaw = pd.concat(sRawList, axis=1)
        RVsDF.columns = sRaw.columns

        #sRaw.to_sql(self.SystemName + "_Brotherhood_sRaw", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        #HedgeRatioDF.to_sql(self.SystemName + "_Brotherhood_HedgeRatioDF", sqlite3.connect("SkeletonDB.db"),if_exists='replace')
        #RVsDF.to_sql(self.SystemName + "_Brotherhood_RVsDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        #sRaw.to_sql(self.SystemName + "_Brotherhood_sRollSharpeFiltered", sqlite3.connect("SkeletonDB.db"),if_exists='replace')

        "First Signal"
        sDF = pe.RVSignalHandler(sRaw, HedgeRatioDF=HedgeRatioDF)

        "Rebase each asset exposure on frequency per pair"
        #for c in sDF.columns:
        #    sDF[c] /= trAssetDF[c]

        "Rebase"
        RebaseOut = StrategiesDeck.ExPostRebase(sDF, localRets, exPostRebaseCompounds="YES", exPostVol="YES")
        sDF = RebaseOut[0]
        for prop in ['compoundScaler', 'volExpander']:
            try:
                RebaseOut[1][prop].to_sql(self.SystemName + "_" + prop, sqlite3.connect("SkeletonDB.db"),
                                          if_exists='replace')
            except Exception as e:
                print(e)
        ##############################################################################################################
        sDF = pe.fd(sDF)
        sDF.to_sql(self.SystemName + "_Brotherhood_sDF", sqlite3.connect("SkeletonDB.db"), if_exists='replace')
        return sDF * self.Leverage

    def UnderMaskPCA(self):
        manifoldTarget = 'PCA'
        localDBconn = sqlite3.connect("SkeletonDB.db")
        localRets = pe.dlog(self.df)
        out = ManSee.gRollingManifold(manifoldTarget, localRets, 50, 3, [0, 1, 2])
        out[0].to_sql('df', localDBconn, if_exists='replace')
        principalCompsDfList = out[1]
        exPostProjectionsList = out[2]
        for k in range(len(principalCompsDfList)):
            principalCompsDfList[k].to_sql(manifoldTarget + '_principalCompsDf_' + str(k), localDBconn, if_exists='replace')
            exPostProjectionsList[k].to_sql(manifoldTarget + '_exPostProjections_' + str(k), localDBconn, if_exists='replace')
        sDF = principalCompsDfList[1]

        ##############################################################################################################
        sDF.to_sql(self.SystemName+"_UnderMaskPCA_sDF", localDBconn, if_exists='replace')
        return sDF * self.Leverage
