import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt
from scipy.stats import norm, t
import time, pickle, inspect
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
import seaborn as sn
from itertools import combinations, permutations
#from PyEurobankBloomberg.PySystems.PyLiveTradingSystems.DataDeck import DataDeck
from pyerb import pyerb as pe
from pyerbML import ML, ManSee
from hurst import compute_Hc
from pykalman import pykalman

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

class StrategiesCrossValidation:

    def __init__(self, sheet, tab, DB):
        self.AlternativeStorageLocation = "C:/Users/panagiotis.papaioann/Desktop/SinceWeHaveLimitedSpace/"
        if DB != "DataDeck.db":
            DB = self.AlternativeStorageLocation + DB
        if tab != "ALL":
            self.Assets = pd.read_excel("AssetsDashboard.xlsx", sheet_name=sheet)[tab].dropna().tolist()
        else:
            self.Assets = []
            for subtab in ["Endurance", "Coast", "Brotherhood", "Shore", "Valley", "Dragons"]:
                subAssets = pd.read_excel("AssetsDashboard.xlsx", sheet_name=sheet)[subtab].dropna().tolist()
                for x in subAssets:
                    self.Assets.append(x)
            self.Assets = list(set(self.Assets))
        self.workConn = sqlite3.connect(DB)
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable',self.workConn).set_index('Point_1', drop=True)
        self.FuturesTable = self.FuturesTable[[x for x in self.FuturesTable.columns if "index" not in x]]
        self.IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck',self.workConn).set_index('date', drop=True)
        self.StrategiesCrossValidationDB = self.AlternativeStorageLocation+"StrategiesCrossValidation.db"
        self.StrategiesCrossValidationConn = sqlite3.connect(self.StrategiesCrossValidationDB)
        self.RnDFolder = "RnD Temp\\"

    def LO(self, **kwargs):

        if 'annualFeeCharged' in kwargs:
            annualFeeCharged = kwargs['annualFeeCharged']
        else:
            annualFeeCharged = 0

        self.StrategyName = "LO"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        raw_rets = pe.dlog(self.df)
        raw_rets[raw_rets == 0] = np.nan
        rets = (raw_rets.copy() - (annualFeeCharged / 100) / 365).fillna(0)
        rawSig = pe.sign(rets.abs())
        ##########################################################################################
        sig = pe.S(rawSig, nperiods=2)
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Endurance(self):

        self.StrategyName = "Endurance"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        fracRets = pe.frac(np.log(self.df), fracOrder=0.5).diff().fillna(0)
        fig, ax = plt.subplots(nrows=2, ncols=1)
        rets.plot(ax=ax[0])
        fracRets.plot(ax=ax[1])
        plt.show()

        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=25) * 100
        #driver = pe.sign(pe.ema(rets, nperiods=250))
        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(fracRets, 'Vol', nIn=25) * 100
        driver = pe.sign(pe.ema(fracRets, nperiods=250))
        #driver = pe.ema(fracRets, nperiods=250)

        driver[driver < 0] = 0
        "FINANCIAL CONDITIONS INDEXES"
        FCIs = self.IndicatorsDF[["BFCIUS+ Index", "BFCIEU Index", "GSUSFCI Index"]]
        FCIs["GSUSFCI Index"] = pe.rollNormalise(FCIs["GSUSFCI Index"], nIn=250)
        kernelFCIs = pd.DataFrame(1, index=FCIs.index, columns=FCIs.columns)
        kernelFCIs[FCIs <= -1.5] = 0
        kernelFCIs["CIUS"] = kernelFCIs["BFCIUS+ Index"] * kernelFCIs["GSUSFCI Index"]

        driver["NQ1 Index"] = driver["NQ1 Index"] * kernelFCIs["CIUS"]
        driver["ES1 Index"] = driver["ES1 Index"] * kernelFCIs["CIUS"]
        driver["DM1 Index"] = driver["DM1 Index"] * kernelFCIs["CIUS"]
        driver["GX1 Index"] = driver["GX1 Index"] * kernelFCIs["BFCIEU Index"]
        driver["CF1 Index"] = driver["CF1 Index"] * kernelFCIs["BFCIEU Index"]
        driver["VG1 Index"] = driver["VG1 Index"] * kernelFCIs["BFCIEU Index"]

        rawSig = driver / RollingVolatilities
        ##########################################################################################
        sig = pe.S(rawSig, nperiods=2)
        #sig.tail(250).plot()
        #plt.show()
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Endurance_csSL(self):
        self.StrategyName = "Endurance_csSL"
        # self.IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck', sqlite3.connect(self.StrategyName + ".db")).set_index('date', drop=True)
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        EMAs = pe.ema(rets, nperiods=250)
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=25) * 100
        # volPremium = RollingVolatilities["NQ1 Index"] - pe.ema(self.IndicatorsDF, nperiods=250)["VIX Index"]

        driver = pe.sign(EMAs)
        driver[driver < 0] = 0

        csRets = pe.cs(rets)
        EvolvingCSMax = csRets.rolling(250).max()
        MaxDivergence = csRets-EvolvingCSMax

        targetAsset = "NQ1 Index"
        fig, ax = plt.subplots(sharex=True, nrows=6, ncols=1)
        csRets[targetAsset].plot(ax=ax[0])
        MaxDivergence[targetAsset].plot(ax=ax[1])
        driver[targetAsset].plot(ax=ax[2])
        pe.cs(pe.S(driver)*rets)[targetAsset].plot(ax=ax[3])
        driver[MaxDivergence < -0.1] = 0
        driver[targetAsset].plot(ax=ax[4])
        pe.cs(pe.S(driver)*rets)[targetAsset].plot(ax=ax[5])
        plt.show()
        time.sleep(3000)

        rawSig = driver / RollingVolatilities
        ##########################################################################################
        sig = pe.S(rawSig, nperiods=2)
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def ExpeditionSingleAsset(self, runMode):
        self.StrategyName = "ExpeditionSingleAsset"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)

        sigList = []
        for selection in rets.columns:
            magicNum = 1
            params = {
                "model": "GPC",
                "HistLag": 0,
                "InputSequenceLength": 250,  # 240 (main) || 25 (Siettos) ||
                "SubHistoryLength": 250,  # 760 (main) || 250 (Siettos) ||
                "SubHistoryTrainingLength": 250 - 5,  # 510 (main) || 250-1 (Siettos) ||
                "Scaler": "Standard",  # Standard
                'Kernel': '0',  # 0 --> Matern, 1 --> Extensive
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }

            if runMode == 'train':
                out = ML.gClassification(rets[selection], params)

                out[0].to_sql('df_predicted_price_train_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[1].to_sql('df_real_price_class_train_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[2].to_sql('df_real_price_train_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[3].to_sql('df_predicted_price_test_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[4].to_sql('df_real_price_class_test_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[5].to_sql('df_real_price_test_DF_' + selection + "_" + str(magicNum),
                              self.StrategiesCrossValidationConn, if_exists='replace')
                out[6].to_sql('df_score_test_DF_' + selection + "_" + str(magicNum), self.StrategiesCrossValidationConn,
                              if_exists='replace')

                modelTrained = out[7]
                pickle.dump(modelTrained, open(
                    "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/ModelsML/" + self.StrategyName + "_" + selection + ".p",
                    "wb"))
                print("Model Picked!")

                sigDF = out[3]

            elif runMode == 'load':
                sigDF = pd.read_sql(
                    "SELECT * FROM 'df_predicted_price_test_DF_" + selection + "_" + str(magicNum) + "'",
                    self.StrategiesCrossValidationConn).set_index('date', drop=True)

                modelTrained = pickle.load(open(
                    "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/ModelsML/" + self.StrategyName + "_" + selection + ".p",
                    "rb"))
                print(modelTrained)
                print(modelTrained.kernel)
                print(modelTrained.kernel_.theta)
                print(modelTrained.log_marginal_likelihood())
                # time.sleep(34000)

            probThr = 0
            probDF = sigDF[["Predicted_Proba_Test_0.0", "Predicted_Proba_Test_1.0", "Predicted_Proba_Test_2.0"]]
            probDF[probDF < probThr] = 0
            sigDF['ProbFilter'] = pe.rs(probDF)
            sigDF.loc[sigDF["Predicted_Test_" + selection] > 1, "Predicted_Test_" + selection] = -1
            # sigDF.loc[sigDF["Predicted_Test_" + selection] > 1, "Predicted_Test_" + selection] = 0
            sigDF[selection] = sigDF["Predicted_Test_" + selection] * sigDF['ProbFilter']

            # fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
            # probDF.plot(ax=ax[0], title="Probas")
            # sigDF['ProbFilter'].plot(ax=ax[1], title='ProbFilter')
            # sigDF[selection].plot(ax=ax[2], title=selection)
            # plt.show()
            # print(sigDF)
            # time.sleep(3000)

            sigList.append(sigDF[selection])

        rawSig = pd.concat(sigList, axis=1).fillna(0)
        rawSig = pe.sign(rawSig)
        rawSig.to_sql(self.StrategyName + '_rawSig', self.StrategiesCrossValidationConn, if_exists='replace')

        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=25) * 100
        rawSig = rawSig / RollingVolatilities
        rawSig.to_sql(self.StrategyName + '_rawSig_RiskParity', self.StrategiesCrossValidationConn, if_exists='replace')

        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Expedition(self, runMode):

        def rollPreds(roll_rets):
            try:
                roll_reframedData = ML.reframeData(roll_rets.values, 1, assetSel)
                # print(roll_rets)
                # print(pd.DataFrame(roll_reframedData[0]))
                # print(pd.DataFrame(roll_reframedData[1]))
                # print(pd.DataFrame(roll_reframedData[2]).T)
                X_train = roll_reframedData[0]
                Y_train = np.sign(roll_reframedData[1])
                Y_train[Y_train > 0] = 0
                Y_train[Y_train < 0] = -1
                model.fit(X_train, Y_train)
                predicted_price_train = model.predict(roll_reframedData[2])
                predicted_price_train_prob = model.predict_proba(roll_reframedData[2])
                subDataList = [roll_rets.index[-1], predicted_price_train[0]]
                for x in predicted_price_train_prob[0]:
                    subDataList.append(x)
                # print(predicted_price_train)
                # print(predicted_price_train_prob)
                # print(subDataList)
                # print("done")
                # time.sleep(3000)
            except Exception as e:
                print(e)
                subDataList = [roll_rets.index[-1], 0, 0, 0, 0]

            return subDataList

        self.StrategyName = "Expedition"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[
            self.Assets]  # .reindex(columns = self.Assets)
        retsRAW = pe.dlog(self.df).fillna(0)
        rets = pe.ema(retsRAW, nperiods=25)
        targetAssetList = ["HYG US Equity", "IEAC LN Equity", "IHYG LN Equity", "LQD US Equity", "EMB US Equity"]
        rets = rets.loc[rets[targetAssetList[0]].ne(0).idxmax():, :]
        modelList = []
        for i in range(len(targetAssetList)):
            mainKernel = 1 * ConstantKernel() + 1 * RBF() + 1 * RationalQuadratic() + 1 * Matern(nu=0.5) + 1 * Matern(
                nu=2.5) + 1 * WhiteKernel()  # + 1*DotProduct() + 1*ExpSineSquared()
            sub_model = GaussianProcessClassifier(kernel=mainKernel, random_state=0)
            modelList.append(sub_model)
        stepper = 250
        assetSel = 0
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")

            model = modelList[assetSel]

            if runMode == 'train':
                startPeriod = 3000  # , stepper
            elif runMode == 'update':
                startPeriod = rets.shape[0] - 5  # last 'x' days

            if runMode in ['train', 'update']:

                PredDataList = []
                for i in tqdm(range(startPeriod, rets.shape[0] + 1)):
                    med_rets = rets.iloc[i - stepper:i, :]
                    "Here I build the regression problem"
                    rolling_Predictions = rollPreds(med_rets)
                    #print(asset, rolling_Predictions)
                    PredDataList.append(rolling_Predictions)
                try:
                    PredsDF = pd.DataFrame(PredDataList, columns=["date", rets.columns[assetSel]] + [str(x) for x in
                                                                                                     list(
                                                                                                         model.classes_)])
                except:
                    PredsDF = pd.DataFrame(PredDataList)

                #PredsDF['date'] = PredsDF['date'].astype(str).str.split(" ").str[0]
                PredsDF = PredsDF.set_index('date', drop=True)
                PredsDF = PredsDF.astype(float)
                PredsDF[["-1.0", "0.0"]] *= 100

                if runMode == 'train':
                    PredsDF.to_sql(self.StrategyName + "_" + subAssetName + '_PredsDF_FirstRun',
                                   self.StrategiesCrossValidationConn, if_exists='replace')

                if runMode == 'update':
                    prev_PredsDF = pd.read_sql(
                        "SELECT * FROM " + self.StrategyName + "_" + subAssetName + "_PredsDF",
                        self.StrategiesCrossValidationConn).set_index('date', drop=True)
                    print(prev_PredsDF)
                    PredsDF = pd.concat([prev_PredsDF, PredsDF])
                    print(PredsDF)
                    PredsDF = PredsDF[~PredsDF.index.duplicated(keep='last')]

                PredsDF.to_sql(self.StrategyName + "_" + subAssetName + '_PredsDF', self.StrategiesCrossValidationConn,
                               if_exists='replace')
                PredsDF.to_csv("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF.csv')
                PredsDF.to_excel("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF.xlsx')

            else:
                pass
                # fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
                # PredsDF[asset].plot(ax=ax[0])
                # PredsDF["-1.0"].plot(ax=ax[1])
                # self.df[asset].iloc[-100:].plot(ax=ax[2])
                # plt.show()

            assetSel += 1

        "Group all together!"
        self.df.iloc[self.df.shape[0] - 50:, :][targetAssetList].to_excel("PortfolioHedge\\targetAssetsDF_Report.xlsx")
        predsList = []
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")
            subPredsDF = pd.read_sql(
                "SELECT * FROM Expedition_" + subAssetName + "_PredsDF", self.StrategiesCrossValidationConn).set_index(
                'date', drop=True)
            subPredsDF.columns = [subPredsDF.columns[0], subPredsDF.columns[0] + " RiskOff Probability",
                                  subPredsDF.columns[0] + " RiskOn Probability"]
            predsList.append(subPredsDF)

        PredsDF = pd.concat(predsList, axis=1)
        RiskOnProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOn' in x]]
        RiskOffProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOff' in x]]
        SignalsDF = PredsDF[[x for x in PredsDF.columns if ('RiskOn' not in x)&('RiskOff' not in x)]]
        RiskOnProbsDF.to_excel("PortfolioHedge\\RiskOnProbsDF.xlsx")
        RiskOffProbsDF.to_excel("PortfolioHedge\\RiskOffProbsDF.xlsx")
        SignalsDF.to_excel("PortfolioHedge\\SignalsDF.xlsx")

        SignalsDF.to_sql(self.StrategyName + '_SignalsDF', self.StrategiesCrossValidationConn, if_exists='replace')

        sig = SignalsDF.copy()
        # sig = rawSig[~sig.index.duplicated(keep='last')]
        sig = pe.S(sig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [retsRAW, sig]

    def ExpeditionFXmacro(self, runMode):

        def rollPreds(roll_rets):
            try:
                roll_reframedData = ML.reframeData(roll_rets.values, 1, assetSel)
                # print(roll_rets)
                # print(pd.DataFrame(roll_reframedData[0]))
                # print(pd.DataFrame(roll_reframedData[1]))
                # print(pd.DataFrame(roll_reframedData[2]).T)
                X_train = roll_reframedData[0]
                Y_train = np.sign(roll_reframedData[1])
                Y_train[Y_train > 0] = 0
                Y_train[Y_train < 0] = -1
                model.fit(X_train, Y_train)
                predicted_price_train = model.predict(roll_reframedData[2])
                predicted_price_train_prob = model.predict_proba(roll_reframedData[2])
                subDataList = [roll_rets.index[-1], predicted_price_train[0]]
                for x in predicted_price_train_prob[0]:
                    subDataList.append(x)
                # print(predicted_price_train)
                # print(predicted_price_train_prob)
                # print(subDataList)
                # print("done")
                # time.sleep(3000)
            except Exception as e:
                print(e)
                subDataList = [roll_rets.index[-1], 0, 0, 0, 0]

            return subDataList

        self.StrategyName = "ExpeditionFXmacro"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[
            self.Assets]  # .reindex(columns = self.Assets)
        retsRAW = pe.dlog(self.df).fillna(0)
        rets = pe.ema(retsRAW, nperiods=25)
        targetAssetList = ["EURUSD Curncy"]
        rets = rets.loc[rets[targetAssetList[0]].ne(0).idxmax():, :]
        modelList = []
        for i in range(len(targetAssetList)):
            mainKernel = 1 * ConstantKernel() + 1 * RBF() + 1 * RationalQuadratic() + 1 * Matern(nu=0.5) + 1 * Matern(
                nu=2.5) + 1 * WhiteKernel()  # + 1*DotProduct() + 1*ExpSineSquared()
            sub_model = GaussianProcessClassifier(kernel=mainKernel, random_state=0)
            modelList.append(sub_model)
        stepper = 250
        assetSel = 0
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")

            model = modelList[assetSel]

            if runMode == 'train':
                startPeriod = 3000  # , stepper
            elif runMode == 'update':
                startPeriod = rets.shape[0] - 5  # last 'x' days

            if runMode in ['train', 'update']:

                PredDataList = []
                for i in tqdm(range(startPeriod, rets.shape[0] + 1)):
                    med_rets = rets.iloc[i - stepper:i, :]
                    "Here I build the regression problem"
                    rolling_Predictions = rollPreds(med_rets)
                    #print(asset, rolling_Predictions)
                    PredDataList.append(rolling_Predictions)
                try:
                    PredsDF = pd.DataFrame(PredDataList, columns=["date", rets.columns[assetSel]] + [str(x) for x in
                                                                                                     list(
                                                                                                         model.classes_)])
                except:
                    PredsDF = pd.DataFrame(PredDataList)

                #PredsDF['date'] = PredsDF['date'].astype(str).str.split(" ").str[0]
                PredsDF = PredsDF.set_index('date', drop=True)
                PredsDF = PredsDF.astype(float)
                PredsDF[["-1.0", "0.0"]] *= 100

                if runMode == 'train':
                    PredsDF.to_sql(self.StrategyName + "_" + subAssetName + '_PredsDF_FirstRun',
                                   self.StrategiesCrossValidationConn, if_exists='replace')

                if runMode == 'update':
                    prev_PredsDF = pd.read_sql(
                        "SELECT * FROM " + self.StrategyName + "_" + subAssetName + "_PredsDF",
                        self.StrategiesCrossValidationConn).set_index('date', drop=True)
                    print(prev_PredsDF)
                    PredsDF = pd.concat([prev_PredsDF, PredsDF])
                    print(PredsDF)
                    PredsDF = PredsDF[~PredsDF.index.duplicated(keep='last')]

                PredsDF.to_sql(self.StrategyName + "_" + subAssetName + '_PredsDF', self.StrategiesCrossValidationConn,
                               if_exists='replace')
                PredsDF.to_csv("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF.csv')
                PredsDF.to_excel("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF.xlsx')

            else:
                pass
                # fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
                # PredsDF[asset].plot(ax=ax[0])
                # PredsDF["-1.0"].plot(ax=ax[1])
                # self.df[asset].iloc[-100:].plot(ax=ax[2])
                # plt.show()

            assetSel += 1

        "Group all together!"
        self.df.iloc[self.df.shape[0] - 50:, :][targetAssetList].to_excel("PortfolioHedge\\targetAssetsDF_Report.xlsx")
        predsList = []
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")
            subPredsDF = pd.read_sql(
                "SELECT * FROM Expedition_" + subAssetName + "_PredsDF", self.StrategiesCrossValidationConn).set_index(
                'date', drop=True)
            subPredsDF.columns = [subPredsDF.columns[0], subPredsDF.columns[0] + " RiskOff Probability",
                                  subPredsDF.columns[0] + " RiskOn Probability"]
            predsList.append(subPredsDF)

        PredsDF = pd.concat(predsList, axis=1)
        RiskOnProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOn' in x]]
        RiskOffProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOff' in x]]
        SignalsDF = PredsDF[[x for x in PredsDF.columns if ('RiskOn' not in x)&('RiskOff' not in x)]]
        RiskOnProbsDF.to_excel("PortfolioHedge\\RiskOnProbsDF.xlsx")
        RiskOffProbsDF.to_excel("PortfolioHedge\\RiskOffProbsDF.xlsx")
        SignalsDF.to_excel("PortfolioHedge\\SignalsDF.xlsx")

        SignalsDF.to_sql(self.StrategyName + '_SignalsDF', self.StrategiesCrossValidationConn, if_exists='replace')

        sig = SignalsDF.copy()
        # sig = rawSig[~sig.index.duplicated(keep='last')]
        sig = pe.S(sig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [retsRAW, sig]

    def EnduranceV4AR(self):
        self.StrategyName = "EnduranceVAR"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        RollingVAR = np.sqrt(252) * pe.rollStatistics(rets, "VAR", nIn=25, alpha=0.01) * 100
        # RollingVAR = np.sqrt(252) * pe.rollCVar(rets, 25, 0.01) * 100
        # rawSig = 1 / RollingVAR
        rawSig = pe.sign(1 / RollingVAR)
        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def EnduranceSkewed(self):
        self.StrategyName = "EnduranceSkewness"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        RollingSkewness = pe.rollStatistics(rets, "Skewness", nIn=25) # EC1 Curncy, CD1 Curncy, OE1 Comdty, Z 1 Index || neg :DU1 Comdty,RX1 Comdty,OE1 Comdty,
        RollingKurtosis = pe.rollStatistics(rets, "Kurtosis", nIn=25)

        RollingSkewness[RollingSkewness > 0] = 0
        RollingSkewness = pe.sign(RollingSkewness)
        RollingKurtosis[RollingKurtosis.abs() < 3] = 0
        RollingKurtosis = pe.sign(RollingKurtosis)

        #driver = RollingSkewness
        driver = RollingKurtosis
        #driver = RollingSkewness * RollingKurtosis
        #driver = pe.sign(pe.ema(rets, nIn=25))
        #driver *= RollingSkewness
        #driver *= RollingSkewness*RollingKurtosis

        RollingSkewness.plot()
        RollingKurtosis.plot()

        RollingVol = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=25) * 100
        rawSig = (driver / RollingVol)
        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Shore(self):
        self.StrategyName = "Shore"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        driver = pe.sign(pe.ema(rets["DX1 Curncy"], nperiods=5))
        driver[driver < 0] = 0
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets["DX1 Curncy"], 'Vol', nIn=5) * 100
        rawSig = pd.DataFrame(1, index=rets.index, columns=rets.columns)
        rawSig["DX1 Curncy"] = 0
        rawSig["EC1 Curncy"] = 0.576
        rawSig["JY1 Curncy"] = 0.136
        rawSig["BP1 Curncy"] = 0.119
        rawSig["CD1 Curncy"] = 0.091
        rawSig["SF1 Curncy"] = 0.042+0.036 # "Getting SEK exposure into Swiss Franc Futures"
        rawSig = (-1) * rawSig.mul((driver * RollingVolatilities).fillna(0), axis=0)
        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def ERB_ML_RiskOffIndicator(self):

        def ProbsToSig(dfIn):
            out = dfIn
            out[out > 50] *= -1
            return out

        self.StrategyName = "ERB_ML_RiskOffIndicator"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        self.RiskOnProbs = pd.read_sql('SELECT * FROM RiskStatus_RiskOnProbsDF', sqlite3.connect("RiskStatus.db")).set_index('index', drop=True)
        driver = ProbsToSig(self.RiskOnProbs)
        #driver[driver < 0] = 0
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=5) * 100

        rawSig = pd.DataFrame(0, index=rets.index, columns=rets.columns)
        for c in rawSig.columns:rawSig[c] = driver.iloc[:,0]

        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Valley(self):
        self.StrategyName = "ValleyRnD"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        localRets = pe.dlog(self.df).fillna(0)
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        #RollingVolatilities = 1
        driver = pe.ema(localRets, nperiods=250).ffill().fillna(0)
        #driver[[x for x in driver.columns if "1 Comdty" in x]] *= 0

        sigRaw = pe.sign(driver)
        sigRaw /= RollingVolatilities
        sig = pe.S(sigRaw, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [localRets, sig]

    def Dragons(self):
        self.StrategyName = "Dragons"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        LookBacks = pd.read_sql("SELECT * FROM LookBacks", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)[rets.columns]
        DEMAs = pd.read_sql("SELECT * FROM DEMAs", sqlite3.connect("DataDeck.db")).set_index('date', drop=True)[rets.columns]
        #RollHurstSpace = "RollHurstAssets_ExpWindow_25"
        RollHurstSpace = "RollHurstAssets_RollWindow_250"
        RollHurst = pd.read_sql("SELECT * FROM "+RollHurstSpace, sqlite3.connect(self.AlternativeStorageLocation+"RollHurst.db")).set_index('date', drop=True)[rets.columns]
        #RollHurst.plot()
        #plt.show()
        #LookBacks.plot()
        #plt.show()

        driver = pe.sign(DEMAs)
        driver = driver * pe.sign(RollHurst-0.5)

        #mlModel = ["RFC", {"mainConfig": "MLdP0"}]
        # mlModel = ["GPC", {"mainConfig":"GekkoMainKernel"}]
        # mlModel = ["GPC", {"mainConfig":"GekkoMainKernelExtended"}]
        #mlModel = ["GPC", {"mainConfig":"MRMaternKernel"}]
        mlModel = ["GPC", {"mainConfig":"TrendMaternKernel"}]

        retsToML = DEMAs#.tail(50)
        runMode = "no" #train, update, read
        RetsSmoothedFlag = True
        FeatureSelector = ["SelfRegulate", 10]
        MemoryDepthIn = 25

        if runMode in ["train", "update"]:
            ML.Roll_SML_Predict(retsToML, runMode, StrategyName=self.StrategyName, workConn=self.StrategiesCrossValidationDB,
                                targetAssetList=list(rets.columns),
                                RetsSmoothed=RetsSmoothedFlag,
                                MLModel=mlModel,
                                FeaturesRegulators=FeatureSelector,
                                MemoryDepth=MemoryDepthIn)

        #driver = pd.read_sql("SELECT * FROM "+self.StrategyName+"_PredsMat", self.StrategiesCrossValidationConn).set_index('date', drop=True)
        #driver = driver.fillna(0)
        #driver = pe.sign(DEMAs+driver)
        #driver = DEMAs * driver

        #RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=250) * 100
        RollingVolatilities = pe.dema(rets, LookBacks, mode="AnnVol")
        #RollingVolatilities.plot()
        #plt.show()
        rawSig = driver / RollingVolatilities

        #ExpandingSharpe = np.sqrt(252) * pe.expander(pe.fd(rawSig*rets), pe.sharpe, 25)
        #ExpandingSharpe.plot()
        #plt.show()
        #rawSig[ExpandingSharpe < 0] = 0
        #rawSig[ExpandingSharpe < 0] *= -1
        #ExpandingSharpe.plot()
        #plt.show()
        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def DragonsRnD(self):
        self.StrategyName = "Dragons"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        localRets = pe.dlog(self.df).fillna(0)

        modelRets = pd.DataFrame(0, index=localRets.index, columns=localRets.columns)
        for c in tqdm(localRets.columns):
            kf = pykalman.KalmanFilter(transition_matrices=[1],  # The value for At. It is a random walk so is set to 1.0
                                       observation_matrices=[1],  # The value for Ht.
                                       initial_state_mean=0,  # Any initial value. It will converge to the true state value.
                                       initial_state_covariance=1,
                                       # Sigma value for the Qt in Equation (1) the Gaussian distribution
                                       observation_covariance=1,
                                       # Sigma value for the Rt in Equation (2) the Gaussian distribution
                                       transition_covariance=0.01)  # A small turbulence in the random walk parameter 1.0
            state_means, _ = kf.filter(localRets[c])
            modelRets[c] = state_means

        nHR = 250
        #RVs = pe.RV(localRets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Roll", n=nHR)
        #RVs = pe.RV(localRets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Expand", n=nHR)
        RVs = pe.RV(localRets, HedgeRatioConnectionMode="Spreads", mode="Linear", n=nHR)
        #RVs = pe.RV(localRets, HedgeRatioConnectionMode="Baskets", mode="HedgeRatioPnL_Expand", n=nHR)
        #RVs = pe.RV(localRets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Expand", n=nHR, RVspace="specificDriverclassicPermutations_TU1 Comdty")

        localRets = pd.concat([localRets, RVs], axis=1).fillna(0)

        #H = lambda x: compute_Hc(x)[0]
        #HurstDF = pd.DataFrame(None, index=localRets.index, columns=localRets.columns)
        #for c in tqdm(localRets.columns):
        #    print(c)
            #HurstDF[c] = localRets[c].rolling(50).apply(H)-0.6
        #    HurstDF[c] = self.df[c].rolling(250).apply(H)-0.6
            #HurstDF[c] = pe.sign(localRets[c].rolling(50).apply(H)-localRets[c].rolling(100).apply(H))
            #HurstDF[c] = pe.sign(self.df[c].rolling(50).apply(H)-self.df[c].rolling(100).apply(H))
        #HurstDF.plot()
        #plt.show()
        #driverEMA = pe.sign(pe.ema(localRets, nperiods=25))
        #driverEMA = pe.sign(pe.ema(localRets, nperiods=250))
        driverEMA = pe.sign(pe.ema(modelRets, nperiods=250))
        #driverEMA = pe.sign(modelRets)
        #driverEMA[HurstDF > 0] *= 0

        RollingVolatilities = (np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=25) * 100).bfill()
        sigRaw = driverEMA #* HurstDF
        #sigRaw[["ER1 Comdty","ED1 Comdty","FF1 Comdty"]] = 0
        #sigRaw /= RollingVolatilities

        sig = pe.S(sigRaw, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [localRets, sig]

    def Lumen(self):
        self.StrategyName = inspect.stack()[0][3]
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)
        #(self.df["NG1 Comdty"]-self.df["NG2 Comdty"]).plot()
        #(self.df["CL1 Comdty"]-self.df["CL2 Comdty"]).plot()
        #plt.show()
        MoreFuturesCurvePoints = pe.getMoreFuturesCurvePoints(self.Assets, self.FuturesTable, [2,3])

        rets = pe.dlog(self.df[MoreFuturesCurvePoints]).fillna(0)
        #pe.cs(rets).plot()
        #pe.cs(rets[["SFR1 Comdty", "SFR2 Comdty", "SFR3 Comdty", "ED1 Comdty", "ED2 Comdty", "ED3 Comdty"]]).plot()
        #plt.show()

        #DynamicEMA = pe.ema(rets, nperiods=3).dropna()
        #DynamicEMA = pe.ema(rets, nperiods=25).dropna()
        #DynamicEMA = pe.ema(rets, nperiods=250).dropna()

        nHR = 250
        RVs = pe.RV(rets, HedgeRatioConnectionMode="Spreads", mode="Linear", n=nHR)
        #RVs = pe.RV(rets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Roll", n=nHR)
        #RVs = pe.RV(rets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Expand", n=nHR)
        #RVs = pe.RV(rets, HedgeRatioConnectionMode="Baskets", mode="HedgeRatioPnL_Expand", n=nHR)
        #RVs = pe.RV(rets, HedgeRatioConnectionMode="Spreads", mode="HedgeRatioPnL_Expand", n=nHR, RVspace="specificDriverclassicPermutations_ZB2 Comdty")

        RVs = RVs[[x for x in RVs.columns if x.split("_")[0].replace("Comdty","")[:2]==x.split("_")[1].replace("Comdty","")[:2]]]
        pe.cs(RVs[["ES1 Index_ES2 Index"]]).plot()
        #pe.cs(RVs[["NG1 Comdty_NG2 Comdty"]]).plot()
        #plt.show()
        #print(RVs.columns)
        #time.sleep(3000)

        #rets = pd.concat([rets, RVs], axis=1).fillna(0)
        rets = RVs.fillna(0)

        RollingVolatilities = 1
        #RollingVolatilities = (np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=250) * 100).bfill()
        #RollingVolatilities[RollingVolatilities == 0] = 1

        #driverEMA = pe.sign(pe.ema(rets, nperiods=12))
        #driverEMA = pe.sign(pe.ema(rets, nperiods=25))
        driverEMA = pe.sign(pe.ema(rets, nperiods=250))

        sigRaw = driverEMA
        #sigRaw = driverEMA / RollingVolatilities

        sig = pe.S(sigRaw, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def Brotherhood(self):
        self.StrategyName = inspect.stack()[0][3]
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        localRets = pe.dlog(self.df)
        RiskReversals = self.IndicatorsDF.fillna(0)[[x for x in self.IndicatorsDF.columns if "25R" in x]]
        signRiskReversals = pe.ema(RiskReversals, nperiods=3)
        driver = pe.sign(pe.ema(localRets, nperiods=25))
        RollingKurtosis = pe.rollStatistics(localRets, 'Kurtosis', nIn=250)
        RollingKurtosis[RollingKurtosis.abs() > 3] = 0
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100

        for c in driver.columns:
            driver[c] = driver["DX1 Curncy"]
        driver["DX1 Curncy"] = 0
        driver["EC1 Curncy"] *= 0.576 #* signRiskReversals["EURUSD25R1W Curncy"]
        driver["JY1 Curncy"] *= 0.136 #* signRiskReversals["USDJPY25R1W Curncy"]*(-1)
        driver["BP1 Curncy"] *= 0.119 #* signRiskReversals["BRLUSD25R1W Curncy"]
        driver["CD1 Curncy"] *= 0.091 #* signRiskReversals["USDCAD25R1W Curncy"]*(-1)
        driver["SF1 Curncy"] *= (0.042 + 0.036) #* signRiskReversals["USDCHF25R1W Curncy"]*(-1) # "Getting SEK exposure into Swiss Franc Futures"
        driver *= (-1) * pe.rowStoch(RollingKurtosis)

        sDF = driver / RollingVolatilities
        # sDF = driver.div(RollingVolatilities["DX1 Curncy"],axis=0)

        sig = pe.S(sDF, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [localRets, sig]

    def Fidei(self):
        self.StrategyName = "Fidei"
        localRets = pd.read_sql('SELECT * FROM rets', sqlite3.connect("DataDeck.db")).set_index('date', drop=True)
        AllDF = pd.concat([localRets[["ES1 Index", "VG1 Index", "SFR1 Comdty", "TY1 Comdty", "FV1 Comdty"]], self.IndicatorsDF[["M2US000$ Index","MXWO000G Index","MAEUMMT Index","M2US000V Index","M2US000G Index"]]], axis=1).sort_index()
        AllDF = AllDF.tail(250)
        #print(AllDF)
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        print(RollingVolatilities)
        pd.concat([RollingVolatilities[["ED1 Comdty", "SFR1 Comdty"]], self.IndicatorsDF["EUNS01 Curncy"]/100], axis=1).sort_index().plot()
        plt.show()

        out = pe.RollMetric(AllDF, metric="MI", RollMode="RollWindow", st=25)
        print([x.iloc[-1] for x in out])
        fig, ax = plt.subplots(nrows=5, ncols=1)
        out[0].plot(ax=ax[0])
        out[1].plot(ax=ax[1])
        out[2].plot(ax=ax[2])
        out[3].plot(ax=ax[3])
        out[4].plot(ax=ax[4])
        plt.show()
        time.sleep(3000)

        sigList = []
        sigDF = pd.concat(sigList, axis=1)

        pnlList = []
        for c in sigDF.columns:
            pnl = pe.S(sigDF[c], nperiods=2) * localRets[assetPair[1]]
            pnlList.append(pnl)

        pnlDF = pd.concat(pnlList, axis=1)
        pnlDF["TOTAL"] = pe.rs(pnlDF)
        sh = np.sqrt(252) * pe.sharpe(pnlDF)
        print(sh)

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        splitThresholdDF.plot(ax=ax[0])
        pe.cs(pnlDF).plot(ax=ax[1])
        plt.show()

        sigRaw = localDF

        sig = pe.S(sigRaw, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [localRets, sig]

    def Coast(self):
        self.StrategyName = "Coast"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        driver = pe.sign(pe.ema(rets, nperiods=250))

        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=25) * 100
        rawSig = driver / RollingVolatilities
        sig = pe.S(rawSig, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def CoastRnD(self):
        self.StrategyName = "CoastRnD"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        Rets_singleAssets = pe.dlog(self.df)

        #RVs = pe.RV(Rets_singleAssets)
        RVs = pe.RV(Rets_singleAssets, mode="HedgeRatioPnL_Roll", n=250)
        #RVs = pe.RV(localRets, mode="HedgeRatioPnL_Roll", RVspace="specificPairs", targetPairs=targetPairsIn)
        #RVs = pe.RV(localRets, RVspace="specificDriverclassicPermutations_PE1 Curncy") #mode="HedgeRatioPnL_Roll"
        #RVs = pe.RV(localRets, mode="HedgeRatioPnL_Roll", RVspace="specificDriverclassicPermutations_BR1 Curncy")
        #RVs = pe.RV(localRets, mode="HedgeRatioPnL_Roll", RVspace="specificLaggerclassicPermutations_DX1 Curncy")
        RVs.to_sql(self.StrategyName+'_RVs', self.StrategiesCrossValidationConn, if_exists='replace')

        #RVs = pd.read_sql('SELECT * FROM '+self.StrategyName+'_RVs', self.StrategiesCrossValidationConn).set_index('date', drop=True)

        RVs = pe.fd(RVs)

        shRaw = np.sqrt(252) * pe.sharpe(RVs)
        shRaw.to_sql(self.StrategyName+'_RVs_shRaw', self.StrategiesCrossValidationConn, if_exists='replace')

        #localRets = Rets_singleAssets
        #localRets = RVs
        localRets = pd.concat([Rets_singleAssets, RVs], axis=1)
        #localRets = RVs[[x for x in RVs.columns if x.split("_")[0]=="BR1 Curncy"]]
        #localRets = RVs[[x for x in RVs.columns if x.split("_")[0]=="BR1 Curncy"]]
        #localRets = RVs[["BR1 Curncy_PE1 Curncy","BR1 Curncy_RA1 Curncy","BR1 Curncy_JY1 Curncy"]]
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(localRets, 'Vol', nIn=250) * 100
        "Directional Signal : Trend/Mean Reversion"
        driver = pe.sign(pe.ema(localRets, nperiods=5)).fillna(0)

        "Interest Rates Differentials (Spreads) & Carry Total Return Indexes"
        mCarryIndexes = self.IndicatorsDF[[x for x in self.IndicatorsDF.columns if "CR Curncy" in x]]
        mCarryIndexes.columns = [(pe.getFutureTicker(x.replace("CR Curncy", "")[:3]) + "_" + pe.getFutureTicker(
            x.replace("CR Curncy", "")[-3:])).replace("_USD", "") for x in mCarryIndexes.columns]
        mIRDs = self.IndicatorsDF[[x for x in self.IndicatorsDF.columns if "IS Curncy" in x]]
        mIRDs.columns = [(pe.getFutureTicker(x.replace("IS Curncy", "")[:3]) + "_" + pe.getFutureTicker(
            x.replace("IS Curncy", "")[-3:])).replace("_USD", "") for x in mIRDs.columns]
        n1 = 5
        n2 = 5
        sigList = [
            ### TREND ###
            [driver,"A"],
            [pe.d(mIRDs),"B"],
            [pe.ema(pe.d(mIRDs), nperiods=n1),"C"],
            [pe.sign(pe.ema(pe.d(mIRDs), nperiods=n1)),"D"],
            [pe.d(mCarryIndexes),"E"],
            [pe.ema(pe.d(mCarryIndexes), nperiods=n1),"F"],
            [pe.sign(pe.ema(pe.d(mCarryIndexes), nperiods=n1)),"G"],
            ### COMBOS TREND ###
            [pe.sign(driver+pe.sign(pe.ema(pe.d(mIRDs), nperiods=n1))), "H"], #BEST, no value if lag changed to 25,50
            [pe.sign(driver+pe.sign(pe.ema(pe.d(mCarryIndexes), nperiods=n1))), "I"],
            [pe.sign(pe.sign(driver+pe.sign(pe.ema(pe.d(mIRDs), nperiods=n1)))+pe.sign(pe.ema(pe.d(mCarryIndexes), nperiods=n1))), "J"],
            [(1/3)*driver+(1/3)*pe.sign(pe.ema(pe.d(mIRDs), nperiods=n1))+(1/3)*pe.sign(pe.ema(pe.d(mCarryIndexes), nperiods=n1)), "K"],
            ### MR ###
            [pe.d(pe.d(mIRDs)), "L"],
            [pe.ema(pe.d(pe.d(mIRDs)), nperiods=n2), "M"],
            [pe.sign(pe.ema(pe.d(pe.d(mIRDs)), nperiods=n2)), "N"], # CHF
            [pe.d(pe.d(mCarryIndexes)), "O"],
            [pe.ema(pe.d(pe.d(mCarryIndexes)), nperiods=n2), "P"],
            [pe.sign(pe.ema(pe.d(pe.d(mCarryIndexes)), nperiods=n2)), "Q"],
            [pe.sign(pe.sign(pe.ema(pe.d(pe.d(mIRDs)), nperiods=n2))+pe.sign(pe.ema(pe.d(pe.d(mCarryIndexes)), nperiods=n2))), "R"], #SOME JPY PAIRS
            ## TAILOR EXPANSIONS
            [pe.sign(pe.ema(pe.d(mIRDs), nperiods=n2))+0.5*pe.sign(pe.ema(pe.d(pe.d(mIRDs)), nperiods=n2)), "S"],
            [pe.sign(driver+pe.sign(pe.ema(pe.d(mIRDs), nperiods=n2)))+0.5*pe.sign(pe.ema(pe.d(pe.d(mIRDs)), nperiods=n2)), "T"],
        ]

        shList = []
        for tradingSignal in tqdm(sigList):
            "Long Only"
            #tradingSignal[0][tradingSignal[0] < 0] = 0
            #pnl = (pe.S(tradingSignal[0], nperiods=2) * localRets)
            pnl = (pe.S(tradingSignal[0]/RollingVolatilities, nperiods=2) * localRets)
            pnl["TOTAL"] = pe.rs(pnl)
            sh = np.sqrt(252) * pe.sharpe(pnl)
            sh.name = tradingSignal[1]
            shList.append(sh)

        shDF = pd.concat(shList, axis=1)#.drop(Rets_singleAssets.columns)
        shDF.to_sql(self.StrategyName+'_shDF', self.StrategiesCrossValidationConn, if_exists='replace')
        sigRaw = sigList[0][0]
        sig = pe.S(sigRaw, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [RVs, sig]

    def ProphecyPCA(self, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'run'
        if 'EVRthr' in kwargs:
            EVRthr = kwargs['EVRthr']
        else:
            EVRthr = 0.9
        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'ExpWindow'
        if 'RollWindowLength' in kwargs:
            RollWindowLength = kwargs['RollWindowLength']
        else:
            RollWindowLength = 250

        if 'targetProjection' in kwargs:
            targetProjection = kwargs['targetProjection']
        else:
            targetProjection = 'Last'

        self.StrategyName = "ProphecyPCA"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)
        RollingVolatilities = np.sqrt(252) * pe.rollStatistics(rets, 'Vol', nIn=250) * 100
        """
        selList = [
            ["M2US000G Index",3],
            ["M2US000V Index",3],
            ["M2US000$ Index",3],
            ["US0003M Index",1],
            ["US0006M Index",1],
            ["USSW2 Curncy",1],
            ["USSW5 Curncy",1],
            ["USSW10 Curncy",1],
            #["USGG2YR Index",1],
            #["USGG5YR Index",1],
            #["USGG10YR Index",1],
            ["USSWIT1 Curncy",1],
            #["USSWIT2 Curncy",1],
            #["USSWIT5 Curncy",1],
            #["USSWIT10 Curncy",1],
            #["M8EU000G Index",2],
            #["MXEU000V Index",2],
            #["MAEUMMT Index",2],
            #["EUR003M Index",1],
            #["EUR006M Index",1],
            #["EUSA10 Curncy",1],
            #["EUSA2 Curncy",1],
            #["EUSA5 Curncy",1],
            #["GTEUR2Y Govt",1],
            #["GTEUR5Y Govt",1],
            #["GTEUR10Y Govt",1],
            #["EUSWI1 Curncy",1],
            #["EUSWI2 Curncy",1],
            #["EUSWI5 Curncy",1],
            #["EUSWI10 Curncy",1]
            ]
        self.localIndicators = self.IndicatorsDF[[x[0] for x in selList]]
        for item in selList:
            if item[1] == 1:
                self.localIndicators[item[0]] = pe.d(self.localIndicators[item[0]]/100)
            elif item[1] == 2:
                self.localIndicators[item[0]] = pe.dlog(self.localIndicators[item[0]])
            elif item[1] == 3:
                self.localIndicators[item[0]] = pe.cs(pe.dlog(self.localIndicators[item[0]]))
        self.localIndicators = pe.fd(self.localIndicators).fillna(0)
        AllDF = pd.concat([rets, self.localIndicators], axis=1)
        """
        #fig, ax = plt.subplots(nrows=2, ncols=1)
        #self.localIndicators.plot(ax=ax[0])
        #pe.rollNormalise(self.localIndicators).plot(ax=ax[1])
        #pe.cs(self.localIndicators).plot(ax=ax[0])
        #pe.cs(pe.rp(self.localIndicators)).plot(ax=ax[1])
        #plt.show()

        if mode == 'run':
            out = ManSee.gRollingManifold("PCA", rets, rets.shape[1]+1, len(rets.columns), range(len(rets.columns)), RollMode='ExpWindow')
            #out = ManSee.gRollingManifold("PCA", self.localIndicators, RollWindowLength, len(self.localIndicators.columns),range(len(self.localIndicators.columns)), RollMode=RollMode)
            out[0].to_sql('df', self.StrategiesCrossValidationConn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            out[3].to_sql(self.StrategyName +'_ExplainedVarianceRatio', self.StrategiesCrossValidationConn, if_exists='replace')
            out[4].to_sql(self.StrategyName +'_LamdasDF', self.StrategiesCrossValidationConn, if_exists='replace')
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(self.StrategyName + '_principalCompsDf_' + str(k),
                                               self.StrategiesCrossValidationConn,
                                               if_exists='replace')
                exPostProjectionsList[k].to_sql(self.StrategyName + '_exPostProjections_' + str(k),
                                                self.StrategiesCrossValidationConn,
                                                if_exists='replace')
            pd.concat(exPostProjectionsList, axis=1).to_sql(self.StrategyName + '_exPostProjectionsDF',
                                                self.StrategiesCrossValidationConn,
                                                if_exists='replace')

        exPostProjectionsDF = pd.read_sql('SELECT * FROM ' + self.StrategyName + '_exPostProjectionsDF',self.StrategiesCrossValidationConn).set_index('date', drop=True)
        #exPostProjectionsDF = pd.concat([exPostProjectionsDF, pe.RV(exPostProjectionsDF, mode="Baskets")], axis=1)
        CorrMatDF = pd.concat([exPostProjectionsDF, rets], axis=1).corr()
        CorrMatDF.to_sql(self.StrategyName + '_CorrMatDF',self.StrategiesCrossValidationConn,if_exists='replace')

        foreCastProjections = (-1) * pe.S(pe.sign(pe.ema(exPostProjectionsDF, nperiods=25)), nperiods=1)*exPostProjectionsDF
        print(np.sqrt(252) * pe.sharpe(exPostProjectionsDF))
        #foreCastProjections["GG0"] = exPostProjectionsDF.iloc[:,3]-exPostProjectionsDF.iloc[:,2]-exPostProjectionsDF.iloc[:,9]-exPostProjectionsDF.iloc[:,12]
        #foreCastProjections["GG1"] = pe.rs(foreCastProjections.iloc[:,0:3])
        sh = np.sqrt(252) * pe.sharpe(foreCastProjections)
        print(sh)

        optShSelector = [x for x in range(len(sh))]
        #optShSelector = [x for x in range(len(sh)) if abs(sh.iloc[x]) > 0.5]

        optProjections = foreCastProjections.iloc[:, optShSelector]
        print("optProjections Sharpe = ", np.sqrt(252) * pe.sharpe(pe.rs(optProjections)))
        #rollSharpesDF = np.sqrt(252) * pe.rollStatistics(optProjections, 'Sharpe', nIn=250)
        rollSharpesDF = np.sqrt(252) * pe.expander(optProjections, pe.sharpe, n=25)
        rollSharpesDF.plot()
        plt.show()

        rollSharpesDF[pe.S(rollSharpesDF).fillna(0).abs() < 0.5] = 0

        optProjections *= pe.sign(rollSharpesDF)
        print("optProjections Sharpe (opt exp Sharpe) = ", np.sqrt(252) * pe.sharpe(pe.rs(optProjections)))

        fig, ax = plt.subplots(nrows=3, ncols=1)
        pe.cs(exPostProjectionsDF.iloc[:,optShSelector]).plot(ax=ax[0])
        pe.cs(optProjections).plot(ax=ax[1])
        pe.cs(pe.rs(optProjections)).plot(ax=ax[2])
        plt.show()

        LambdasDF = pd.read_sql('SELECT * FROM ' + self.StrategyName +'_LamdasDF',self.StrategiesCrossValidationConn).set_index('date', drop=True)
        LambdasDF = pe.rowStoch(LambdasDF)
        #LambdasDF.plot()
        #plt.show()

        if targetProjection != "EVR":
            csEVR = pd.DataFrame(1, index=LambdasDF.index, columns=LambdasDF.columns)
        else:
            targetProjection = range(len(LambdasDF.columns))
            EVR = pd.read_sql('SELECT * FROM ' + self.StrategyName + '_ExplainedVarianceRatio',
                              self.StrategiesCrossValidationConn).set_index('date', drop=True)
            csEVR = EVR.cumsum(axis=1)
            csEVR[csEVR >= EVRthr] = 0
            csEVR = pe.sign(csEVR)
            csEVR.to_sql(self.StrategyName + '_csEVR', self.StrategiesCrossValidationConn, if_exists='replace')
            #time.sleep(3000)
        print("targetProjection = ", targetProjection)
        targetProjectionDF = pd.read_sql(
                'SELECT * FROM ' + self.StrategyName + '_principalCompsDf_' + str(targetProjection[0]),
                self.StrategiesCrossValidationConn).set_index('date', drop=True).fillna(0).mul(csEVR.iloc[:,0], axis=0).div(LambdasDF.iloc[:,0], axis=0)
        for t in targetProjection[1:]:
            targetProjectionDF += pd.read_sql(
                'SELECT * FROM ' + self.StrategyName + '_principalCompsDf_' + str(t),
                self.StrategiesCrossValidationConn).set_index('date', drop=True).fillna(0).mul(csEVR.iloc[:,t], axis=0).div(LambdasDF.iloc[:,t], axis=0)

        # Build Projections from loadings
        fig, ax = plt.subplots(nrows=2, ncols=1)
        targetProjectionDF.plot(ax=ax[0])
        ax[0].legend(loc='upper left')
        targetProjectionDF.abs().plot(ax=ax[1])
        ax[1].legend(loc='upper left')
        #plt.show()

        #targetProjectionDF = targetProjectionDF.abs()

        #slowManifoldDriver = pe.sign(pe.ema(pe.rs(targetProjectionDF * rets), nperiods=25))# * (-1)
        #targetProjectionDF = targetProjectionDF.div(slowManifoldDriver,axis=0)
        #targetProjectionDF = pe.sign(targetProjectionDF)
        driver = pe.sign(pe.ema(rets, nperiods=25))
        """
        driver = pd.DataFrame(0, index=rets.index, columns=rets.columns)
        EMA = pd.DataFrame(1, index=rets.index, columns=rets.columns)
        #EMA = pe.sign(pe.ema(rets, nperiods=250))/RollingVolatilities
        for c in [
            ["NQ1 Index","M2US000G Index",1, "VXN Index"],#M2US000G
            ["ES1 Index","M2US000$ Index",1, "VIX Index"],#M2US000$
            ["DM1 Index","M2US000V Index",1, "VXD Index"],
            ["TY1 Comdty","USGG10YR Index",1, "VIX Index"],#US0003M,USGG2YR
            ["FV1 Comdty","USGG5YR Index",1, "VIX Index"],
            ["TU1 Comdty","USGG2YR Index",1, "VIX Index"],
            ["RX1 Comdty","US0003M Index",1, "VIX Index"],#US0003M,USGG2YR
            ["OE1 Comdty","US0003M Index",1, "VIX Index"],
            ["DU1 Comdty","US0003M Index",1, "VIX Index"],
        ]:
            #out = pe.roll_OLS_Regress(AllDF, c[1], X=[c[0]])
            #fig, ax = plt.subplots(nrows=2, ncols=1)
            #out[0].plot(ax=ax[0])
            #pe.cs(out[1]).plot(ax=ax[1])
            #plt.show()

            driver[c[0]] = EMA[c[0]]*c[2]*targetProjectionDF[c[1]]#/(pd.concat([self.IndicatorsDF[c[3]], RollingVolatilities[c[0]]],axis=1).max(axis=1))
            driver[c[0]] = EMA[c[0]]*c[2]*targetProjectionDF[c[1]]#/(pd.concat([self.IndicatorsDF[c[3]], RollingVolatilities[c[0]]],axis=1).max(axis=1))
            driver[c[0]] = EMA[c[0]]*c[2]*targetProjectionDF[c[1]]#/(pd.concat([self.IndicatorsDF[c[3]], RollingVolatilities[c[0]]],axis=1).max(axis=1))
        """
        driver = driver.fillna(0)
        #print(driver)

        #pnl = pe.cs(pe.S(driver, nperiods=2)*rets)
        #pnl["TOTAL"]=pe.rs(pnl)
        #pnl.plot()
        #plt.show()
        sig = pe.S(driver, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def ProphecyBeta(self, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'run'

        if 'BetaMode' in kwargs:
            BetaMode = kwargs['BetaMode']
        else:
            BetaMode = 'Beta'

        if 'targetProjectionList' in kwargs:
            targetProjectionList = kwargs['targetProjectionList']
        else:
            targetProjectionList = [0]

        self.StrategyName = "Prophecy" + BetaMode
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)

        if mode == 'run':
            out = ManSee.gRollingManifold(BetaMode, rets, 25, len(rets.columns), range(len(rets.columns)))
            out[0].to_sql('df', self.StrategiesCrossValidationConn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(self.StrategyName + '_principalCompsDf_' + str(k),
                                               self.StrategiesCrossValidationConn,
                                               if_exists='replace')
                exPostProjectionsList[k].to_sql(self.StrategyName + '_exPostProjections_' + str(k),
                                                self.StrategiesCrossValidationConn,
                                                if_exists='replace')

            targetProjectionDF = principalCompsDfList[targetProjectionList[0]]
            for targetPr in targetProjectionList[1:]:
                targetProjectionDF += principalCompsDfList[targetPr]

        else:
            targetProjectionDF = pd.read_sql(
                'SELECT * FROM ' + self.StrategyName + '_principalCompsDf_' + str(targetProjectionList[0]),
                self.StrategiesCrossValidationConn).set_index('date', drop=True)
            for targetPr in targetProjectionList[1:]:
                targetProjectionDF += pd.read_sql(
                    'SELECT * FROM ' + self.StrategyName + '_principalCompsDf_' + str(targetPr),
                    self.StrategiesCrossValidationConn).set_index('date', drop=True)

        sig = pe.S(targetProjectionDF, nperiods=2)
        # sig = pe.sign(sig)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def ProphecyCustom(self, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 'run'

        if 'targetProjectionNum' in kwargs:
            targetProjectionNum = kwargs['targetProjectionNum']
        else:
            targetProjectionNum = 0

        self.StrategyName = "ProphecyCustom"
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)[self.Assets]
        rets = pe.dlog(self.df).fillna(0)

        if mode == 'run':
            "CustomMetricStatistic : Vol, Skewness, Kurtosis, VAR, CVAR, Sharpe"
            "CustomMetric : euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean"
            out = ManSee.gRollingManifold("CustomMetric", rets, 25, len(rets.columns), range(len(rets.columns)),
                                      CustomMetricStatistic="Sharpe", CustomMetric="manhattan")
            out[0].to_sql('df', self.StrategiesCrossValidationConn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(self.StrategyName + '_principalCompsDf_' + str(k),
                                               self.StrategiesCrossValidationConn,
                                               if_exists='replace')
                exPostProjectionsList[k].to_sql(self.StrategyName + '_exPostProjections_' + str(k),
                                                self.StrategiesCrossValidationConn,
                                                if_exists='replace')
            targetProjection = principalCompsDfList[targetProjectionNum]
        else:
            targetProjection = pd.read_sql(
                'SELECT * FROM ' + self.StrategyName + '_principalCompsDf_' + str(targetProjectionNum),
                self.StrategiesCrossValidationConn).set_index('date', drop=True)

        sig = pe.S(targetProjection, nperiods=2)
        ##########################################################################################
        sig = pe.fd(sig)
        self.out = [rets, sig]

    def pnlReport(self, **kwargs):
        if 'specificPeriod' in kwargs:
            specificPeriod = kwargs['specificPeriod']
        else:
            specificPeriod = None

        #try:
        ########### PROFITABILITY #########
        rets = self.out[0]
        sig = self.out[1]
        pnl = (sig * rets).fillna(0)
        if specificPeriod is not None:
            pnl = pnl.loc[specificPeriod[0]:specificPeriod[1], :]
        shPnl = np.sqrt(252) * pe.sharpe(pnl).sort_values(ascending=False)
        try:
            shPnl.index = ['("'+'","'.join(x.split("_"))+'")' for x in shPnl.index if "_" in x]
        except:
            pass
        RollingSharpeAnnualised = (np.sqrt(252) * pe.roller(pnl, pe.sharpe, n=250))
        print(shPnl)
        rspnl = pe.rs(pnl)
        print(np.sqrt(252) * pe.sharpe(rspnl))

        "KEEP THIS AS A QUICK DEBUGGER on plots ..."
        #fig, ax = plt.subplots(nrows=4,ncols=1)
        #rets.plot(ax=ax[0])
        #sig.plot(ax=ax[1])
        #pe.cs((sig * rets).fillna(0)).plot(ax=ax[2])
        #pe.cs(pe.rs((sig * rets).fillna(0))).plot(ax=ax[3])
        #plt.show()

        "CORRELATIONS on CONTRIBUTIONS"
        try:
            corrDF = pnl.corr()
            corrDF.to_sql(self.StrategyName + "_Contributions_CorrelationMatrix",self.StrategiesCrossValidationConn, if_exists='replace')
        except Exception as e:
            print(e)
        RollingRsSharpeAnnualised = (np.sqrt(252) * pe.roller(rspnl, pe.sharpe, n=250))
        try:
            rets.to_sql(self.StrategyName + "_rets", self.StrategiesCrossValidationConn, if_exists='replace')
            sig.to_sql(self.StrategyName + "_sig", self.StrategiesCrossValidationConn, if_exists='replace')
            pnl.to_sql(self.StrategyName + "_pnl", self.StrategiesCrossValidationConn, if_exists='replace')
            RollingSharpeAnnualised.to_sql(self.StrategyName + "_RollingSharpeAnnualised",
                                           self.StrategiesCrossValidationConn, if_exists='replace')
            RollingRsSharpeAnnualised.to_sql(self.StrategyName + "_RollingRsSharpeAnnualised",
                                             self.StrategiesCrossValidationConn, if_exists='replace')
        except Exception as e:
            print(e)

        rspnl.to_sql(self.StrategyName + "_rspnl", self.StrategiesCrossValidationConn, if_exists='replace')
        shPnl.to_sql(self.StrategyName + "_shPnl", self.StrategiesCrossValidationConn, if_exists='replace')

        # mainStrategyPnL = pd.read_sql('SELECT * FROM Endurance_NetRets', sqlite3.connect("Endurance.db")).set_index('date', drop=True)

        "PLOTS"
        dfList = [pe.cs(pnl), pe.cs(rspnl), RollingRsSharpeAnnualised]
        titleList = ['csPnL', 'csrsPnL', 'Sh']
        fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
        plt.locator_params(axis='x', nbins=30)
        c = 0
        for df in dfList:
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
            if c > 0:
                (df * 100).plot(ax=ax[c], legend=None)
            else:
                (df * 100).plot(ax=ax[c])
            for label in ax[c].get_xticklabels():
                label.set_fontsize(20)
                label.set_ha("right")
                label.set_rotation(40)
            ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
            ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
            ax[c].set_ylabel("%", fontsize=24)
            if c == 0:
                ax[c].legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'weight': 'bold', 'size': 14})
            ax[c].grid()
            c += 1

        plt.subplots_adjust(top=0.95, bottom=0.15, right=0.85, left=0.12, hspace=0.1, wspace=0)
        plt.show()

        #except Exception as e:
        #    print(e)

    def PlotIndicatorsRnD(self):
        dfInd = self.IndicatorsDF[[
            "USSW5 Curncy","USSW2 Curncy","USSW10 Curncy",
            #"EUSA10 Curncy","EUSA5 Curncy","EUSA2 Curncy",
            #"USSWIT1 Curncy", "USSWIT2 Curncy", "USSWIT5 Curncy", "USSWIT10 Curncy",
            #"EUSWI1 Curncy", "EUSWI2 Curncy", "EUSWI5 Curncy", "EUSWI10 Curncy",
        ]]
        #dfInd = self.IndicatorsDF[["GSUSFCI Index","GSEAFCI Index","GSAUFCI Index"]]
        #dfInd = self.IndicatorsDF[[x for x in self.IndicatorsDF.columns if "IS Curncy" in x]]

        #df_Assets = pe.dlog(pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True))[["NQ1 Index", "GX1 Index"]]
        #df_Assets = pe.cs(df_Assets)
        #df_Assets = df_Assets.rolling(250).mean()

        #dfInd = pe.rollNormalise(dfInd)
        df_Ind_RV = pe.RV(dfInd,RVspace="specificLaggerclassicPermutations_USSW10 Curncy")
        #df_Ind_RV = pe.RV(dfInd,RVspace="specificLaggerclassicPermutations_USSWIT10 Curncy")
        #df_Ind_RV = pe.RV(dfInd,RVspace="specificLaggerclassicPermutations_EUSA10 Curncy")
        #df_Ind_RV["Thr"] = 0
        #df_Assets_RV = pe.RV(df_Assets,RVspace="specificLaggerclassicPermutations_NQ1 Index")
        #df_Assets_RV["Thr"] = 0

        print(df_Ind_RV.iloc[-1])

        dfInd.plot(title="df_Ind")
        df_Ind_RV.plot(title="df_Ind_RV")

        #fig, ax = plt.subplots(nrows=2, ncols=1)
        #df_Ind_RV.plot(ax=ax[0])
        #df_Assets_RV.plot(ax=ax[1])
        plt.show()

#DataDeck("DataDeck.db").Run()
#DataDeck("DataDeck_Research.db").Run()
#DataDeck("DataDeck_Mega.db").Run()
##############################################################################################
#obj = StrategiesCrossValidation("DataDeck", "Asset", "DataDeck_Mega.db")
##############################################################################################
#obj = StrategiesCrossValidation("ActiveStrategies", "ALL", "DataDeck.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "ALL", "DataDeck_1950.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "Endurance", "DataDeck.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "Endurance", "DataDeck_1950.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "Coast", "DataDeck.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "Brotherhood", "DataDeck.db")
#obj = StrategiesCrossValidation("ActiveStrategies", "Valley", "DataDeck.db")
obj = StrategiesCrossValidation("ActiveStrategies", "Dragons", "DataDeck.db")
##############################################################################################
#obj = StrategiesCrossValidation("ResearchStrategies", "Endurance", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Brotherhood", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Coast", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Expedition", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "ExpeditionFXmacro", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Shore", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Valley", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "ValleyRnD", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Dragons", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "MSQIS", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "Lumen", "DataDeck_Research.db")
#obj = StrategiesCrossValidation("ResearchStrategies", "ProphecyPCA", "DataDeck_Research.db")

#obj.PlotIndicatorsRnD()
# obj.see()
# obj.runRollingManifold("PCA")
# obj.ManifoldFactorSee("PCA")

#obj.LO(annualFeeCharged=1.5)
#obj.ERB_ML_RiskOffIndicator()
#obj.Endurance()
#obj.EnduranceSkewed()
#obj.Coast()
#obj.Endurance_csSL()
#obj.Brotherhood()
#obj.Mask()
#obj.Shore()
#obj.Valley()
obj.Dragons()
#obj.Expedition("train")
#obj.Expedition("update")
#obj.Expedition("stealth")
#obj.Lumen()
#obj.Fidei()
####
#obj.ProphecyPCA(mode='run', targetProjection=[39,40], RollMode="RollWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='load', targetProjection=[39,40], RollMode="RollWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='run', targetProjection="EVR", RollMode="RollWindow", RollWindowLength=250, EVRthr=0.25)
#obj.ProphecyPCA(mode='load', targetProjection="EVR", RollMode="RollWindow", RollWindowLength=250, EVRthr=0.75)
#obj.ProphecyPCA(mode='run', targetProjection=[0,1,2,3], RollMode="RollWindow", RollWindowLength=250)
#obj.ProphecyPCA(mode='load', targetProjection=[0,1,2,3], RollMode="RollWindow", RollWindowLength=250)
####
#obj.ProphecyPCA(mode='run', targetProjection="EVR", RollMode="ExpWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='load', targetProjection="EVR", RollMode="ExpWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='run', targetProjection=[0], RollMode="ExpWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='load', targetProjection=[0], RollMode="ExpWindow", RollWindowLength=25)

#obj.ProphecyBeta(mode='run', targetProjection=[0], RollMode="ExpWindow", RollWindowLength=25)
#obj.ProphecyPCA(mode='load', targetProjection=[0], RollMode="ExpWindow", RollWindowLength=25)

obj.pnlReport()
#obj.pnlReport(specificPeriod=["2006-01-01 00:00:00","2007-01-01 00:00:00"])
#obj.pnlReport(specificPeriod=["2011-01-01 00:00:00","2012-01-01 00:00:00"])
#obj.pnlReport(specificPeriod=["2021-01-01 00:00:00","2022-01-01 00:00:00"])

#obj.Contribution(mode="CorrMatrix")
#obj.Contribution(mode="CorrMatrix", TargetStrategy="Endurance")

"PLOT ROUND CONTRACTS TO CHECK OVERALLOCATIONS"
#df = pd.read_sql('SELECT * FROM Valley_roundContractsDF', sqlite3.connect("Valley.db")).set_index("date", drop=True)
#df.plot()
#plt.show()