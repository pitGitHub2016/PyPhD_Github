import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt
from scipy.stats import norm, t
import time, pickle
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
import seaborn as sn
from PyEurobankBloomberg.PySystems.PyLiveTradingSystems.DataDeck import DataDeck
from pyerb import pyerb as pe
from pyerbML import ML, ManSee
import win32com.client as win32
from email.mime.text import MIMEText
import quantstats as qs

class RiskStatus:

    def __init__(self):
        self.StrategyName = "RiskStatus"
        self.Assets = pd.read_excel("AssetsDashboard.xlsx", sheet_name="ResearchStrategies")["ExpeditionFXmacro"].dropna().tolist()
        self.dataConn = sqlite3.connect("DataDeck_Research.db")
        self.workConn = sqlite3.connect("RiskStatusFX.db")
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"
        self.GPC_MemoryDepth = 250

    def Expedition(self, runMode):

        def rollPreds(roll_rets, assetSel):
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

        self.df = pd.read_sql('SELECT * FROM DataDeck', self.dataConn).set_index('date', drop=True)[
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

        model_c = 0
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")

            model = modelList[model_c]

            if runMode == 'train':
                startPeriod = rets.shape[0] - 1000  # , stepper
            elif runMode == 'update':
                startPeriod = rets.shape[0] - 5  # last 'x' days

            if runMode in ['train', 'update']:

                PredDataList = []
                for i in tqdm(range(startPeriod, rets.shape[0] + 1)):
                    med_rets = rets.iloc[i - self.GPC_MemoryDepth:i, :]
                    "Here I build the regression problem"
                    rolling_Predictions = rollPreds(med_rets, list(rets.columns).index(asset))
                    #print(asset, rolling_Predictions)
                    PredDataList.append(rolling_Predictions)
                try:
                    PredsDF = pd.DataFrame(PredDataList, columns=["date", asset] + [str(x) for x in
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
                                   self.workConn, if_exists='replace')

                if runMode == 'update':
                    prev_PredsDF = pd.read_sql(
                        "SELECT * FROM " + self.StrategyName + "_" + subAssetName + "_PredsDF",
                        self.workConn).set_index('date', drop=True)
                    print(prev_PredsDF)
                    PredsDF = pd.concat([prev_PredsDF, PredsDF])
                    print(PredsDF)
                    PredsDF = PredsDF[~PredsDF.index.duplicated(keep='last')]

                PredsDF.to_sql(self.StrategyName + "_" + subAssetName + '_PredsDF', self.workConn,
                               if_exists='replace')
                PredsDF.to_csv("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF_FX.csv')
                PredsDF.to_excel("PortfolioHedge\\" + self.StrategyName + "_" + subAssetName + '_PredsDF_FX.xlsx')

            else:
                pass
                # fig, ax = plt.subplots(sharex=True, nrows=3, ncols=1)
                # PredsDF[asset].plot(ax=ax[0])
                # PredsDF["-1.0"].plot(ax=ax[1])
                # self.df[asset].iloc[-100:].plot(ax=ax[2])
                # plt.show()

            model_c += 1

        "Group all together!"
        self.df.iloc[self.df.shape[0] - 25:, :][targetAssetList].to_excel("PortfolioHedge\\targetAssetsDF_Report_FX.xlsx")
        predsList = []
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")
            subPredsDF = pd.read_sql(
                "SELECT * FROM "+self.StrategyName+"_" + subAssetName + "_PredsDF", self.workConn).set_index(
                'date', drop=True)
            subPredsDF.columns = [subPredsDF.columns[0], subPredsDF.columns[0] + " RiskOff Probability",
                                  subPredsDF.columns[0] + " RiskOn Probability"]
            predsList.append(subPredsDF)

        PredsDF = pd.concat(predsList, axis=1)
        RiskOnProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOn' in x]]
        RiskOffProbsDF = PredsDF[[x for x in PredsDF.columns if 'RiskOff' in x]]
        SignalsDF = PredsDF[[x for x in PredsDF.columns if ('RiskOn' not in x)&('RiskOff' not in x)]]
        RiskOnProbsDF.to_excel("PortfolioHedge\\RiskOnProbsDF_FX.xlsx")
        RiskOffProbsDF.to_excel("PortfolioHedge\\RiskOffProbsDF_FX.xlsx")
        SignalsDF.to_excel("PortfolioHedge\\SignalsDF_FX.xlsx")

        SignalsDF.to_sql(self.StrategyName + '_SignalsDF', self.workConn, if_exists='replace')

        RiskOffProbsDF.columns = [x+" (%)" for x in RiskOffProbsDF.columns]
        dfToHtml = RiskOffProbsDF.tail(50).round(2).sort_index(ascending=False).reset_index()
        dfToHtml[dfToHtml.columns[0]] = dfToHtml[dfToHtml.columns[0]].astype(str).str.replace(r' 00:00:00', '', regex=True)
        pe.RefreshableFile([[dfToHtml, 'RiskOffProbsDF']],
                        self.GreenBoxFolder + 'ERB_ML_RiskOffProbsDF_FX.html',
                        5, cssID='QuantitativeStrategies', specificID="ML_RiskOff", addButtons="RiskStatusFX")

        sig = SignalsDF.copy()
        sig = pe.S(sig, nperiods=1)
        sig = pe.fd(sig)

        for strat in ['Hedge', 'Alpha']:
            print(strat)
            if strat == 'Alpha':
                sig[sig == 0] = 1

            pnl = sig * retsRAW
            pnl = pnl.loc[sig.index[0]:,:].dropna(axis=1, how='all')

            qs.extend_pandas()
            for c in pnl.columns:
                subPnl = pnl[c]
                subPnl.index = pd.to_datetime(subPnl.index)
                qs.reports.html(subPnl, compounded=False,
                                output=self.GreenBoxFolder + "EcoHedge_" + strat + "_" + c + "_FX.html")

            sh = np.sqrt(252) * pe.sharpe(pnl)
            print("Sharpe = ", sh)

            #rsPnl = pe.rs(pnl)
            #csPnl = pe.cs(pnl)
            #fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1)
            #csPnl.plot(ax=ax[0])
            #pe.cs(rsPnl).plot(ax=ax[1])
            #plt.show()

            #plt.savefig(self.GreenBoxFolder + strat + '.jpg')

    def SendEMail(self):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = 'panpapaioannou@eurobank.gr' # For Testing
        #mail.To = 'FX_TRADING@eurobank.gr'
        #mail.To = 'Global_Markets_Trading@eurobank.gr; AIoannidis@eurobank.gr'
        mail.Subject = 'Eurobank ML Risk Off Probability Index'
        #mail.Body = 'Message body'
        html = open(self.GreenBoxFolder + 'ERB_ML_RiskOffProbsDF_FX.html')
        msg = MIMEText(html.read(), 'html')
        mail.HTMLBody = msg.as_string()

        # To attach a file to the email (optional):
        #attachment = "Path to the attachment"
        #mail.Attachments.Add(attachment)

        mail.Send()

#DataDeck("DataDeck_Research.db").Run()
obj = RiskStatus()
obj.Expedition("train")
#obj.Expedition("update")
#obj.Expedition("stealth")
##obj.SendEMail()
