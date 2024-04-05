import time, pickle
try:
    from os.path import dirname, basename, isfile, join
    import glob, os, sys, pdblp,csv
    import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt, matplotlib as mpl
    import streamlit as st
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

class Galileo:

    def __init__(self, tab, DB):
        self.tab = tab
        self.Assets = pd.read_excel("AssetsDashboard.xlsx", sheet_name="Galileo")[tab].dropna().tolist()
        self.workConn = sqlite3.connect(DB)
        self.StrategiesAggregatorDB = sqlite3.connect("StrategiesAggregator.db")
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"

    def getData(self, mode):
        if mode == 0:
            print('Breaking out of data retreival ... ')
        else:
            fields = ["PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME"]

            try:
                self.customDataDF = pd.read_sql('SELECT * FROM DataDeck_'+self.tab, self.workConn).set_index('date', drop=True)
                startDate = self.customDataDF.index[-5].split(" ")[0].replace("-", "")
                print(startDate)
                updateStatus = 'update'
            except Exception as e:
                print(e)
                startDate = '20000101'
                updateStatus = 'fetchSinceInception'

            self.con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
            self.fetchData = self.con.bdh(self.Assets, fields, startDate, '21000630').ffill().bfill()
            self.fetchData.to_sql("bloombergDataDF_"+self.tab, self.workConn, if_exists='replace')

            self.bloombergDataDF = pd.read_sql('SELECT * FROM bloombergDataDF_'+self.tab, self.workConn).set_index('date', drop=True)

            if updateStatus == 'fetchSinceInception':
                self.customDataDF = self.bloombergDataDF
            else:
                self.customDataDF = pd.concat([self.customDataDF, self.bloombergDataDF], axis=0)
                self.customDataDF = self.customDataDF[~self.customDataDF.index.duplicated(keep='last')]
                print(self.bloombergDataDF)
                print(self.customDataDF)

            self.customDataDF.to_sql("DataDeck_"+self.tab, self.workConn, if_exists='replace')

    def FieldFilter(self, field):
        self.customDataDF = pd.read_sql('SELECT * FROM DataDeck_'+self.tab, self.workConn).set_index('date', drop=True)
        self.df = self.customDataDF[[x for x in self.customDataDF.columns if field in x]]
        self.df.columns = [x.split(",")[0].replace('(', "").replace("'", "") for x in self.df.columns]

    def BetaEye(self, **kwargs):
        if 'n' in kwargs:
            n = kwargs['n']
        else:
            n = 250

        if 'runVector' in kwargs:
            runVector = kwargs['runVector']
        else:
            runVector = [1,1,1]

        if 'BetaMode' in kwargs:
            BetaMode = kwargs['BetaMode']
        else:
            BetaMode = 'BetaRegressV'

        self.FieldFilter('PX_LAST')
        self.rets = pe.dlog(self.df).fillna(0)

        if runVector[0] == 1:
            BetaEyeDF = pe.BetaKernel(self.rets.iloc[-n:,:]).round(2)

            pe.RefreshableFile([[BetaEyeDF.reset_index(), 'BetaEyeDF_' + self.tab]],
                               self.GreenBoxFolder + 'BetaEyeDF_' + self.tab + '.html', 5,
                               cssID='QuantitativeStrategies', addButtons="QuantTools", specificID="BetaEyeDF")

        if runVector[1] == 1:
            out = ManSee.gRollingManifold(BetaMode, self.rets, 250, len(self.rets.columns), range(len(self.rets.columns)), Scaler=None)
            out[0].to_sql('df_Galileo', self.workConn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(BetaMode + '_Galileo_principalCompsDf_' + str(out[0].columns[k]), self.workConn, if_exists='replace')
                exPostProjectionsList[k].to_sql(BetaMode + '_Galileo_exPostProjections_' + str(out[0].columns[k]), self.workConn, if_exists='replace')

        if runVector[2] == 1:
            driver = 'RX1 Comdty'
            targetProjectionDF = pd.read_sql('SELECT * FROM "'+BetaMode+'_Galileo_principalCompsDf_'+driver+'"', self.workConn).set_index('date', drop=True)
            targetIdx = targetProjectionDF.index
            targetProjectionDF = targetProjectionDF.reset_index()
            targetProjectionDF['date'] = targetProjectionDF['date'].str.split(" ").str[0]
            targetProjectionDF = targetProjectionDF.rename(columns={'date':'Dates'})
            targetProjectionDF = targetProjectionDF.set_index('Dates', drop=True).fillna(0)
            ##############################################################################################
            targetProjectionDF.plot(title=driver+" Betas")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder+"BetaTS_"+driver.replace(" ", "")+"_All.jpg")
            ##############################################################################################
            subset = ["FF1 Comdty", "DU1 Comdty", "TU1 Comdty", "ED1 Comdty"]
            targetProjectionDF_subset = targetProjectionDF[subset]
            targetProjectionDF_subset.plot(title=driver+" Subset")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder+"BetaTS_"+driver.replace(" ", "")+"_Subset.jpg")
            ##############################################################################################
            pe.rowStoch(targetProjectionDF).plot(title=driver+" Subset (RowStochastic)")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder+"BetaTS_"+driver.replace(" ", "")+"_Subset_RowStochastic.jpg")
            ##############################################################################################
            targetProjectionDF.iloc[-250:].plot(title=driver+" 1Y")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder+"BetaTS_"+driver.replace(" ", "")+"_1Y.jpg")
            ##############################################################################################
            targetProjectionDF.iloc[-250:].to_csv(self.GreenBoxFolder+"BetaTS_"+driver.replace(" ", "")+".csv", quoting=csv.QUOTE_ALL)
            ##############################################################################################
            ################################### REGRESSION PNL ###########################################
            ##############################################################################################
            targetProjectionDF_subset.index = targetIdx
            trKernel = pe.sign(pd.DataFrame(pe.rs(targetProjectionDF_subset * self.rets[subset]), columns=[driver])).iloc[-50:,:]
            trKernel.plot(title=driver + " Subset TradingKernel (Sign)- Last 50 Days")
            plt.legend(loc=3, prop={'size': 8})
            #plt.show()
            plt.savefig(self.GreenBoxFolder + "BetaTS_" + driver.replace(" ", "") + "_Subset_TradingKernel.jpg")
            ##############################################################################################
            pnl_Beta = pd.DataFrame(self.rets[driver]) * pe.S(trKernel)
            pe.cs(pnl_Beta).plot(title=driver + " Subset PnL Beta (Sharpe = "+ str(np.sqrt(252)*pe.sharpe(pnl_Beta).values[0])+ ")")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder + "BetaTS_" + driver.replace(" ", "") + "_pnl_Beta.jpg")
            ##############################################################################################
            pnl_Beta_Binary = pd.DataFrame(self.rets[driver]) * pe.S(pe.sign(trKernel))
            pe.cs(pnl_Beta_Binary).plot(title=driver + " Subset PnL Beta Binary (Sharpe = "+ str(np.sqrt(252)*pe.sharpe(pnl_Beta_Binary).values[0])+ ")")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder + "BetaTS_" + driver.replace(" ", "") + "_pnl_Beta_Binary.jpg")

    def StrategiesFactorsExposures(self, Strategies, runVector):
        self.FieldFilter('PX_LAST')
        self.rets = pe.dlog(self.df).fillna(0)

        QuantPortfolio = []
        for strategy in Strategies:
            QuantPortfolio.append(pd.read_sql('SELECT * FROM '+strategy, self.StrategiesAggregatorDB).set_index('date', drop=True))
        QuantPortfolioDF = pd.concat(QuantPortfolio, axis=1)
        self.rets = pd.concat([self.rets, QuantPortfolioDF], axis=1).fillna(0)

        BetaMode = 'BetaRegressV'

        if runVector[0] == 1:
            out = ManSee.gRollingManifold(BetaMode, self.rets, 250, len(self.rets.columns), range(len(self.rets.columns)), Scaler=None)
            out[0].to_sql('df_StrategiesFactorsExposures', self.workConn, if_exists='replace')
            principalCompsDfList = out[1]
            exPostProjectionsList = out[2]
            for k in range(len(principalCompsDfList)):
                principalCompsDfList[k].to_sql(BetaMode + '_Galileo_StrategiesFactorsExposures_principalCompsDf_' + str(out[0].columns[k]),
                                               self.workConn, if_exists='replace')
                exPostProjectionsList[k].to_sql(BetaMode + '_Galileo_StrategiesFactorsExposures_exPostProjections_' + str(out[0].columns[k]),
                                                self.workConn, if_exists='replace')

        if runVector[1] == 1:
            driver = 'Endurance'
            targetProjectionDF = pd.read_sql('SELECT * FROM "'+BetaMode+'_Galileo_StrategiesFactorsExposures_principalCompsDf_'+driver+'"', self.workConn)
            targetProjectionDF = targetProjectionDF.rename(columns={'index':'date'})
            targetProjectionDF = targetProjectionDF.set_index('date', drop=True)
            targetProjectionDF = targetProjectionDF.reset_index()
            targetProjectionDF['date'] = targetProjectionDF['date'].str.split(" ").str[0]
            targetProjectionDF = targetProjectionDF.rename(columns={'date':'Dates'})
            targetProjectionDF = targetProjectionDF.set_index('Dates', drop=True).fillna(0)
            targetProjectionDF = targetProjectionDF.drop(columns=[driver])
            ##############################################################################################
            targetProjectionDF.iloc[-25:].plot(title=driver + " Betas (Last 25 Days)")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder + "BetaTS_StrategiesFactorsExposures_" + driver.replace(" ", "") + "_All.jpg")
            ##############################################################################################
            pe.rowStoch(targetProjectionDF).plot(title=driver + " Betas (Row Stochastic)")
            plt.legend(loc=3, prop={'size': 8})
            plt.savefig(self.GreenBoxFolder + "BetaTS_StrategiesFactorsExposures_" + driver.replace(" ", "") + "_RowStochastic.jpg")

    def ApplyTechnicals(self):
        self.customDataDF = pd.read_sql('SELECT * FROM DataDeck_'+self.tab, self.workConn).set_index('date', drop=True)
        for asset in self.Assets:
            print(asset)
            targetCols = [x for x in self.customDataDF.columns if asset in x]
            df = self.customDataDF[targetCols]
            df.columns = ["Open", "High", "Low", "Close"]
            df["Volume_BTC"] = 1
            df = add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=True)
            df.to_sql(asset+"_Galileo", self.workConn, if_exists='replace')

    def RiskOnOffIndexing(self, mode):
        self.FieldFilter('PX_LAST')
        self.rets = pe.dlog(self.df).fillna(0)

        if mode == 'setup':
            out_corr = ManSee.gRollingManifold("PCA_Correlations", self.rets, 25, len(self.rets.columns), range(len(self.rets.columns)), RollMode='ExpWindow')
            df_corr = out_corr[4]
            df_corr.to_sql('PCA_Correlations_EigValsDF', self.workConn, if_exists='replace')

            out = ManSee.gRollingManifold("PCA", self.rets, 25, len(self.rets.columns), range(len(self.rets.columns)), RollMode='ExpWindow')
            df = out[4]
            df.to_sql('PCA_EigValsDF', self.workConn, if_exists='replace')

        elif mode == 'run':
            df = pd.read_sql('SELECT * FROM PCA_EigValsDF', self.workConn).set_index('date', drop=True)
            df_corr = pd.read_sql('SELECT * FROM PCA_Correlations_EigValsDF', self.workConn).set_index('date', drop=True)

        plotIndex = pd.concat([df.iloc[:,0],df_corr.iloc[:,0]],axis=1).loc['2001-01-01 00:00:00':,:].abs()
        #plotIndex = pd.concat([df.iloc[:,10],df_corr.iloc[:,10]],axis=1)
        plotIndex.columns = ['Volatility PCA Index', 'Correlation PCA Index']
        plotIndex = (plotIndex-plotIndex.rolling(25*2).mean())/plotIndex.rolling(25*2).std()
        signal = np.sign(plotIndex['Volatility PCA Index']-plotIndex['Correlation PCA Index'])
        signal[signal == 1] = 0
        pickle.dump(signal, open("F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\PortfolioHedge\\" + "Galileo_PCA_RiskOff_Indexing_signal.p", "wb"))

        fig_report, ax_report = plt.subplots(sharex=True, nrows=2, ncols=1, figsize=(15, 15))
        plotIndex.plot(ax=ax_report[0], title='PCA Sentiment Indexing', legend=None)
        signal.plot(ax=ax_report[1], title='Signal', legend=None)
        plt.legend(loc='lower left')
        #plt.xticks(plt.xticks()[0],[ax_report[1].get_xticklabels()[0].set_text("")] + [t.get_text().replace(" 00:00:00", "") for t in ax_report[1].get_xticklabels() if t.get_position()[0] >= 0], rotation=45)
        plt.show()

def GalileoBetaRunner():

    print("Running Galileo Betas ... $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    for assetClass in ["FX", "MajorFutures", "Macros"]:
        print("Calculating ", assetClass)
        obj = Galileo(assetClass, "Galileo.db")

        #obj.getData(0) # pass data fetching
        obj.getData(1)
        if assetClass == 'MajorFutures':
            obj.BetaEye(runVector=[1, 1, 1])
            #obj.BetaEye(runVector=[1, 0, 1])
        elif assetClass == 'Macros':
            obj.StrategiesFactorsExposures(Strategies=["Endurance", "Coast", "Brotherhood"], runVector=[1, 1])
            #obj.StrategiesFactorsExposures(Strategies=["Endurance", "Coast", "Brotherhood"], runVector=[0, 1])

    #obj.ApplyTechnicals()
    obj.BetaEye(runVector=[1, 1, 1])

def RiskOnOffIndex():
    obj = Galileo("RiskOnOffIndexing", "Galileo.db")
    #obj.getData(1)
    #obj.RiskOnOffIndexing('setup')
    obj.RiskOnOffIndexing('run')

def MacroGlue(mode):
    AssetsDF = pe.dlog(pd.read_sql('SELECT * FROM DataDeck', sqlite3.connect("DataDeck.db")).set_index('date', drop=True))
    IndicatorsDF = pe.d(pd.read_sql('SELECT * FROM IndicatorsDeck', sqlite3.connect("DataDeck.db")).set_index('date', drop=True))
    CTA = pd.read_sql('SELECT * FROM subPortfoliosRetsDF', sqlite3.connect("StrategiesAggregator.db")).set_index('index', drop=True)

    #df = pd.concat([AssetsDF, IndicatorsDF], axis=1).sort_index()
    #df = df[["ES1 Index", "NQ1 Index", "TY1 Comdty", "RX1 Comdty","VIX Index", "MOVE Index", "BFCIUS Index", "BCMPEAFC Index","USSWIT1 Curncy", "USSWIT2 Curncy", "EUSWI1 Curncy", "EUSWI2 Curncy","EUR003M Index", "US0003M Index", "USGG10YR Index", "M2US000G Index", "MAEUMMT Index"]]

    df = pd.concat([CTA, IndicatorsDF[["VIX Index", "MOVE Index", "BFCIUS Index", "BCMPEAFC Index","USSWIT1 Curncy", "USSWIT2 Curncy", "EUSWI1 Curncy", "EUSWI2 Curncy","EUR003M Index", "US0003M Index", "USGG10YR Index", "M2US000G Index", "MAEUMMT Index"]]], axis=1).sort_index()
    df = df.fillna(0)#.iloc[:,[0,1,2]].tail(100)

    Roll_MI_Settings = {"metric":"MI", "RollMode":"RollWindow", "st":25}
    #Roll_MI_Settings = {"metric":"MI", "RollMode":"RollWindow", "st":250}
    #Roll_MI_Settings = {"metric":"MI", "RollMode":"ExpWindow", "st":25}

    ID = '_'.join(f'{key}_{value}' for key, value in Roll_MI_Settings.items())

    if mode == "run":
        #out = pe.RollMetric(df, metric="MI", RollMode="ExpWindow")
        out = pe.RollMetric(df, metric="MI", RollMode="RollWindow", st=25)

        for subDF in out:
            subDF.fillna(0).to_sql(subDF.name+"_"+ID, sqlite3.connect("Galileo.db"), if_exists='replace')

    elif mode == "read":
        for c in df.columns:
            print(c)
            subDF = pd.read_sql("SELECT * FROM '"+c+"_"+ID+"'", sqlite3.connect("Galileo.db")).set_index('index', drop=True)
            subDF = subDF[["BFCIUS Index",
                           #"BCMPEAFC Index",
                           "USSWIT1 Curncy",
                           #"USSWIT2 Curncy", "EUSWI1 Curncy", "EUSWI2 Curncy",
                           #"EUR003M Index",
                           #"US0003M Index",
                           "USGG10YR Index",
                           "M2US000G Index",
                           "MAEUMMT Index"
                           ]]
            st.write("Gallileo")
            st.line_chart(subDF["BFCIUS Index"])
            #subDF.plot()
            #plt.show()

def TestStreamLit():
    LocalDataConn = sqlite3.connect("DataDeck.db")
    df = pd.read_sql('SELECT * FROM DataDeck', LocalDataConn).set_index('date', drop=True)

#GalileoBetaRunner()
#RiskOnOffIndex()

#MacroGlue("run")
#MacroGlue("read")

