import pandas as pd, numpy as np, datetime, sqlite3, matplotlib.pyplot as plt, os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')
from pyerb import pyerb as pe
import quantstats as qs
import warnings, time,glob, shutil
warnings.filterwarnings("ignore")
pd.set_option('max_columns', None)

class StrategiesAggregator:

    def __init__(self, DataDB, mainPortfolio, TargetPortfolios, Leverages):
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.DataDB = DataDB
        self.StrategiesAggregatorConn = sqlite3.connect("StrategiesAggregator.db")
        self.factSheetReportPath = "F:\Dealing\Panagiotis Papaioannou\pyerb\PyEurobankBloomberg\PySystems\PyLiveTradingSystems\StrategiesFactSheets/"
        self.Indicators = pd.read_sql('SELECT * FROM IndicatorsDeck', sqlite3.connect("DataDeck.db")).set_index('date', drop=True).ffill()
        self.Indicators.index = pd.DatetimeIndex(self.Indicators.index)
        self.AssetsRets = pe.dlog(pd.read_sql('SELECT * FROM DataDeck', sqlite3.connect(self.DataDB)).set_index('date', drop=True))
        self.mainPortfolioDF = pd.read_sql('SELECT * FROM ' + mainPortfolio + '_NetCashPnlAccCrncy', sqlite3.connect(mainPortfolio+".db")).set_index('date', drop=True)
        self.PortfolioAllocations = pd.read_sql('SELECT * FROM ' + mainPortfolio + '_trKernel', sqlite3.connect(mainPortfolio+".db")).set_index('date', drop=True)
        self.mainPortfolioDF.columns = [x + "_" + mainPortfolio for x in self.mainPortfolioDF.columns]
        self.mainPortfolio = mainPortfolio
        self.TargetPortfolios = TargetPortfolios
        self.Leverages = Leverages
        self.AUM = 1000000

    def LinearAggregation(self):
        portfolioList = []
        portfolioList.append(self.mainPortfolioDF)
        subPortfoliosRetsList = []
        subPortfoliosRetsList.append(pd.read_sql('SELECT * FROM ' + self.mainPortfolio, self.StrategiesAggregatorConn).set_index('date', drop=True))
        pd.DataFrame(pe.rs(self.mainPortfolioDF), columns=[self.mainPortfolio]).to_sql(self.mainPortfolio, self.StrategiesAggregatorConn, if_exists='replace')
        c = 0
        for portfolio in self.TargetPortfolios:
            medPortfolio = self.Leverages[c] * pd.read_sql('SELECT * FROM ' + portfolio + '_NetCashPnlAccCrncy', sqlite3.connect(portfolio+".db")).set_index('date', drop=True)
            medPortfolio.to_sql(portfolio, self.StrategiesAggregatorConn, if_exists='replace')
            subPortfoliosRetsList.append(pd.DataFrame(pe.rs(medPortfolio), columns=[portfolio]))
            medAllocations = self.Leverages[c] * pd.read_sql('SELECT * FROM ' + portfolio + '_trKernel', sqlite3.connect(portfolio+".db")).set_index('date', drop=True)
            self.PortfolioAllocations[medAllocations.columns] = 0
            self.PortfolioAllocations[medAllocations.columns] += medAllocations
            pd.DataFrame(pe.rs(medPortfolio), columns=[portfolio]).to_sql(portfolio, self.StrategiesAggregatorConn, if_exists='replace')
            medPortfolio.columns = [x+"_"+portfolio for x in medPortfolio.columns]
            portfolioList.append(medPortfolio)
            c += 1
        self.PortfolioAllocations = self.PortfolioAllocations.sort_index()
        self.PortfolioAllocations.to_sql("TotalAllocations", self.StrategiesAggregatorConn,if_exists='replace')
        subPortfoliosRetsDF = pd.concat(subPortfoliosRetsList, axis=1).sort_index() / self.AUM
        subPortfoliosRetsDF.sort_index().to_sql("subPortfoliosRetsDF", self.StrategiesAggregatorConn, if_exists='replace')

        self.TotalPortfolio = pd.concat(portfolioList, axis=1).fillna(0)
        self.TotalPortfolio.index = pd.DatetimeIndex(self.TotalPortfolio.index)
        self.TotalPortfolio = self.TotalPortfolio.sort_index()
        self.TotalPortfolio.to_sql("TotalPortfolio", self.StrategiesAggregatorConn, if_exists='replace')
        rsTotalPortfolio = pe.rs(self.TotalPortfolio)
        rsTotalPortfolio.to_sql("rsTotalPortfolio", self.StrategiesAggregatorConn, if_exists='replace')
        csrsTotalPortfolio = pe.cs(rsTotalPortfolio)
        csrsTotalPortfolio.to_sql("csrsTotalPortfolio", self.StrategiesAggregatorConn, if_exists='replace')
        print("Total Sharpe = ", np.sqrt(252) * pe.sharpe(rsTotalPortfolio))
        today = datetime.datetime.now()
        LivePeriodCashPnL = (pe.cs(rsTotalPortfolio.loc[str(today.year)+"-01-01 00:00:00":])).iloc[-1]
        print("YTD Cash PnL = ", round(LivePeriodCashPnL,2), ", AUM = ", self.AUM, ", (EUR)")
        #print("YTD Cash PnL (with OE Adjustments = -...) = ", LivePeriodCashPnL-..., ", AUM = ", self.AUM, ", (EUR)")

        AssetsContributions = pe.S(self.PortfolioAllocations, nperiods=2) * self.AssetsRets[self.PortfolioAllocations.columns]
        AssetsContributions.to_sql("AssetsContributions", self.StrategiesAggregatorConn, if_exists='replace')
        pe.cs(AssetsContributions).to_sql("csAssetsContributions", self.StrategiesAggregatorConn, if_exists='replace')
        AssetsContributionsVols = (np.sqrt(252)*100*pe.roller(AssetsContributions, np.std, n=250))
        AssetsContributionsVols.to_sql("AssetsContributionsVols", self.StrategiesAggregatorConn, if_exists='replace')

        #fig, ax = plt.subplots(nrows=3, ncols=1)
        #pe.cs(self.TotalPortfolio).plot(ax=ax[0], title='Strategies Contributions', legend=None)
        #AssetsContributionsVols.plot(ax=ax[1], title='Assets Contributions Volatilities')
        #ax[1].legend(loc='lower left')
        #csrsTotalPortfolio.plot(ax=ax[2], title='Total Book', legend=None)
        #plt.show()

        self.TotalPortfolioReturns = rsTotalPortfolio / self.AUM
        self.TotalPortfolioReturns.to_sql("TotalPortfolioReturns", self.StrategiesAggregatorConn, if_exists='replace')

        print("Strategies Aggregator : QUANTSTATS HTML REPORT ")
        qs.extend_pandas()
        self.BenchmarkDF = pe.dlog(self.Indicators["NEIXCTA Index"]).fillna(0)
        self.BenchmarkDF.index = pd.to_datetime(self.BenchmarkDF.index)

        self.BenchVolScaler = self.TotalPortfolioReturns.std() / self.BenchmarkDF.std()
        qs.reports.html(self.TotalPortfolioReturns, compounded=False, title="+".join(self.TargetPortfolios), benchmark=self.BenchmarkDF * self.BenchVolScaler,
                        output=self.factSheetReportPath+self.mainPortfolio+"_"+"_".join(self.TargetPortfolios)+"_TimeStory.html")

    def PlotContributions(self):

        dfPortfolio = pd.read_sql('SELECT * FROM TotalPortfolio', self.StrategiesAggregatorConn)
        try:
            dfPortfolio = dfPortfolio.set_index('date', drop=True)
        except Exception as e:
            print(e)
            dfPortfolio = dfPortfolio.set_index('index', drop=True)
        ############################################################################################
        for pack in ['Portfolio', 'Allocations']:
            self.VisualCheckPath = self.AlternativeStorageLocation + "HealthChecksPacks/" + pack + "_VisualCheck/"
            ############################################################################################
            for folderPath in [self.VisualCheckPath, self.VisualCheckPath+'/HTMLs/']:
                files_list = os.listdir(folderPath)
                for files in files_list:
                    if files != "HTMLs":
                        os.remove(folderPath+files)
            ############################################################################################
            StrategiesSet = list(set([x.split("_")[1] for x in dfPortfolio.columns]))
            if pack == 'Portfolio':
                dfMain = dfPortfolio.copy()
            ############################################################################################
            for strategy in StrategiesSet:
                if pack == "Allocations":
                    df = pd.read_sql('SELECT * FROM ' + strategy + '_targetNotionalsAccCrncy', sqlite3.connect(strategy+".db")).set_index('date', drop=True)
                elif (pack == 'Portfolio')&(strategy != "TOTAL"):
                    df = dfMain[[x for x in dfMain.columns if strategy in x]]
                ############################################################################################
                for plotPack in [[df, "SinceInception"],[df.iloc[-250:,:], "1Y"]]:

                    plotDF = plotPack[0]
                    plotDF.index = [x.split(" ")[0] for x in plotDF.index]

                    if pack == "Portfolio":
                        csplotDF = pe.cs(plotDF)
                        csplotDF.plot()
                    elif pack == "Allocations":
                        plotDF.plot()

                    plt.legend(loc='center left')
                    plt.xticks(rotation=25)
                    #plt.show()
                    plt.savefig(self.VisualCheckPath + plotPack[1] + "__" + strategy)

            ############################################################################################
            for strategy in StrategiesSet:
                f = open(self.VisualCheckPath + '/HTMLs/' + strategy + '.html', 'w')
                DT_html_template = '<html> <head> <title>Title</title></head><body><h2>'+pack+' Visualiser for ' + strategy + '</h2>'
                DT_html_template += '<a href="file:///F:/Dealing/Panagiotis%20Papaioannou/pyerb/PyEurobankBloomberg/PyBloomyFlask/template/PyEurobankFlask_Home.html">Eurobank Flask Home</a><br>'
                #####################
                for name in glob.glob(self.VisualCheckPath + '/*' + strategy + '*.png'):
                    DT_html_template += '<img src="' + name + '" alt="DTVisualiser" width="1200" height="800"><hr>'
                #####################
                DT_html_template += '</body></html>'
                f.write(DT_html_template)
                f.close()

def AdHocRunner():
    DataDeckExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
    Live_Strategies_Control_Panel = pd.read_excel(DataDeckExcel, sheet_name="Live Strategies Control Panel", engine='openpyxl').dropna(subset=["Strategy Name"]).dropna(axis=1)
    obj = StrategiesAggregator("DataDeck.db", Live_Strategies_Control_Panel["Strategy Name"].iloc[0],
                                        Live_Strategies_Control_Panel["Strategy Name"].iloc[1:].tolist(),
                                        Live_Strategies_Control_Panel["CTA Allocation"].iloc[1:].tolist())
    obj.LinearAggregation()
    obj.PlotContributions()

#AdHocRunner()

