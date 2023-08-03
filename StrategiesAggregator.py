import pandas as pd, numpy as np, datetime, sqlite3, matplotlib.pyplot as plt
from pyerb import pyerb as pe
import quantstats as qs
import warnings, time
warnings.filterwarnings("ignore")
pd.set_option('max_columns', None)

class StrategiesAggregator:

    def __init__(self, DataDB, mainPortfolio, TargetPortfolios, Leverages):
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
            #print(portfolio)
            subPortfolioRets = pd.read_sql('SELECT * FROM ' + portfolio, self.StrategiesAggregatorConn).set_index('date', drop=True)
            subPortfoliosRetsList.append(subPortfolioRets)
            medPortfolio = self.Leverages[c] * pd.read_sql('SELECT * FROM ' + portfolio + '_NetCashPnlAccCrncy', sqlite3.connect(portfolio+".db")).set_index('date', drop=True)
            medAllocations = self.Leverages[c] * pd.read_sql('SELECT * FROM ' + portfolio + '_trKernel', sqlite3.connect(portfolio+".db")).set_index('date', drop=True)
            try:
                self.PortfolioAllocations[medAllocations.columns] += medAllocations
            except Exception as e:
                pass
                #print(e)
            self.PortfolioAllocations = pd.concat([self.PortfolioAllocations, medAllocations[[x for x in medAllocations if x not in self.PortfolioAllocations.columns]]],axis=1).fillna(0)
            pd.DataFrame(pe.rs(medPortfolio), columns=[portfolio]).to_sql(portfolio, self.StrategiesAggregatorConn, if_exists='replace')
            medPortfolio.columns = [x+"_"+portfolio for x in medPortfolio.columns]
            portfolioList.append(medPortfolio)
            c += 1
        self.PortfolioAllocations = self.PortfolioAllocations.sort_index()
        self.PortfolioAllocations.to_sql("TotalAllocations", self.StrategiesAggregatorConn,if_exists='replace')
        subPortfoliosRetsDF = pd.concat(subPortfoliosRetsList, axis=1).sort_index() / self.AUM
        subPortfoliosRetsDF.to_sql("subPortfoliosRetsDF", self.StrategiesAggregatorConn, if_exists='replace')

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
        print("YTD Cash PnL = ", LivePeriodCashPnL, ", AUM = ", self.AUM, ", (EUR)")
        print("YTD Cash PnL (with OE Adjustments = -25k) = ", LivePeriodCashPnL-25000, ", AUM = ", self.AUM, ", (EUR)")

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
        self.BenchmarkDF = pe.dlog(self.Indicators["NEIXCTA  Index"]).fillna(0)
        self.BenchVolScaler = self.TotalPortfolioReturns.std() / self.BenchmarkDF.std()
        qs.reports.html(self.TotalPortfolioReturns, compounded=False, title="+".join(self.TargetPortfolios), benchmark=self.BenchmarkDF * self.BenchVolScaler,
                        output=self.factSheetReportPath+self.mainPortfolio+"_"+"_".join(self.TargetPortfolios)+"_TimeStory.html")

def PlotAllocations(strategy):

    df = pd.read_sql('SELECT * FROM TotalPortfolio', sqlite3.connect("StrategiesAggregator.db"))
    try:
        df = df.set_index('date',drop=True)
    except Exception as e:
        print(e)
        df = df.set_index('index', drop=True)

    if strategy != "TOTAL":
        df = df[[x for x in df.columns if strategy in x]]

    csdf = pe.cs(df)
    print(csdf.tail(5))

    csdf.plot()
    plt.legend(loc='center left')
    plt.show()

StrategiesAggregator("DataDeck.db", "Endurance", ["Coast", "Brotherhood", "Valley", "Shore", "Dragons"], [1, 1, 1, 1, 1]).LinearAggregation()
#StrategiesAggregator("DataDeck.db", "Endurance", ["Coast", "Brotherhood", "Valley", "Shore", "Dragons"], [1, 1, 1, 1, 0]).LinearAggregation()

#PlotAllocations("TOTAL")
#PlotAllocations("Endurance")
#PlotAllocations("Coast")
#PlotAllocations("Brotherhood")
#PlotAllocations("Valley")
#PlotAllocations("Shore")
