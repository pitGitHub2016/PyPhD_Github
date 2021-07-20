from Slider import Slider as sl
from scipy.linalg import svd
import numpy as np, investpy, json, time, pickle, glob
from tqdm import tqdm
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from pydiffmap import kernel
import pyriemann as pr
from pyriemann.utils import distance as mt
from scipy import stats as st
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

conn = sqlite3.connect('SmartGlobalAssetAllocation.db')
fromDate = '01/02/2005'
toDate = '01/02/2021'
twList = [250]
spacialND = range(5)

BondsTickers = ["U.S. 10Y", "U.S. 5Y", "U.S. 2Y", "U.S. 1Y", "U.S. 6M",
                "U.K. 10Y", "U.K. 5Y", "U.K. 2Y", "U.K. 1Y", "U.K. 6M",
                "China 10Y", "China 5Y", "China 2Y", "China 1Y",
                "Japan 10Y", "Japan 5Y", "Japan 2Y", "Japan 1Y", "Japan 6M",
                "Germany 10Y", "Germany 5Y", "Germany 2Y", "Germany 1Y", "Germany 6M"]
EqsTickers = [["S&P 500", "United States"], ["Nasdaq", "United States"],
              ["DAX", "Germany"], ["CAC 40", "France"], ["FTSE 100", "United Kingdom"],
              ["Shanghai", "China"], ["Nikkei 225", "Japan"]]
FxTickers = ['EUR/USD', 'GBP/USD', 'USD/CNY', 'USD/JPY']
CommoditiesTickers = ['Gold', 'Silver', 'Brent Oil', 'Crude Oil WTI', 'Natural Gas']
CustomTickers = [x for x in glob.glob("Investing_csvData/*.csv")]

subPortfoliosList = [[[x[0] for x in EqsTickers], 'EqsFuts'],
                   [['Euro SCHATZ Futures', 'Euro Bund Futures', 'Euro BOBL Futures',
                     'US 2 Year T-Note Futures', 'US 5 Year T-Note Futures', 'US 10 Year T-Note Futures',
                     'US 30 Year T-Bond Futures', 'UK Gilt Futures', 'Japan Government Bond Futures'], 'BondsFuts'],
                   [['Euribor Futures', 'Eurodollar Futures'], 'IRsFuts'],
                   [CommoditiesTickers, 'Commodities'], [['EURUSD', 'GBPUSD', 'CNYUSD', 'JPYUSD','US Dollar Index Futures'], 'FX']]

def ProductsSearch():
    search_results = investpy.search.search_quotes(text='CBOE Volatility Index', n_results=100)
    # 'id_': 44336, 'name': 'CBOE Volatility Index'
    # 'id_': 8859, 'name': 'Nikkei 225 Futures'
    # 'id_': 8984, 'name': 'Hang Seng Futures'
    # 'id_': 8867, 'name': 'Euro Stoxx 50 Futures'
    # 'id_': 8826, 'name': 'DAX Futures'
    # 'id_': 8874, 'name': 'NASDAQ Futures'
    # 'id_': 8839, 'name': 'S&P 500 Futures'
    # 'id_': 8830, 'name': 'Gold Futures'
    # 'id_': 8827, 'name': 'US Dollar Index Futures'
    # 'id_': 992749, 'name': 'MSCI Emerging Markets Equity Futures'
    # 'id_': 8895, 'name': 'Euro Bund Futures'
    # 'id_': 8880, 'name': 'US 10 Year T-Note Futures'
    # 'id_': 8986, 'name': 'Japan Government Bond Futures'
    # 'id_': 1073160, 'name': '10 Years Russian Federation Government Bond Future'
    for search_result in search_results:
        jResult = json.loads(str(search_result))
        print(jResult)
        print(jResult["id_"])
        search_result.retrieve_historical_data(from_date=fromDate, to_date=toDate)
        print(search_result.retrieve_historical_data(from_date=fromDate, to_date=toDate))
        print(search_result)
        break

def DataHandler(mode):
    if mode == 'run':

        dataVec = [0, 0, 0, 0, 1]

        if dataVec[0] == 1:

            BondsList = []
            for bond in BondsTickers:
                print(bond)
                df = investpy.get_bond_historical_data(bond=bond, from_date=fromDate, to_date=toDate).reset_index().rename(
                    columns={"Date": "Dates", "Close": bond}).set_index('Dates')[bond]
                BondsList.append(df)

            BondsDF = pd.concat(BondsList, axis=1)
            BondsDF[BondsDF == 0] = np.nan
            BondsDF[BondsDF.abs() > 100] = np.nan
            BondsDF = BondsDF.ffill().sort_index()
            BondsDF.to_sql('Bonds_Prices', conn, if_exists='replace')

        if dataVec[1] == 1:

            EqsList = []
            for EqIndex in EqsTickers:
                print(EqIndex)
                df = investpy.get_index_historical_data(index=EqIndex[0], country=EqIndex[1], from_date=fromDate, to_date=toDate).reset_index().rename(
                    columns={"Date": "Dates", "Close": EqIndex[0]}).set_index('Dates')[EqIndex[0]]
                EqsList.append(df)

            EqsDF = pd.concat(EqsList, axis=1)
            EqsDF[EqsDF == 0] = np.nan
            EqsDF = EqsDF.ffill().sort_index()
            EqsDF.to_sql('Eqs_Prices', conn, if_exists='replace')

        if dataVec[2] == 1:

            FxList = []
            for fx in FxTickers:
                print(fx)
                name = fx.replace('/', '')
                df = investpy.get_currency_cross_historical_data(currency_cross=fx, from_date=fromDate,
                                                                 to_date=toDate).reset_index().rename(
                    columns={"Date": "Dates", "Close": name}).set_index('Dates')[name]
                FxList.append(df)

            FxDF = pd.concat(FxList, axis=1)
            FxDF[FxDF == 0] = np.nan
            FxDF = FxDF.ffill().sort_index()
            FxDF["JPYUSD"] = FxDF["USDJPY"].apply(lambda x: 1 / x)
            FxDF["CNYUSD"] = FxDF["USDCNY"].apply(lambda x: 1 / x)
            FxDF[['EURUSD', 'GBPUSD', 'USDJPY', 'USDCNY']].to_sql('Fx_Prices_raw', conn, if_exists='replace')
            FxDF[['EURUSD', 'GBPUSD', 'CNYUSD', 'JPYUSD']].to_sql('Fx_Prices', conn, if_exists='replace')

        if dataVec[3] == 1:

            CommoditiesList = []
            for comdty in CommoditiesTickers:
                print(comdty)
                name = comdty.replace('/', '')
                df = investpy.get_commodity_historical_data(commodity=comdty, from_date=fromDate,
                                                                 to_date=toDate).reset_index().rename(
                    columns={"Date": "Dates", "Close": name}).set_index('Dates')[name]
                CommoditiesList.append(df)

            CommoditiesDF = pd.concat(CommoditiesList, axis=1)
            CommoditiesDF[CommoditiesDF == 0] = np.nan
            CommoditiesDF = CommoditiesDF.ffill().sort_index()
            CommoditiesDF.to_sql('Commodities_Prices', conn, if_exists='replace')

        if dataVec[4] == 1:

            CustomList = []
            for customProduct in CustomTickers:
                print(customProduct)
                df = pd.read_csv(customProduct)
                df['Date'] = pd.to_datetime(df['Date'])
                CustomName = customProduct.split("/")[1].replace(" Historical Data.csv", "")
                df = df.rename(columns={"Date": "Dates", "Price": CustomName}).set_index('Dates')[CustomName].sort_index()
                CustomList.append(df)

            CustomDF = pd.concat(CustomList, axis=1)
            CustomDF[CustomDF == 0] = np.nan
            CustomDF = CustomDF.ffill().sort_index()
            CustomDF.to_sql('Custom_Prices', conn, if_exists='replace')

    elif mode == 'plot':

        dataAll = []
        for x in ['Bonds_Prices', 'Eqs_Prices', 'Fx_Prices', 'Commodities_Prices', 'Custom_Prices']:
            df = pd.read_sql('SELECT * FROM '+x, conn)
            if x == 'Bonds_Prices':
                df = df.rename(columns={"index": "Dates"})
            df['Dates'] = pd.to_datetime(df['Dates'])
            df = df.set_index('Dates', drop=True)
            dataAll.append(df)

        Prices = pd.concat(dataAll, axis=1)
        Prices.to_sql('Prices', conn, if_exists='replace')

        Rates = sl.d(Prices[BondsTickers]).fillna(0)
        Rates.to_sql('Rates', conn, if_exists='replace')

        PricesTrading = Prices.drop(BondsTickers, axis=1)
        rets = sl.dlog(PricesTrading).fillna(0)
        rets.to_sql('AssetsRets', conn, if_exists='replace')

        # TO PLOT
        Prices = pd.read_sql('SELECT * FROM Prices', conn).set_index('Dates', drop=True)
        Prices.index = [x.replace("00:00:00", "").strip() for x in Prices.index]

        for subset in ['A', 'B']:
            if subset == 'A':
                df = Prices[BondsTickers]
                ylabel = '$B_t$'
                returnTs = pd.read_sql('SELECT * FROM Rates', conn).set_index('Dates', drop=True)
                returnTs.index = [x.replace("00:00:00", "").strip() for x in returnTs.index]
                returnTs_ylabel = '$r_t$'
                ncolIn = 1
                legendSize = [17]
            else:
                df = Prices[[x for x in Prices.columns if x not in BondsTickers]].ffill()
                ylabel = '$S_t$'
                returnTs = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
                returnTs.index = [x.replace("00:00:00", "").strip() for x in returnTs.index]
                returnTs_ylabel = '$x_t$'
                ncolIn = 1
                legendSize = [14]
            # Plot 1
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            df.plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(Prices) + 1)
            mpl.pyplot.ylabel(ylabel, fontsize=32)
            plt.legend(loc=2, ncol=ncolIn, bbox_to_anchor=(1, 1.02), frameon=False, prop={'size': legendSize[0]})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.8, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()

            # Plot 2
            fig, ax = plt.subplots()
            mpl.pyplot.locator_params(axis='x', nbins=35)
            returnTs.plot(ax=ax)
            for label in ax.get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(45)
            ax.set_xlim(xmin=0.0, xmax=len(Prices) + 1)
            mpl.pyplot.ylabel(returnTs_ylabel, fontsize=32)
            plt.legend(loc=2, ncol=ncolIn, bbox_to_anchor=(1, 1.02), frameon=False, prop={'size': legendSize[0]})
            plt.subplots_adjust(top=0.95, bottom=0.2, right=0.8, left=0.08, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.grid()
            plt.show()

def LongOnly():
    dfAll = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    dfAll.drop(['CBOE Volatility Index'], axis=1, inplace=True)
    for subset in subPortfoliosList:
        df = dfAll[subset[0]]
        longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df), columns=["Sharpe"]).round(4)
        longOnlySharpes.to_sql('longOnlySharpes_'+subset[1], conn, if_exists='replace')

        subrsDf = pd.DataFrame(sl.E(df))
        print(subset[1], ",", np.sqrt(252) * sl.sharpe(subrsDf))
        subrsDf.to_sql('LongOnlyEWPrsDf_'+subset[1], conn, if_exists='replace')

        randomWalkPnl_subrsDf = sl.S(sl.sign(subrsDf)) * subrsDf
        print("Random Walk "+subset[1]+" Sharpe : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_subrsDf).round(4))

    rollSharpe_dfAll = sl.rollStatistics(dfAll, 'Sharpe', nIn=250)
    rollSharpe_dfAll.to_sql('rollSharpe_dfAll', conn, if_exists='replace')

    randomWalkPnl_Df = sl.S(sl.sign(dfAll)) * dfAll
    print("Random Walk dfAll Sharpe : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_Df).round(4))

    rsDf = pd.DataFrame(sl.E(dfAll))
    print("Total LO : ", np.sqrt(252) * sl.sharpe(rsDf))
    rsDf.to_sql('LongOnlyEWPrsDf', conn, if_exists='replace')

    randomWalkPnl_rsDf = sl.S(sl.sign(rsDf)) * rsDf
    print("Random Walk rsdfAll Sharpe : ", np.sqrt(252) * sl.sharpe(randomWalkPnl_rsDf).round(4))

def RiskParity():
    dfAll = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    dfAll.drop(['CBOE Volatility Index'], axis=1, inplace=True)

    SRollVol = np.sqrt(252) * sl.S(sl.rollStatistics(dfAll, method='Vol', nIn=250)) * 100
    SRollVolToPlot = SRollVol.copy()
    SRollVolToPlot.index = [x.replace("00:00:00", "").strip() for x in SRollVolToPlot.index]
    SRollVol.to_sql('SRollVol', conn, if_exists='replace')
    dfAll = (dfAll / SRollVol).replace([np.inf, -np.inf], 0).fillna(0)
    dfAll.loc["2006-01-06 00:00:00", "S&P 500"] = 0
    dfAll.to_sql('riskParityDF', conn, if_exists='replace')

    riskParitySharpesAll = pd.DataFrame(np.sqrt(252) * sl.sharpe(dfAll), columns=["Sharpe"]).round(4)
    rsDfAll = pd.DataFrame(sl.rs(dfAll))
    riskParitySharpesAll.to_sql('riskParitySharpeRatiosAll', conn, if_exists='replace')
    print("RiskParityEWPrsDfAll Sharpe = ", (np.sqrt(252) * sl.sharpe(rsDfAll)).round(4))

    for subset in subPortfoliosList:
        df = dfAll[subset[0]]
        rsDf = pd.DataFrame(sl.rs(df))
        rsDf.to_sql('riskParityDF_'+subset[1], conn, if_exists='replace')
        riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(rsDf), columns=["Sharpe"]).round(4)
        riskParitySharpes.to_sql('riskParitySharpeRatios_'+subset[1], conn, if_exists='replace')

def RollingSharpes(mode):
    if mode == 'Benchmark':
        dfAll = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
        dfAll.drop(['CBOE Volatility Index'], axis=1, inplace=True)
        dfAll_assets = sl.rollStatistics(dfAll, 'Sharpe')

        dfAll_assets.index = [x.replace("00:00:00", "").strip() for x in dfAll_assets.index]
        fig, ax = plt.subplots()
        mpl.pyplot.locator_params(axis='x', nbins=35)
        dfAll_assets.plot(ax=ax)
        for label in ax.get_xticklabels():
            label.set_fontsize(25)
            label.set_ha("right")
            label.set_rotation(45)
        ax.set_xlim(xmin=0.0, xmax=len(dfAll_assets) + 1)
        mpl.pyplot.ylabel("$sh_{A,i,t}$", fontsize=32)
        plt.legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 14})
        plt.subplots_adjust(top=0.95, bottom=0.2, right=0.8, left=0.08, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.grid()
        plt.show()

        LO_List = []
        for subset in subPortfoliosList:
            LO_List.append(pd.read_sql('SELECT * FROM LongOnlyEWPrsDf_'+subset[1], conn).set_index('Dates', drop=True))
        LO_DF = pd.concat(LO_List, axis=1)
        LO_DF.columns = ["Equities Futures", "Bonds Futures", "Commodities Futures", "Interest Rates Futures", "FX"]
        RP_List = []
        for subset in subPortfoliosList:
            RP_List.append(pd.read_sql('SELECT * FROM riskParityDF_'+subset[1], conn).set_index('Dates', drop=True))
        RP_DF = pd.concat(RP_List, axis=1)
        RP_DF.columns = LO_DF.columns

        dfList = [sl.rollStatistics(LO_DF, 'Sharpe'), sl.rollStatistics(RP_DF, 'Sharpe')]

        fig, ax = plt.subplots(sharex=True, nrows=len((dfList)), ncols=1)
        mpl.pyplot.locator_params(axis='x', nbins=35)
        titleList = ['(a)', '(b)']
        c = 0
        for df in dfList:
            df.index = [x.replace("00:00:00", "").strip() for x in df.index]
            df.plot(ax=ax[c])
            for label in ax[c].get_xticklabels():
                label.set_fontsize(25)
                label.set_ha("right")
                label.set_rotation(40)
            ax[c].set_xlim(xmin=0.0, xmax=len(df) + 1)
            # ax[c].set_title(titleList[c], y=1.0, pad=-20)
            ax[c].text(.5, .9, titleList[c], horizontalalignment='center', transform=ax[c].transAxes, fontsize=30)
            #ax[c].legend(loc=2, fancybox=True, frameon=True, shadow=True, prop={'weight': 'bold', 'size': 24})
            ax[c].legend(loc=2, bbox_to_anchor=(1, 1), frameon=False, prop={'size': 16})
            ax[c].grid()
            c += 1
        plt.subplots_adjust(top=0.95, bottom=0.15, right=0.82, left=0.08, hspace=0.1, wspace=0)
        plt.show()

def RunRollManifold(manifoldIn, universe):
    df = sl.fd(pd.read_sql('SELECT * FROM '+universe, conn).set_index('Dates', drop=True).fillna(0))

    for tw in twList:
        print("tw = ", tw)

        if manifoldIn == "DMAP_gDmapsRun":
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, len(spacialND), spacialND, Scaler='Standard')
        else:
            out = sl.AI.gRollingManifold(manifoldIn, df, tw, len(spacialND), spacialND, Scaler='Standard', ProjectionMode='Transpose')

        out[0].to_sql(manifoldIn + "_" + universe + '_df_tw_' + str(tw), conn, if_exists='replace')
        principalCompsDfList_Target = out[1][0]
        principalCompsDfList_First = out[1][1]
        principalCompsDfList_Last = out[1][2]
        out[2].to_sql(manifoldIn + "_" + universe + '_lambdasDf_tw_' + str(tw), conn, if_exists='replace')
        out[3].to_sql(manifoldIn + "_" + universe + '_sigmasDf_tw_' + str(tw), conn, if_exists='replace')
        for k in range(len(principalCompsDfList_Target)):
            principalCompsDfList_Target[k].to_sql(manifoldIn + "_" + universe + '_principalCompsDf_Target_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')
            principalCompsDfList_First[k].to_sql(manifoldIn + "_" + universe + '_principalCompsDf_First_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')
            principalCompsDfList_Last[k].to_sql(manifoldIn + "_" + universe + '_principalCompsDf_Last_tw_' + str(tw) + "_" + str(k), conn, if_exists='replace')

def gDMAP_TES(mode, universe, alphaChoice, lifting):

    df = sl.fd(pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)).fillna(0)
    df.drop(['CBOE Volatility Index'], axis=1, inplace=True)

    nD = 5

    if mode == 'create':
        # CREATE COVARIANCES LIST
        print("Calculating Covariance Matrices List ... ")
        start = 25
        covMatList = []
        alphaList = []
        for i in tqdm(range(start, len(df) + 1)):
            subDF = df.iloc[i - start:i, :]
            subCov = np.cov(subDF)
            alphaCov = kernel.choose_optimal_epsilon_BGH(squareform(pdist(subDF.T)))[0]
            alphaList.append(alphaCov)
            covMatList.append([subDF, subCov, alphaCov])
        pickle.dump(covMatList, open(universe+"_covMatList.p", "wb"))

        alphaDF = pd.DataFrame(alphaList)
        alphaDF.to_sql('alphaDF_' + universe, conn, if_exists='replace')
        #alphaDF = sl.fd(pd.read_sql('SELECT * FROM alphaDF_' + universe, conn).set_index('index', drop=True)).fillna(0)
        #alphaDF.plot()
        #plt.show()

    elif mode == 'run':

        covMatList = pickle.load(open(universe+"_covMatList.p", "rb"))
        # ROLL ON COVARIANCES LIST
        print("Rolling on Covariance Matrices List ... ")

        Loadings = [[] for j in range(nD)]
        alphaList = []
        st = 25
        for x in tqdm(range(st, len(covMatList)+1)):
            MegaCovList = covMatList[x - st:x]
            subDF = MegaCovList[-1][0]
            covList = [j[1] for j in MegaCovList]

            # CREATE SQUARE MATRIX Wt1t2
            sumKLdf = pd.DataFrame(np.zeros((len(covList), len(covList))))
            for t1 in range(len(covList)):
                for t2 in range(len(covList)):
                    cov1 = covList[t1]
                    cov2 = covList[t2]
                    sumKL = 0.5 * (np.trace(np.dot(np.linalg.pinv(cov1), cov2) - np.ones(cov1.shape[0]))
                                 + np.trace(np.dot(np.linalg.pinv(cov2), cov1) - np.ones(cov2.shape[0])))
                    #print(t1, t2, sumKL, np.exp(-sumKL/alpha))
                    sumKLdf.iloc[t1, t2] = sumKL

            if alphaChoice == 'sumKLMedian':
                alpha = np.median(sumKLdf)
            elif alphaChoice == 'BGH':
                alpha = MegaCovList[-1][-1]
            else:
                alpha = 100

            #print("alpha = ", alpha)
            alphaList.append([ subDF.index[-1], alpha])
            alphaDF = pd.DataFrame(alphaList, columns=['Dates','alpha']).set_index('Dates', drop=True)
            alphaDF.to_sql('alphaDF_'+universe+"_"+str(alphaChoice), conn, if_exists='replace')

            Wdf = np.exp(-sumKLdf/alpha)

            s = np.sqrt(Wdf.sum(axis=1))
            D = pd.DataFrame(0, index=s.index, columns=s.index, dtype=s.dtype)
            np.fill_diagonal(D.values, s)
            Wnorm = pd.DataFrame(np.dot(np.dot(D, Wdf), D))
            Wnorm = sl.fd(Wnorm.fillna(0))
            #print(Wnorm)
            #time.sleep(400)
            Utv, stv, VTtv = svd(Wnorm)

            Utv = pd.DataFrame(Utv)
            psitv = Utv
            for col in Utv.columns:
                'Building psi and phi projections'
                psitv[col] = Utv[col] * stv[col]

            targetPsiTv = psitv.iloc[:, :nD]

            if lifting == 'Temporal':
                targetPsiTv.index = subDF.index
                #print(targetPsiTv)

                Loadings[0].append(targetPsiTv.reset_index().iloc[-1])

                resDF = pd.DataFrame(Loadings[0]).rename(columns={'index': 'Dates'}).set_index('Dates', drop=True)
                resDF.to_sql('gDMAP_TES_' + universe + "_" + str(alphaChoice) + "_" + str(lifting), conn, if_exists='replace')

                #print(eigOut_tv)
                #time.sleep(400)

            elif lifting == 'LinearRegression':
                model = LinearRegression()
                model.fit(targetPsiTv, subDF)

                out = pd.DataFrame(model.coef_, index=subDF.columns)
                #print(out)
                #time.sleep(400)

                c = 0
                for col in out.columns:
                    eigOut_tv = out[col]
                    eigOut_tv['Dates'] = subDF.index[-1]

                    Loadings[c].append(eigOut_tv)

                    resDF = pd.DataFrame(Loadings[c]).set_index('Dates', drop=True)
                    resDF.to_sql('gDMAP_TES_' + universe + "_" + str(alphaChoice) + "_" + str(lifting) + "_" + str(c), conn, if_exists='replace')

                    c += 1

    elif mode == 'TradeProjections':

        sigList = []
        for tradingAssetsMode in ['Assets', 'RiskParity', 'ARIMA_Raw_Assets_100', 'ARIMA_Raw_Assets_200']:
            if tradingAssetsMode == 'RiskParity':
                df = sl.rp(df)
            elif tradingAssetsMode == 'ARIMA_Raw_Assets_100':
                df = pd.read_sql('SELECT * FROM Assets_arimaPnLDF_100_250', conn).set_index('Dates', drop=True).fillna(0)
            elif tradingAssetsMode == 'ARIMA_Raw_Assets_200':
                df = pd.read_sql('SELECT * FROM Assets_arimaPnLDF_200_250', conn).set_index('Dates', drop=True).fillna(0)

            ###########################################################################################################
            for runSet in ['First', 'Last']:
                for scenario in tqdm(range(12)):

                    if scenario == 0:
                        sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_AssetsRets_principalCompsDf_'+runSet+'_tw_250_0', conn).set_index('Dates', drop=True)
                        sig = sl.rowStoch(sig, mode='abs')
                        sig.name = "scenario_"+str(scenario)+"_"+"0"
                        sigList.append(sig)

                        for pr in [1,2,3,4]: #None, 1,2,3,4

                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_AssetsRets_principalCompsDf_'+runSet+'_tw_250_'+str(pr), conn).set_index('Dates', drop=True)
                                sub_sig.name = "scenario_" + str(scenario) + "_" + str(pr)
                                sigList.append(sub_sig)
                                #sig += sub_sig
                                sig += sl.rowStoch(sub_sig, mode='abs')
                                sig.name = "scenario_" + str(scenario) + "_0to" + str(pr)
                                sigList.append(sig)

                            except Exception as e:
                                print(e)

                    elif scenario == 1:
                        sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_AssetsRets_principalCompsDf_'+runSet+'_tw_250_0', conn).set_index('Dates', drop=True)
                        sig = sl.rowStoch(sig, mode='abs')
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)

                        for pr in [1,2,3,4]: #None, 1,2,3,4

                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_AssetsRets_principalCompsDf_'+runSet+'_tw_250_'+str(pr), conn).set_index('Dates', drop=True)
                                sub_sig.name = "scenario_" + str(scenario) + "_" + str(pr)
                                sigList.append(sub_sig)
                                sig += sl.rowStoch(sub_sig, mode='abs')
                                sig.name = "scenario_" + str(scenario) + "_0to" + str(pr)
                                sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 2:
                        sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_LinearRegression_0', conn).set_index('Dates', drop=True)
                        sig = sl.rowStoch(sig, mode='abs')
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)

                        for pr in [1,2,3,4]:  # 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_LinearRegression_'+str(pr), conn).set_index('Dates', drop=True)
                                sub_sig.name = "scenario_" + str(scenario) + "_" + str(pr)
                                sigList.append(sub_sig)
                                sig += sl.rowStoch(sub_sig, mode='abs')
                                sig.name = "scenario_" + str(scenario) + "_0to" + str(pr)
                                sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 3:

                        Temporal_sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_Temporal', conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Temporal_sig, mode='abs')
                        sig = medSig.iloc[:, 0]
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)

                        for pr in [1,2,3,4]: #None, 1,2,3,4

                            try:
                                sub_sig = medSig.iloc[:,pr]
                                sub_sig.name = "scenario_" + str(scenario) + "_" + str(pr)
                                sigList.append(sub_sig)
                                sig += sub_sig
                                sig.name = "scenario_" + str(scenario) + "_0to" + str(pr)
                                sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 4:

                        Rates_Sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_Rates_principalCompsDf_'+runSet+'_tw_250_0', conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')

                        for ratesKernel in ['U.S.','Germany','U.K.','China','Japan']:
                            sig = sl.rs(medSig.loc[:,[x for x in medSig.columns if ratesKernel in x]])
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]: #None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_Rates_principalCompsDf_'+runSet+'_tw_250_'+str(pr), conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')

                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sig = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 5:

                        Rates_Sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_Rates_principalCompsDf_'+runSet+'_tw_250_0', conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')

                        for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                            sig = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]: #None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_Rates_principalCompsDf_'+runSet+'_tw_250_'+str(pr), conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')

                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sig = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 6:

                        Rates_Sig = pd.read_sql('SELECT * FROM gDMAP_TES_Rates_sumKLMedian_LinearRegression_0', conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')

                        for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                            sig = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_Rates_principalCompsDf_' + runSet + '_tw_250_' + str(pr),conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')

                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sig = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 7:

                        preCursorParams = [25, 1, 'roll']
                        #preCursorParams = [25, 1, 'exp']
                        AssetsRets_Sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_AssetsRets_principalCompsDf_' + runSet + '_tw_250_0',conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(AssetsRets_Sig, mode='abs')

                        sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)
                        for c in sig.columns:
                            subSubSig = sig[c]
                            subSubSig.name = "scenario_" + str(scenario) + "_" + "0" + "_" + str(c)
                            sigList.append(subSubSig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_AssetsRets_principalCompsDf_' + runSet + '_tw_250_' + str(pr),conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')
                                sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr)
                                sigList.append(sig)

                                for c in sig.columns:
                                    subSubSig = sig[c]
                                    subSubSig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + str(c)
                                    sigList.append(subSubSig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 8:

                        preCursorParams = [25, 1, 'roll']
                        #preCursorParams = [25, 1, 'exp']
                        AssetsRets_Sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_AssetsRets_principalCompsDf_' + runSet + '_tw_250_0',conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(AssetsRets_Sig, mode='abs')

                        sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1],
                                           mode=preCursorParams[2])[1]
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)
                        for c in sig.columns:
                            subSubSig = sig[c]
                            subSubSig.name = "scenario_" + str(scenario) + "_" + "0" + "_" + str(c)
                            sigList.append(subSubSig)

                        for pr in [1, 2, 3, 4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_AssetsRets_principalCompsDf_' + runSet + '_tw_250_' + str(pr), conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')
                                sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                                sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr)
                                sigList.append(sig)

                                for c in sig.columns:
                                    subSubSig = sig[c]
                                    subSubSig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + str(c)
                                    sigList.append(subSubSig)

                            except Exception as e:
                                pass
                                # print(e)

                    elif scenario == 9:

                        preCursorParams = [25, 1, 'roll']
                        #preCursorParams = [25, 1, 'exp']
                        AssetsRets_Sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_LinearRegression_0',conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(AssetsRets_Sig, mode='abs')

                        sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                        sig.name = "scenario_" + str(scenario) + "_" + "0"
                        sigList.append(sig)
                        for c in sig.columns:
                            subSubSig = sig[c]
                            subSubSig.name = "scenario_" + str(scenario) + "_" + "0" + "_" + str(c)
                            sigList.append(subSubSig)

                        for pr in [1, 2, 3, 4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_LinearRegression_' + str(pr), conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')
                                sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                                sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr)
                                sigList.append(sig)

                                for c in sig.columns:
                                    subSubSig = sig[c]
                                    subSubSig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + str(c)
                                    sigList.append(subSubSig)

                            except Exception as e:
                                pass
                                # print(e)

                    elif scenario == 10:

                        preCursorParams = [25, 1, 'roll']
                        #preCursorParams = [25, 1, 'exp']
                        Rates_Sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_Rates_principalCompsDf_' + runSet + '_tw_250_0',conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')
                        for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                            sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                            sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_pyDmapsRun_Rates_principalCompsDf_' + runSet + '_tw_250_' + str(pr),conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')
                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 11:

                        #preCursorParams = [25, 1, 'roll']
                        preCursorParams = [25, 1, 'exp']
                        Rates_Sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_Rates_principalCompsDf_' + runSet + '_tw_250_0',conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')

                        for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                            sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                            sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM DMAP_gDmapsRun_Rates_principalCompsDf_' + runSet + '_tw_250_' + str(pr),conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')

                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 12:

                        preCursorParams = [25, 1, 'roll']
                        #preCursorParams = [25, 1, 'exp']
                        Rates_Sig = pd.read_sql('SELECT * FROM gDMAP_TES_Rates_sumKLMedian_LinearRegression_0', conn).set_index('Dates', drop=True)
                        medSig = sl.rowStoch(Rates_Sig, mode='abs')

                        for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                            sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                            sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                            sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                            sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                sub_sig = pd.read_sql('SELECT * FROM gDMAP_TES_Rates_sumKLMedian_LinearRegression_' + str(pr),conn).set_index('Dates', drop=True)
                                medSig += sl.rowStoch(sub_sig, mode='abs')
                                for ratesKernel in ['U.S.', 'Germany', 'U.K.', 'China', 'Japan']:
                                    sigKernel = sl.rs(medSig.loc[:, [x for x in medSig.columns if ratesKernel in x]])
                                    sig = sl.preCursor(df, sigKernel, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                    sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                    sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 13:

                        #preCursorParams = [25, 1, 'roll']
                        preCursorParams = [25, 1, 'exp']
                        Assets_Sig = pd.read_sql('SELECT * FROM gDMAP_TES_AssetsRets_sumKLMedian_Temporal',conn).set_index('Dates', drop=True)
                        Assets_MedSig = sl.rowStoch(Assets_Sig, mode='abs')
                        medSig = Assets_MedSig.iloc[:, 0]

                        sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                        sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                        sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                medSig += Assets_MedSig.iloc[:, pr]
                                sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

                    elif scenario == 14:

                        #preCursorParams = [25, 1, 'roll']
                        preCursorParams = [25, 1, 'exp']
                        Rates_Sig = pd.read_sql('SELECT * FROM gDMAP_TES_Rates_sumKLMedian_Temporal',conn).set_index('Dates', drop=True)
                        Rates_MedSig = sl.rowStoch(Rates_Sig, mode='abs')
                        medSig = Rates_MedSig.iloc[:, 0]
                        sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1], mode=preCursorParams[2])[1]
                        sig.name = "scenario_" + str(scenario) + "_" + "0_" + ratesKernel
                        sigList.append(sig)

                        for pr in [1,2,3,4]:  # None, 1,2,3,4
                            try:
                                medSig += Rates_MedSig.iloc[:, pr]
                                sig = sl.preCursor(df, medSig, nIn=preCursorParams[0], multiplier=preCursorParams[1],mode=preCursorParams[2])[1]
                                sig.name = "scenario_" + str(scenario) + "_" + "0to" + str(pr) + "_" + ratesKernel
                                sigList.append(sig)

                            except Exception as e:
                                pass
                                #print(e)

            print("Calculating PnLs ... ")
            ###########################################################################################################

            shList = []
            for sigDF in tqdm(sigList):
                pnl = df.mul(sl.S(sigDF), axis=0).fillna(0)
                rs_pnl = sl.rs(pnl, formatOut="DataFrameOut")
                sh_rs_pnl = sl.sharpe(rs_pnl, mode='processNA')
                sh_rs_pnl['StrategyName'] = sigDF.name
                sh_rs_pnl[["rawSharpe", "finalSharpe"]] *= np.sqrt(252)
                shList.append(sh_rs_pnl)

            shDF = pd.concat(shList)#, columns=['StrategyName', 'Strategies Raw_Sharpes', 'Strategies Length DF', 'Strategies Final_Sharpes']).set_index('StrategyName', drop=True)
            shDF.to_sql('shDF_'+tradingAssetsMode, conn, if_exists='replace')

pnlCalculator = 0

calcMode = 'runSerial'
#calcMode = 'read'
pnlCalculator = 0
probThr = 0.5
targetSystems = [2] #[0,1]

def ARIMAlocal(argList):
    selection = argList[0]
    df = argList[1].dropna()
    trainLength = argList[2]
    orderIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", orderIn, ", ", rw)

    Arima_Results = sl.ARIMA_Walk(df, trainLength, orderIn, rw)

    Arima_Results[0].to_sql(selection + '_ARIMA_testDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')
    Arima_Results[1].to_sql(selection + '_ARIMA_PredictionsDF_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn,
                            if_exists='replace')

    pickle.dump(Arima_Results[2], open("AR_Params/"+selection + '_ARIMA_arparamList_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw) +".p", "wb"))

    if pnlCalculator == 0:
        sig = sl.sign(Arima_Results[1])
        pnl = sig * Arima_Results[0]

    pnl.to_sql(selection + '_ARIMA_pnl_' + str(orderIn[0]) + str(orderIn[1]) + str(orderIn[2]) + '_' + str(rw), conn, if_exists='replace')
    print(selection, ",", trainLength, ",", orderIn, ", Sharpe = ", np.sqrt(252) * sl.sharpe(pnl))

def ARIMAonPortfolios(Portfolios, scanMode, mode):

    if Portfolios == 'Assets':

        allProjectionsDF = sl.fd(pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)).fillna(0)

    orderList = [(1,0,0),(2,0,0)]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for orderIn in orderList:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.1, orderIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(ARIMAlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for orderIn in orderList:
                arimaPnL_list = []
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM "' + selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw) + '"',
                                          conn).set_index('Dates', drop=True).iloc[round(0.1*len(allProjectionsDF)):]
                        pnl.columns = [selection]
                        arimaPnL_list.append(pnl.copy())
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252*100).set_index("index", drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                        stats = pd.concat([statsMat.iloc[0,:], statsMat.iloc[1,:]], axis=0)
                        stats.index = ["ARIMA_sh", "ARIMA_Mean", "ARIMA_tConf", "ARIMA_Std", "RW_sh", "RW_Mean", "RW_tConf", "RW_Std"]
                        stats[["ARIMA_tConf", "RW_tConf"]] = stats[["ARIMA_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["ttestPair_statistic"] = np.round(ttestPair.statistic,2)
                        stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue,2)
                        stats["order"] = str(orderIn[0])

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_ARIMA_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw))
                arimaPnLDF = pd.concat(arimaPnL_list, axis=1)
                arimaPnLDF.to_sql(Portfolios + '_arimaPnLDF_' +str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+ '_' + str(rw) , conn, if_exists='replace')
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(2)
            shDF.to_sql(Portfolios+'_sh_ARIMA_pnl_' + str(rw), conn, if_exists='replace')

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF'+ '_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        rw = 250
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF'+ '_' + str(rw), conn).set_index('index', drop=True)

        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_ARIMA_pnl_")
            selection = splitInfo[0]
            if 1==1:
            #if float(selection.split("_")[2]) <= 5:
                orderStr = str(splitInfo[1])
                orderIn = (int(orderStr[0]), int(orderStr[1]), int(orderStr[2]))
                processList.append([selection, allProjectionsDF[selection], 0.1, orderIn, rw])

        print("#ARIMA Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(ARIMAlocal, tqdm(processList))
        p.close()
        p.join()

def ClassificationProcess(argList):
    selection = argList[0]
    df = argList[1]
    params = argList[2]
    magicNum = argList[3]

    if calcMode in ['runSerial', 'runParallel']:
        print("Running gClassification")
        out = sl.AI.gClassification(df, params)

        out[0].to_sql('df_predicted_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[1].to_sql('df_real_price_class_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[2].to_sql('df_real_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[3].to_sql('df_predicted_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[4].to_sql('df_real_price_class_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')
        out[5].to_sql('df_real_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn, if_exists='replace')

    elif calcMode == 'read':
        print(selection)
        out = [
            pd.read_sql('SELECT * FROM df_predicted_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_class_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_train_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_predicted_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_class_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
            pd.read_sql('SELECT * FROM df_real_price_test_' + params["model"] + "_" + selection + "_" + str(magicNum), conn).set_index('Dates', drop=True),
        ]

    sig = out[3] # Predicted Price
    df_real_price_class_DF = out[4]
    df_real_price_test_DF = out[5]

    sigDF = sig.copy()
    if pnlCalculator == 0:
        probDF = sigDF[["Predicted_Proba_Test_0.0", "Predicted_Proba_Test_1.0", "Predicted_Proba_Test_2.0"]]
        probDF[probDF < probThr] = 0
        #sigDF['ProbFilter'] = sl.rs(probDF)
        sigDF['ProbFilter'] = sl.sign(sl.rs(probDF)-probThr)
        sigDF.loc[sigDF["ProbFilter"] < 0, "ProbFilter"] = np.nan
        sigDF.loc[sigDF["Predicted_Test_" + selection] > 1, "Predicted_Test_" + selection] = 0
        sigDF[selection] = sigDF["Predicted_Test_" + selection] * sigDF['ProbFilter']

    dfPnl = pd.concat([df_real_price_test_DF, sigDF[selection]], axis=1)
    dfPnl.columns = ["Real_Price", "Sig"]

    pnl = dfPnl["Real_Price"] * dfPnl["Sig"]
    print("PriorLength = ", len(pnl))
    pnl = pnl.dropna()
    print("PostLength = ", len(pnl))
    sh_pnl = np.sqrt(252) * sl.sharpe(pnl)
    print("selection = ", selection, ", Target System = ", magicNum, ", ", sh_pnl)

    pnl.to_sql('pnl_'+params['model']+'_' + selection + "_" + str(magicNum), conn, if_exists='replace')

def runClassification(Portfolios, scanMode, mode):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 1:

            paramsSetup = {
                "model": "RNN",
                "HistLag": 0,
                "InputSequenceLength": 25,  # 240
                "SubHistoryLength": 250,  # 760
                "SubHistoryTrainingLength": 250-1,  # 510
                "Scaler": "Standard",  # Standard
                "epochsIn": 100,  # 100
                "batchSIzeIn": 10,  # 16
                "EarlyStopping_patience_Epochs": 10,  # 10
                "LearningMode": 'static',  # 'static', 'online'
                "medSpecs": [
                    {"LayerType": "LSTM", "units": 25, "RsF": False, "Dropout": 0.25}
                ],
                "modelNum": magicNum,
                "CompilerSettings": ['adam', 'mean_squared_error'],
            }

        elif magicNum == 2:

            paramsSetup = {
                "model": "GPC",
                "HistLag": 0,
                "InputSequenceLength": 25,  # 240 (main) || 25 (Siettos) ||
                "SubHistoryLength": 250,  # 760 (main) || 250 (Siettos) ||
                "SubHistoryTrainingLength": 250 - 1,  # 510 (main) || 250-1 (Siettos) ||
                "Scaler": "Standard",  # Standard
                'Kernel': '1',
                "LearningMode": 'static',  # 'static', 'online'
                "modelNum": magicNum
            }

        return paramsSetup

    if Portfolios == 'Labels':
        allProjectionsDF = pd.read_sql('SELECT * FROM pnlDF_Scenario_0', conn).set_index('Dates', drop=True)
        print(allProjectionsDF)

    if scanMode == 'Main':

        if mode == "runSerial":
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    try:
                        ClassificationProcess([selection, allProjectionsDF[selection], params, magicNum])
                    except Exception as e:
                        print(e)

        elif mode == "runParallel":
            processList = []
            for magicNum in targetSystems:
                params = Architecture(magicNum)
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], params, magicNum])

            if calcMode == 'read':
                p = mp.Pool(2)
            else:
                p = mp.Pool(mp.cpu_count())
                #p = mp.Pool(len(processList))
            #result = p.map(ClassificationProcess, tqdm(processList))
            result = p.map(ClassificationProcess, processList)
            p.close()
            p.join()

        elif mode == "report":
            shList = []
            notProcessed = []
            for magicNum in targetSystems:
                if magicNum in [1]:
                    Classifier = "RNN"
                elif magicNum in [2]:
                    Classifier = "GPC"
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql(
                        'SELECT * FROM pnl_'+Classifier+'_' + selection + '_' + str(magicNum), conn).set_index('Dates', drop=True).dropna()

                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        sh = np.sqrt(252) * sl.sharpe(pnl)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252 * 100).set_index("index",drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        ttestPair = st.ttest_ind(pnl[selection].values, pnl['RW'].values, equal_var=False)
                        pnl_ttest_0 = st.ttest_1samp(pnl[selection].values, 0)
                        rw_pnl_ttest_0 = st.ttest_1samp(pnl['RW'].values, 0)
                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)

                        stats = pd.concat([statsMat.iloc[0, :], statsMat.iloc[1, :]], axis=0)
                        stats.index = ["Classifier_sh", "Classifier_Mean", "Classifier_tConf", "Classifier_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["Classifier_tConf", "RW_tConf"]] = stats[["Classifier_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["ttestPair_pvalue"] = np.round(ttestPair.pvalue,2)
                        stats["pnl_ttest_0_pvalue"] = np.round(pnl_ttest_0.pvalue, 2)
                        stats["rw_pnl_ttest_0_value"] = np.round(rw_pnl_ttest_0.pvalue, 2)
                        stats["Classifier"] = Classifier+str(magicNum)

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append('pnl_'+Classifier+'_' + selection + '_' + str(magicNum))

            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(2)
            shDF.to_sql(Portfolios+"_"+Classifier+"_sharpe", conn, if_exists='replace')
            print("shDF = ", shDF)

            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_'+Classifier, conn, if_exists='replace')
            print("notProcessedDF = ", notProcessedDF)

    elif scanMode == 'ScanNotProcessed':
        systemClass = 'GPC'
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_'+systemClass, conn).set_index('index', drop=True)
        print("len(notProcessedDF) = ", len(notProcessedDF))
        notProcessedList = []
        for idx, row in notProcessedDF.iterrows():
            Info = row['NotProcessedProjection'].replace("pnl_"+systemClass+"_", "")
            selection = Info[:-2]
            magicNum = Info[-1]
            params = Architecture(magicNum)
            print("Rerunning NotProcessed : ", selection, ", ", magicNum)
            notProcessedList.append([selection, allProjectionsDF[selection], params, magicNum])

        p = mp.Pool(mp.cpu_count())
        result = p.map(ClassificationProcess, tqdm(notProcessedList))
        p.close()
        p.join()

def DB_Handler():
    From_conn = sqlite3.connect('SmartGlobalAssetAllocation.db')
    To_conn = sqlite3.connect('SmartGlobalAssetAllocation_To.db')
    cursor = From_conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for tableTuple in cursor.fetchall():
        if tableTuple[0].split("_")[0] not in ["sig", "pnl", "sh", "rspnlDF", "rs"]:
            print(tableTuple)
            TableDF = pd.read_sql('SELECT * FROM ' + tableTuple[0], From_conn)
            TableDF = TableDF.set_index(TableDF.columns[0], drop=True)
            TableDF.to_sql(tableTuple[0], To_conn, if_exists='replace')

if __name__ == '__main__':
    #ProductsSearch()
    #DataHandler("run")
    #DataHandler("plot")

    #LongOnly()
    #RiskParity()

    #RollingSharpes('Benchmark')

    #RunRollManifold("DMAP_pyDmapsRun", 'AssetsRets')
    #RunRollManifold("DMAP_pyDmapsRun", 'Rates')
    #RunRollManifold("DMAP_gDmapsRun", 'AssetsRets')
    #RunRollManifold("DMAP_gDmapsRun", 'Rates')

    #gDMAP_TES("create", "AssetsRets", "", "")
    #gDMAP_TES("create", "Rates", "", "")
    #gDMAP_TES("run", "AssetsRets", 'sumKLMedian', 'LinearRegression')
    #gDMAP_TES("run", "Rates", 'sumKLMedian', 'LinearRegression')
    #gDMAP_TES("run", "AssetsRets", 'sumKLMedian', 'Temporal')
    #gDMAP_TES("run", "Rates", 'sumKLMedian', 'Temporal')

    gDMAP_TES("TradeProjections", "AssetsRets", 'sumKLMedian', '')

    #ARIMAonPortfolios('Assets', 'Main', 'run')
    #ARIMAonPortfolios('Assets', 'Main', 'report')
    #ARIMAonPortfolios('Assets', 'ScanNotProcessed', '')

    #runClassification("Labels", 'Main', "runSerial")
    #runClassification("Labels", 'Main', "report")

    #DB_Handler()
