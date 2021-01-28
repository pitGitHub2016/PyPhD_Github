from Slider import Slider as sl
import numpy as np, investpy, json
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('SmartGlobalAssetAllocation.db')
fromDate = '01/01/2005'
toDate = '25/11/2020'

def ProductsSearch():
    search_results = investpy.search(text='United States 10-Year Bond', n_results=100)
    # 'id_': 44336, 'name': 'CBOE Volatility Index'
    # 'id_': 8830, 'name': 'Gold Futures'
    # 'id_': 8895, 'name': 'Euro Bund Futures'
    # 'id_': 8880, 'name': 'US 10 Year T-Note Futures'
    # 'id_': 8986, 'name': 'Japan Government Bond Futures'
    # 'id_': 1073160, 'name': '10 Years Russian Federation Government Bond Future'
    for search_result in search_results:
        jResult = json.loads(str(search_result))
        print(jResult)
        #print(jResult["id_"])
        #search_result.retrieve_historical_data(from_date=fromDate, to_date=toDate)
        #print(search_result.data.head())
        #break

def DataHandler(mode):
    EquitiesIndicesList = [['DAX', 'Germany'], ["S&P 500", "United States"]]
    fxPairsList = ['USD/EUR']
    BondsList = ["U.S. 30Y","U.S. 10Y","U.S. 5Y","U.S. 2Y","U.S. 3M",
                 "Germany 30Y","Germany 10Y","Germany 5Y","Germany 2Y","Germany 3M"]
    CustomSearchProductsIDsAndNames = [[8830, 'Gold Futures'], [8895, 'Euro Bund Futures'], [8880, 'US 10 Year T-Note Futures']]
    if mode == 'run':
        dataAll = []

        for bond in BondsList:
            print(bond)
            df = investpy.get_bond_historical_data(bond=bond, from_date=fromDate, to_date=toDate).reset_index().rename(
                columns={"Date": "Dates", "Close": bond}).set_index('Dates')[bond]
            dataAll.append(df)

        for EqIndex in EquitiesIndicesList:
            print(EqIndex)
            df = investpy.get_index_historical_data(index=EqIndex[0], country=EqIndex[1], from_date=fromDate, to_date=toDate).reset_index().rename(
                columns={"Date": "Dates", "Close": EqIndex[0]}).set_index('Dates')[EqIndex[0]]
            dataAll.append(df)

        for fx in fxPairsList:
            print(fx)
            name = fx.replace('/', '')
            df = investpy.get_currency_cross_historical_data(currency_cross=fx, from_date=fromDate,
                                                             to_date=toDate).reset_index().rename(
                columns={"Date": "Dates", "Close": name}).set_index('Dates')[name]
            dataAll.append(df)

        for customProduct in CustomSearchProductsIDsAndNames:
            print(customProduct[1])
            search_results = investpy.search(text=customProduct[1], n_results=5)
            for search_result in search_results:
                jResult = json.loads(str(search_result))
                if ((jResult["id_"] == customProduct[0]) & (jResult["name"] == customProduct[1])):
                    df = search_result.retrieve_historical_data(from_date=fromDate, to_date=toDate).reset_index().rename(
                    columns={"Date": "Dates", "Close": customProduct[1]}).set_index('Dates')[customProduct[1]]
                    dataAll.append(df)

        dataDF = pd.concat(dataAll, axis=1)
        dataDF[dataDF==0] = np.nan
        dataDF = dataDF.ffill().sort_index()
        print(dataDF)
        dataDF.to_sql('Prices', conn, if_exists='replace')

    else:

        dataDF = pd.read_sql('SELECT * FROM Prices', conn).set_index('Dates', drop=True)

        Rates = sl.d(dataDF[BondsList]).fillna(0)
        Rates.to_sql('Rates', conn, if_exists='replace')

        rets = sl.dlog(dataDF.drop(BondsList, axis=1)).fillna(0)
        rets.to_sql('AssetsRets', conn, if_exists='replace')

        fig, ax = plt.subplots(nrows=2, ncols=1)
        sl.cs(rets).plot(ax=ax[0])
        sl.cs(Rates).plot(ax=ax[1])
        plt.show()

def LongOnly():
    df = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    longOnlySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    longOnlySharpes["Sharpe"] = "& " + longOnlySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    rsDf = pd.DataFrame(sl.rs(df))
    rsDf.to_sql('LongOnlyEWPrsDf', conn, if_exists='replace')
    print('LongOnly rsDf = ', (np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    pnl3 = sl.S(sl.sign(sl.ema(rsDf, nperiods=3))) * rsDf
    pnl5 = sl.S(sl.sign(sl.ema(rsDf, nperiods=5))) * rsDf
    pnl25 = sl.S(sl.sign(sl.ema(rsDf, nperiods=25))) * rsDf
    pnl50 = sl.S(sl.sign(sl.ema(rsDf, nperiods=50))) * rsDf
    pnl250 = sl.S(sl.sign(sl.ema(rsDf, nperiods=250))) * rsDf
    pnlSharpes3 = np.sqrt(252) * sl.sharpe(pnl3).round(4)
    pnlSharpes5 = np.sqrt(252) * sl.sharpe(pnl5).round(4)
    pnlSharpes25 = np.sqrt(252) * sl.sharpe(pnl25).round(4)
    pnlSharpes50 = np.sqrt(252) * sl.sharpe(pnl50).round(4)
    pnlSharpes250 = np.sqrt(252) * sl.sharpe(pnl250).round(4)
    print(pnlSharpes3)
    print(pnlSharpes5)
    print(pnlSharpes25)
    print(pnlSharpes50)
    print(pnlSharpes250)

    """
    fig, ax = plt.subplots()
    sl.cs(pnl3).plot(ax=fig.add_subplot(221), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(a)")
    sl.cs(pnl5).plot(ax=fig.add_subplot(222), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(b)")
    sl.cs(pnl25).plot(ax=fig.add_subplot(223), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(c)")
    sl.cs(pnl50).plot(ax=fig.add_subplot(224), legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    mpl.pyplot.xlabel("(d)")
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    csdf = sl.cs(df)
    csdf.index = [x.replace("00:00:00", "").strip() for x in csdf.index]
    fig, ax = plt.subplots()
    csdf.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14},  borderaxespad=0.)
    plt.show()
    """
    fig, ax = plt.subplots()
    csrsDf = sl.cs(rsDf)
    csrsDf.index = [x.replace("00:00:00", "").strip() for x in csrsDf.index]
    csrsDf.plot(ax=ax, legend=False) #title='Equally Weighted Portfolio'
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    plt.show()

def RiskParity():
    df = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    SRollVol = np.sqrt(252) * sl.S(sl.rollerVol(df, 250)) * 100
    SRollVolToPlot = SRollVol.copy()
    SRollVolToPlot.index = [x.replace("00:00:00", "").strip() for x in SRollVolToPlot.index]
    SRollVol.to_sql('SRollVol', conn, if_exists='replace')

    df = (df / SRollVol).replace([np.inf, -np.inf], 0).fillna(0)
    df[df > 5]=0
    df.to_sql('riskParityDF', conn, if_exists='replace')
    riskParitySharpes = pd.DataFrame(np.sqrt(252) * sl.sharpe(df).round(4), columns=["Sharpe"])
    riskParitySharpes["Sharpe"] = "& " + riskParitySharpes["Sharpe"].round(4).astype(str) + " \\\\"
    print(riskParitySharpes)
    rsDf = pd.DataFrame(sl.rs(df))
    riskParitySharpes.to_sql('riskParitySharpeRatios', conn, if_exists='replace')
    rsDf.to_sql('RiskParityEWPrsDf', conn, if_exists='replace')
    print("RiskParityEWPrsDf Sharpe = ", (np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    pnl3 = sl.S(sl.sign(sl.ema(rsDf, nperiods=3))) * rsDf
    pnl5 = sl.S(sl.sign(sl.ema(rsDf, nperiods=5))) * rsDf
    pnl25 = sl.S(sl.sign(sl.ema(rsDf, nperiods=25))) * rsDf
    pnl50 = sl.S(sl.sign(sl.ema(rsDf, nperiods=50))) * rsDf
    pnl250 = sl.S(sl.sign(sl.ema(rsDf, nperiods=250))) * rsDf
    pnlSharpes3 = np.sqrt(252) * sl.sharpe(pnl3).round(4)
    pnlSharpes5 = np.sqrt(252) * sl.sharpe(pnl5).round(4)
    pnlSharpes25 = np.sqrt(252) * sl.sharpe(pnl25).round(4)
    pnlSharpes50 = np.sqrt(252) * sl.sharpe(pnl50).round(4)
    pnlSharpes250 = np.sqrt(252) * sl.sharpe(pnl250).round(4)
    print(pnlSharpes3)
    print(pnlSharpes5)
    print(pnlSharpes25)
    print(pnlSharpes50)
    print(pnlSharpes250)

    csdf = sl.cs(df)
    csdf.index = [x.replace("00:00:00", "").strip() for x in csdf.index]
    csrsDf = sl.cs(rsDf)
    csrsDf.index = [x.replace("00:00:00", "").strip() for x in csrsDf.index]

    fig, ax = plt.subplots()
    SRollVolToPlot.iloc[51:,:].plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Annualised Rolling Volatilities (%)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    csdf.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, prop={'size': 14}, borderaxespad=0.)
    plt.show()

    fig, ax = plt.subplots()
    csrsDf.plot(ax=ax, legend=None)
    mpl.pyplot.ylabel("Cumulative Returns")
    plt.show()

def RunRollManifold(manifoldIn, target, mode):
    df = pd.read_sql('SELECT * FROM ' + target, conn).set_index('Dates', drop=True)#.iloc[-500:]

    if mode == 'Cumulative':
        df = sl.cs(df)

    if manifoldIn == 'PCA':
        out = sl.AI.gRollingManifold(manifoldIn, df, 25, 3, [0, 1, 2, 3, 4], Scaler='Standard', RollMode='ExpWindow') #ExpWindow
    elif manifoldIn == 'ICA':
        out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0, 1, 2, 3, 4], Scaler='Standard') # RollMode='ExpWindow'
    elif manifoldIn == 'LLE':
        out = sl.AI.gRollingManifold(manifoldIn, df, 25, 5, [0, 1, 2, 3, 4], LLE_n_neighbors=5, ProjectionMode='Transpose',
                                     RollMode='ExpWindow', Scaler='Standard')
    elif manifoldIn == "DMAP":
        out = sl.AI.gRollingManifold(manifoldIn, df, 250, 3, [0,1,2], ProjectionMode='Transpose')
    elif manifoldIn == 'DMD':
        out = sl.AI.gRollingManifold('DMD', df, 250, len(df.columns), range(len(df.columns)), contractiveObserver=1, DMAPS_sigma='bgh')

    out[0].to_sql(manifoldIn+'df'+"_"+target+mode, conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    out[3].to_sql(manifoldIn+"_"+target+mode+'_lambdasDf', conn, if_exists='replace')
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql(manifoldIn+"_"+target+mode+'_principalCompsDf_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql(manifoldIn+"_"+target+mode+'_exPostProjections_'+str(k), conn, if_exists='replace')

def ProjectionsPlots(manifoldIn, target, mode):
    df = pd.read_sql('SELECT * FROM ' + target, conn).set_index('Dates', drop=True)
    rng = [0,1,2,3,4]
    list = []
    for c in rng:
        try:
            loadings = pd.read_sql('SELECT * FROM ' + manifoldIn +"_"+target+mode+ '_principalCompsDf_' + str(c), conn).set_index('Dates', drop=True).fillna(0)
            if manifoldIn == "DMAP":
                projectionDF = sl.rs(df.mul(sl.S(sl.sign(loadings))))
            else:
                projectionDF = sl.rs(df.mul(sl.S(loadings)))
            list.append(projectionDF)
        except:
            pass
    exPostProjections = pd.concat(list, axis=1, ignore_index=True)
    exPostProjections.columns = [manifoldIn +"_"+target+mode+"_"+str(x) for x in exPostProjections.columns]

    print("exPostProjections Sh = ", np.sqrt(252) * sl.sharpe(exPostProjections))
    exPostProjections.to_sql(manifoldIn +"_"+target+mode+ '_RsExPostProjections', conn, if_exists='replace')
    print("exPostProjections rsSh = ", np.sqrt(252) * sl.sharpe(sl.rs(exPostProjections)))

    exPostProjections.index = [x.replace("00:00:00", "").strip() for x in exPostProjections.index]

    fig, ax = plt.subplots()
    sl.cs(exPostProjections).plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    mpl.pyplot.ylabel("Cumulative Return")
    plt.show()

    rsProjection = sl.cs(sl.rs(exPostProjections))
    rsProjection.name = '$Y_(s'+manifoldIn+')(t)$'
    fig, ax = plt.subplots()
    rsProjection.plot(ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, frameon=False, borderaxespad=0.)
    mpl.pyplot.ylabel("Cumulative Return")
    plt.show()

def ProjectionsTrade(mode):
    rets = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    Rates = pd.read_sql('SELECT * FROM Rates', conn).set_index('Dates', drop=True)
    #RollVol = (1 / (np.sqrt(252) * sl.S(sl.rollerVol(rets, 250)) * 100))
    #RollVol[RollVol>100] = 1
    #rets = rets.mul(sl.S(RollVol), axis=0).replace([np.inf, -np.inf], 0)
    print("rets Sh = ", np.sqrt(252) * sl.sharpe(rets))

    allProjections_AssetsRets_PCA = pd.read_sql('SELECT * FROM PCA_AssetsRets'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True)
    allProjections_Rates_PCA = pd.read_sql('SELECT * FROM PCA_Rates'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True)
    allExPostProjections_AssetsRets_PCA = pd.read_sql('SELECT * FROM PCA_AssetsRets'+mode+'_exPostProjections_0', conn).set_index('Dates', drop=True)
    allExPostProjections_Rates_PCA = pd.read_sql('SELECT * FROM PCA_Rates'+mode+'_exPostProjections_0', conn).set_index('Dates', drop=True)
    allprincipalCompsDf_AssetsRets_PCA = pd.read_sql('SELECT * FROM PCA_AssetsRets'+mode+'_principalCompsDf_0', conn).set_index('Dates', drop=True)
    allprincipalCompsDf_Rates_PCA = pd.read_sql('SELECT * FROM PCA_Rates'+mode+'_principalCompsDf_0', conn).set_index('Dates', drop=True)

    allProjections_AssetsRets_DMAP = pd.read_sql('SELECT * FROM DMAP_AssetsRets'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True)
    allProjections_Rates_DMAP = pd.read_sql('SELECT * FROM DMAP_Rates'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True)
    allExPostProjections_AssetsRets_DMAP = pd.read_sql('SELECT * FROM DMAP_AssetsRets'+mode+'_exPostProjections_0', conn).set_index('Dates', drop=True)
    allExPostProjections_Rates_DMAP = pd.read_sql('SELECT * FROM DMAP_Rates'+mode+'_exPostProjections_0', conn).set_index('Dates', drop=True)
    allprincipalCompsDf_AssetsRets_DMAP = pd.read_sql('SELECT * FROM DMAP_AssetsRets'+mode+'_principalCompsDf_0', conn).set_index('Dates', drop=True)
    allprincipalCompsDf_Rates_DMAP = pd.read_sql('SELECT * FROM DMAP_Rates'+mode+'_principalCompsDf_2', conn).set_index('Dates', drop=True)

    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allExPostProjections_AssetsRets_PCA[c])), axis=0) for c in allExPostProjections_AssetsRets_PCA.columns], axis=1)

    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allProjections_Rates_DMAP))), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP)+sl.rs(allProjections_Rates_DMAP))), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))+sl.sign(sl.rs(allProjections_Rates_DMAP))), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(allExPostProjections_AssetsRets_DMAP)), axis=0) # Good Gold (0)
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allExPostProjections_Rates_DMAP))), axis=0) # Good Bund/TY (1), Good UsdEur (2)
    #pnl = rets.mul(sl.S(sl.sign(allprincipalCompsDf_AssetsRets_DMAP)), axis=0) # Good UsdEur (1)
    pnl = rets.mul(sl.S(sl.sign(sl.rs(allprincipalCompsDf_AssetsRets_DMAP))), axis=0) # More tests 012
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(allprincipalCompsDf_Rates_DMAP))), axis=0) # Good dax/sp ? (0), Good total (rs) (1),
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allProjections_AssetsRets_DMAP[c])), axis=0) for c in allProjections_AssetsRets_DMAP.columns], axis=1)

    #### NO ML on these projections ####
    """
    #pnl = rets.mul(sl.S(allExPostProjections_AssetsRets_DMAP), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(allExPostProjections_AssetsRets_DMAP)), axis=0)
    #pnl = rets.mul(sl.S(sl.sign(sl.rs(Rates))), axis=0)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(Rates[c])), axis=0) for c in Rates.columns], axis=1)
    """

    #### Project DMAP and PCA MetaData On Assets ####
    """
    #pnl = pd.concat([rets.mul(sl.S(allprincipalCompsDf_AssetsRets_DMAP[c]), axis=0) for c in allprincipalCompsDf_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allprincipalCompsDf_AssetsRets_DMAP[c])), axis=0) for c in allprincipalCompsDf_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(allprincipalCompsDf_Rates_DMAP[c]), axis=0) for c in allprincipalCompsDf_Rates_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allprincipalCompsDf_Rates_DMAP[c])), axis=0) for c in allprincipalCompsDf_Rates_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(allExPostProjections_AssetsRets_DMAP[c]), axis=0) for c in allExPostProjections_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allExPostProjections_AssetsRets_DMAP[c])), axis=0) for c in allExPostProjections_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(allExPostProjections_Rates_DMAP[c]), axis=0) for c in allExPostProjections_Rates_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allExPostProjections_Rates_DMAP[c])), axis=0) for c in allExPostProjections_Rates_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(allProjections_AssetsRets_DMAP[c]), axis=0) for c in allProjections_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allProjections_AssetsRets_DMAP[c])), axis=0) for c in allProjections_AssetsRets_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(allProjections_Rates_DMAP[c]), axis=0) for c in allProjections_Rates_DMAP.columns], axis=1)
    #pnl = pd.concat([rets.mul(sl.S(sl.sign(allProjections_Rates_DMAP[c])), axis=0) for c in allProjections_Rates_DMAP.columns], axis=1)
    """

    manifoldProjection = "DMAP"
    eqs = rets[["DAX", "S&P 500"]]
    bonds = rets[["Euro Bund Futures", "US 10 Year T-Note Futures"]]
    usdeur = rets[["USDEUR"]]
    gold = rets[["Gold Futures"]]

    if manifoldProjection == "PCA":
        # mode ===> Cumulative
        pnleqs = eqs.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))+sl.sign(sl.rs(allProjections_Rates_PCA))), axis=0)
        pnlusdeur = usdeur.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)
        pnlgold = gold.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)
        pnlbonds = bonds.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)

        # mode ===> ""
        pnleqs = eqs.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))+sl.sign(sl.rs(allProjections_Rates_PCA))), axis=0)
        pnlusdeur = usdeur.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)
        pnlgold = gold.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)
        pnlbonds = bonds.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_PCA))), axis=0)

    elif manifoldProjection == "DMAP":
        pnleqs = eqs.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))+sl.sign(sl.rs(allProjections_Rates_DMAP))), axis=0)
        pnlusdeur = usdeur.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
        pnlgold = gold.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
        pnlbonds = bonds.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)

        # mode ===> ""
        #pnleqs = eqs.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))+sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
        pnleqs = eqs.mul(sl.S(sl.sign(sl.rs(allProjections_Rates_DMAP))), axis=0)
        pnlusdeur = usdeur.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
        pnlgold = gold.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)
        pnlbonds = bonds.mul(sl.S(sl.sign(sl.rs(allProjections_AssetsRets_DMAP))), axis=0)

    #pnl = pd.concat([pnleqs, pnlusdeur, pnlgold, pnlbonds], axis=1)
    #pnl.to_sql('ProjectionsTrade', conn, if_exists='replace')

    pnlSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4)
    #pnlSh = pnlSh[pnlSh.abs()>0.5]
    print(pnlSh)
    rspnlSh = (np.sqrt(252) * sl.sharpe(sl.rs(pnl))).round(4)
    print(rspnlSh)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    sl.cs(pnl).plot(ax=ax[0])
    sl.cs(sl.rs(pnl)).plot(ax=ax[1])
    plt.show()

def ARIMAonProjections(mode, ARIMAmode):
    # DMAP PAPER SELECTED #
    allProjectionsDF = pd.concat([pd.read_sql('SELECT * FROM PCA_AssetsRets'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True),
                                  pd.read_sql('SELECT * FROM DMAP_AssetsRets'+mode+'_RsExPostProjections', conn).set_index('Dates', drop=True)], axis=1)

    for OrderP in [1]:#[1, 3, 5]:
        orderIn = (OrderP, 0, 0)
        #orderIn = (OrderP, 0, 1)
        if ARIMAmode == "run":
            for selection in allProjectionsDF.columns:
                try:
                    print(selection)
                    Arima_Results = sl.ARIMA_Walk(allProjectionsDF[selection], 0.3, orderIn)#.iloc[-100:]

                    Arima_Results[0].to_sql(selection + '_ARIMA'+mode+'_testDF_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')
                    Arima_Results[1].to_sql(selection + '_ARIMA'+mode+'_PredictionsDF_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

                    sig = sl.sign(Arima_Results[1])

                    pnl = sig * Arima_Results[0]
                    pnl.to_sql(selection + '_ARIMA'+mode+'_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

                    print("ARIMA ("+str(orderIn[0])+str(orderIn[1])+str(orderIn[2])+") Sharpe = ", np.sqrt(252) * sl.sharpe(pnl))
                except Exception as e:
                    print("selection = ", selection, ", error : ", e)

        elif ARIMAmode == "report":
            shList = []
            for selection in allProjectionsDF.columns:
                pnl = pd.read_sql('SELECT * FROM ' + selection + '_ARIMA'+mode+'_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                medSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4).values[0]
                shList.append([selection, medSh])
            shDF = pd.DataFrame(shList, columns=['selection', 'sharpe']).set_index("selection", drop=True)
            print(shDF)
            shDF.to_sql('sh_ARIMA'+mode+'_pnl_'+str(orderIn[0])+str(orderIn[1])+str(orderIn[2]), conn, if_exists='replace')

def BetaRegressions(mode):
    rets = pd.read_sql('SELECT * FROM AssetsRets', conn).set_index('Dates', drop=True)
    Rates = pd.read_sql('SELECT * FROM Rates', conn).set_index('Dates', drop=True)
    PcaDF_AssetsRets = pd.read_sql('SELECT * FROM PCA_AssetsRets_RsExPostProjections', conn).set_index('Dates', drop=True)
    DmapDF_AssetsRets = pd.read_sql('SELECT * FROM DMAP_AssetsRets_RsExPostProjections', conn).set_index('Dates', drop=True)
    allProjectionsDF_AssetsRets = pd.concat([sl.rs(PcaDF_AssetsRets), sl.rs(DmapDF_AssetsRets)], axis=1)
    allProjectionsDF_AssetsRets.columns = ["PcaDF_AssetsRets", "DmapDF_AssetsRets"]
    PcaDF_Rates = pd.read_sql('SELECT * FROM PCA_AssetsRets_RsExPostProjections', conn).set_index('Dates', drop=True)
    DmapDF_Rates = pd.read_sql('SELECT * FROM DMAP_AssetsRets_RsExPostProjections', conn).set_index('Dates', drop=True)
    allProjectionsDF_Rates = pd.concat([sl.rs(PcaDF_Rates), sl.rs(DmapDF_Rates)], axis=1)
    allProjectionsDF_Rates.columns = ["PcaDF_Rates", "DmapDF_Rates"]

    if mode == 'rets':
        BetaScan = rets
        regressX = rets.columns
    elif mode == 'Rates':
        BetaScan = pd.concat([rets, Rates], axis=1)
        regressX = Rates.columns
    elif mode == 'PcaDF_AssetsRets':
        BetaScan = pd.concat([rets, PcaDF_AssetsRets], axis=1)
        regressX = PcaDF_AssetsRets.columns
    elif mode == 'DmapDF_AssetsRets':
        BetaScan = pd.concat([rets, DmapDF_AssetsRets], axis=1)
        regressX = DmapDF_AssetsRets.columns
    elif mode == 'allProjectionsDF_AssetsRets':
        BetaScan = pd.concat([rets, allProjectionsDF_AssetsRets], axis=1)
        regressX = allProjectionsDF_AssetsRets.columns
    elif mode == 'PcaDF_Rates':
        BetaScan = pd.concat([rets, PcaDF_Rates], axis=1)
        regressX = PcaDF_Rates.columns
    elif mode == 'DmapDF_Rates':
        BetaScan = pd.concat([rets, DmapDF_Rates], axis=1)
        regressX = DmapDF_Rates.columns
    elif mode == 'allProjectionsDF_Rates':
        BetaScan = pd.concat([rets, allProjectionsDF_Rates], axis=1)
        regressX = allProjectionsDF_Rates.columns

    BetaRegPnLlist = []
    for asset in rets:
        print(asset)
        betaReg = sl.BetaRegression(BetaScan, asset)
        regRHS = sl.rs(BetaScan[regressX].mul(betaReg[0][regressX]).fillna(0))
        medPnL = BetaScan[asset].mul(sl.S(sl.sign(regRHS)))
        medPnL.name = asset
        BetaRegPnLlist.append(medPnL)

    BetaRegPnLDF = pd.concat(BetaRegPnLlist, axis=1)
    BetaRegPnLSh = (np.sqrt(252) * sl.sharpe(BetaRegPnLDF)).round(4)
    print(BetaRegPnLSh)
    BetaRegPnLSh.to_sql("BetaReg_" + mode + '_sh', conn, if_exists='replace')

    sl.cs(BetaRegPnLDF).plot()
    # fig, ax = plt.subplots(nrows=3, ncols=1)
    # betasDF.plot(ax=ax[0], title="Betas")
    # betaReg[1].plot(ax=ax[1], title="RollVols")
    # sl.cs(BetaRegPnL).plot(ax=ax[1], title="Regress")
    # sl.cs(sl.rs(BetaRegPnL)).plot(ax=ax[2], title="Regress")
    plt.show()

def ContributionAnalysis():
    portfoliosList = []
    for SelectedPortfolio in ["ProjectionsTrade", "DMAP_AssetsRetsCumulative_4_ARIMACumulative_pnl_500"]:
        subPortfolio = sl.rs(pd.read_sql('SELECT * FROM '+SelectedPortfolio, conn).set_index('Dates', drop=True))
        subPortfolio.name = SelectedPortfolio
        portfoliosList.append(subPortfolio.iloc[round(0.3*len(subPortfolio)):])
    portfolios = pd.concat(portfoliosList, axis=1)
    portfolios = sl.ExPostOpt(portfolios)[0]

    pnl = portfolios
    #betaReg = sl.BetaRegression(portfolios, portfolios.columns[0])
    #regRHS = sl.rs(portfolios.mul(betaReg[0]).fillna(0))
    #pnl = portfolios.mul(sl.S(sl.sign(regRHS)), axis=0)

    pnlSh = (np.sqrt(252) * sl.sharpe(pnl)).round(4)
    print(pnlSh)
    rspnlSh = (np.sqrt(252) * sl.sharpe(sl.rs(pnl))).round(4)
    print(rspnlSh)

    fig, ax = plt.subplots(nrows=2, ncols=1)
    sl.cs(pnl).plot(ax=ax[0])
    sl.cs(sl.rs(pnl)).plot(ax=ax[1])
    plt.show()

#ProductsSearch()
#DataHandler("run")
#DataHandler("")
#LongOnly()
#RiskParity()

#RunRollManifold('PCA', "AssetsRets", "")
#RunRollManifold('PCA', "Rates", "")
#ProjectionsPlots('PCA', "AssetsRets", "")
#ProjectionsPlots('PCA', "Rates", "")

#RunRollManifold('PCA', "AssetsRets", "Cumulative")
#RunRollManifold('PCA', "Rates", "Cumulative")
#ProjectionsPlots('PCA', "AssetsRets", "Cumulative")
#ProjectionsPlots('PCA', "Rates", "Cumulative")

#RunRollManifold('DMAP', "AssetsRets", "")
#RunRollManifold('DMAP', "Rates", "")
#ProjectionsPlots('DMAP', "AssetsRets", "")
#ProjectionsPlots('DMAP', "Rates", "")

#RunRollManifold('DMAP', "AssetsRets", "Cumulative")
#RunRollManifold('DMAP', "Rates", "Cumulative")
#ProjectionsPlots('DMAP', "AssetsRets", "Cumulative")
#ProjectionsPlots('DMAP', "Rates", "Cumulative")

ProjectionsTrade("")
#ProjectionsTrade("Cumulative")

#BetaRegressions("rets")
#BetaRegressions("Rates") # good
#BetaRegressions("PcaDF_AssetsRets") # not good
#BetaRegressions("DmapDF_AssetsRets")
#BetaRegressions("allProjectionsDF_AssetsRets")
#BetaRegressions("PcaDF_Rates")
#BetaRegressions("DmapDF_Rates")
#BetaRegressions("allProjectionsDF_Rates")

#ARIMAonProjections("")
#ARIMAonProjections("Cumulative", "run")
#ARIMAonProjections("Cumulative", "report")

#ContributionAnalysis()