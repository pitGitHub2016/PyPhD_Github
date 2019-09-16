from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')

pd.read_csv('BasketTS.csv', delimiter=' ').to_sql('BasketTS', conn, if_exists='replace')
pd.read_csv('BasketGekko.csv', delimiter=' ').to_sql('BasketGekko', conn, if_exists='replace')

def DataBuilder():
    P = pd.read_csv('P.csv', header=None); P.columns = pd.read_csv('Basket.csv', delimiter=' ')['Names'].tolist()
    P['Dates'] = pd.read_csv('Dates.csv', header=None)
    P['Dates'] = pd.to_datetime(P['Dates'], infer_datetime_format=True)
    P = P.set_index('Dates', drop=True)
    P.to_sql('FxData', conn, if_exists='replace')

def DataSelect(set, mode):
    df = pd.read_sql('SELECT * FROM FxData', conn).set_index('Dates', drop=True)
    specNames = pd.read_sql('SELECT * FROM '+set, conn)['Names'].tolist()
    df = df[specNames]
    df = df.replace([np.inf, -np.inf, 0], np.nan).ffill()
    if mode == 'rets':
        df = sl.dlog(df).fillna(0)

    return df

def plotData(assetSel, set, mode):

    if assetSel == 'raw':
        df = DataSelect(set, 'rets')

    elif assetSel == 'projections':
        allProjections = []
        for pr in range(5):
            exPostProjections = pd.read_sql('SELECT * FROM ' + set + '_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)
            allProjections.append(sl.rs(exPostProjections))
        df = pd.concat(allProjections, axis=1, ignore_index=True)
        df.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    fig, ax = plt.subplots()
    if mode == 'raw':
        csDf = sl.cs(df)
        csDf.plot(ax=ax, title='Assets')

        from mpl_toolkits.mplot3d import Axes3D
        cols = ['P0', 'P1', 'P2']
        threedee = plt.figure().gca(projection='3d')
        threedee.scatter(csDf[cols[0]], csDf[cols[1]], csDf[cols[2]])
        plt.title('3D Plot of ' + set + ' Projections')
        threedee.set_xlabel(cols[0]); threedee.set_ylabel(cols[1]); threedee.set_zlabel(cols[2])
        plt.show()

    elif mode == 'rv':
        sl.cs(sl.RV(df)).plot(ax=ax, title=set + ' Relative Values')
    plt.show()

def RunRollManifoldOnFXPairs(set, manifoldIn):
    df = DataSelect(set, 'rets')

    if manifoldIn == 'PCA':
        out = sl.AI.gRollingManifold(manifoldIn, df, 50, 5, [0,1,2,3,4])
    elif manifoldIn == 'LLE':
        out = sl.AI.gRollingManifold(manifoldIn, df, 50, 5, [0, 1, 2, 3, 4], LLE_n_neighbors=15,
                                     ProjectionMode='Transpose')

    out[0].to_sql('df', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql(manifoldIn+'_principalCompsDf_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql(manifoldIn+'_exPostProjections_'+str(k), conn, if_exists='replace')

def LongOnly(set):
    df = DataSelect(set, 'rets')
    longOnlySharpes = (np.sqrt(252) * sl.sharpe(df)).round(4)
    longOnlySharpes.to_sql('LongOnlySharpeRatios', conn, if_exists='replace')
    rsDf = sl.rs(df)
    print((np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    #sl.cs(df).plot(title='Cumulative Returns of Contributions')
    sl.cs(rsDf).plot(title='Equally Weighted Portfolio')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def RiskParity(set):
    df = DataSelect(set, 'rets')
    SRollVol = np.sqrt(252) * (sl.S(sl.rollerVol(df, 25)))
    SRollVol.to_sql('SRollVol', conn, if_exists='replace')
    df = df / SRollVol
    df.to_sql('riskParityDF', conn, if_exists='replace')
    riskParitySharpes = (np.sqrt(252) * sl.sharpe(df)).round(4)
    riskParitySharpes.to_sql('riskParitySharpeRatios', conn, if_exists='replace')
    rsDf = sl.rs(df)
    print((np.sqrt(252) * sl.sharpe(rsDf)).round(4))

    #sl.cs(df).plot(title='Cumulative Returns of Contributions')
    #sl.cs(df['EURCHF']).plot(title='Cumulative Returns of Contributions')
    SRollVol.iloc[51:,:].plot(title='Annualized 25 Days Rolling Volatilities of Contributions (Shifted by 1 Trading Day)')
    #sl.cs(rsDf).plot(title='Equally Weighted Portfolio')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def CorrMatrix(set):
    df = DataSelect(set, 'rets')
    mat = sl.CorrMatrix(df)
    mat.to_sql('CorrMatrix', conn, if_exists='replace')
    print(mat[mat!=1].abs().mean().mean())

    #sl.correlation_matrix_plot(df)

def semaOnProjections(manifoldIn, L, mode, shThr, prIn):

    if mode == 'classic':
        allProjections = []
        for pr in range(5):
            exPostProjections = pd.read_sql('SELECT * FROM '+manifoldIn+'_exPostProjections_'+str(pr), conn).set_index('Dates', drop=True)
            allProjections.append(sl.rs(exPostProjections))
        allProjectionsDF = pd.concat(allProjections, axis=1, ignore_index=True)
        #allProjectionsDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']; allProjectionsDF = sl.RV(allProjectionsDF)

    elif mode == 'signCorrection':
        allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_SignCorrected_E_exPostProjections',
                                       conn).set_index('Dates', drop=True)

    elif mode == 'ADF':
        allProjectionsDF = pd.read_sql('SELECT * FROM ADF_pvalue_Filtered_' + manifoldIn + '_Projections', conn).set_index('Dates', drop=True)

    elif mode == 'enhanced':
        allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_' + str(prIn),
                                       conn).set_index('Dates', drop=True)

    elif mode == 'enhanced_signCorrected':
        allProjectionsDF = pd.read_sql(
            'SELECT * FROM ' + manifoldIn + '_SignCorrected_E_exPostProjections_' + str(prIn), conn).set_index('Dates',
                                                                                                               drop=True)

    allProjectionsDF = allProjectionsDF.fillna(0)
    allProjectionsDF.to_sql(manifoldIn+'_'+mode+'_allProjectionsDF', conn, if_exists='replace')

    ######### Raw Projections Trading #########
    print('ProjectionsSharpes')
    print((np.sqrt(252) * sl.sharpe(allProjectionsDF)).round(4))

    pnl = sl.S(sl.sema(allProjectionsDF, nperiods=L)) * allProjectionsDF
    #pnl = sl.ExPostOpt(pnl)[0]
    print('semaPnLSharpe')
    print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
    pnl = sl.rs(pnl)
    print('sema rs Sharpe : ', (np.sqrt(252) * sl.sharpe(pnl)).round(4))

    ######### ExPostOpt Projections Trading #########
    print('ExPostOpt exPostProjectionsSharpes')
    print((np.sqrt(252) * sl.sharpe(sl.ExPostOpt(allProjectionsDF)[0])).round(4))
    print('rsExPostOpt exPostProjectionsSharpes')
    print((np.sqrt(252) * sl.sharpe(sl.rs(sl.ExPostOpt(allProjectionsDF)[0]))).round(4))

    pnl = sl.ExPostOpt(sl.S(sl.sema(allProjectionsDF, nperiods=L)) * allProjectionsDF)[0]
    print('ExPostOpt semaPnLSharpe')
    print(((np.sqrt(252) * sl.sharpe(pnl)).round(4)))
    pnl = sl.rs(pnl)
    print('ExPostOpt Sema rs Sharpe : ', (np.sqrt(252) * sl.sharpe(pnl)).round(4))

    ######### Rolling Sharpe Filtered Projections Trading #########
    RollShProjections = sl.S(sl.rollSh(allProjectionsDF, window=50)).fillna(0)
    RollShProjections.to_sql(manifoldIn + '_' + mode + '_RollShProjections', conn, if_exists='replace')
    #RollShProjections = sl.S(sl.expSh(allProjectionsDF, window=50))

    DFRollShFilteredPositive = allProjectionsDF[RollShProjections > shThr].fillna(0)
    DFRollShFilteredNegative = allProjectionsDF[RollShProjections < shThr].fillna(0)

    DFRollShFilteredPnl = pd.concat([sl.rs(DFRollShFilteredPositive), sl.rs(DFRollShFilteredNegative)], axis=1, ignore_index=True)
    print('DFRollShFilteredPnlSharpe')
    print(((np.sqrt(252) * sl.sharpe(DFRollShFilteredPnl)).round(4)))
    rsDFRollShFilteredPnl = sl.rs(DFRollShFilteredPnl)
    print('rsDFRollShFilteredPnlSharpe')
    print(((np.sqrt(252) * sl.sharpe(rsDFRollShFilteredPnl)).round(4)))

    semaDFRollShFilteredPnl = sl.ExPostOpt(sl.S(sl.sema(DFRollShFilteredPnl, nperiods=L)) * DFRollShFilteredPnl)[0]
    print('semaDFRollShFilteredPnl')
    print(((np.sqrt(252) * sl.sharpe(semaDFRollShFilteredPnl)).round(4)))
    rsSemaDFRollShFilteredPnl = sl.rs(semaDFRollShFilteredPnl)
    print('rsSemaDFRollShFilteredPnl')
    print(((np.sqrt(252) * sl.sharpe(rsSemaDFRollShFilteredPnl)).round(4)))

    DFRollShFilteredPositive.to_sql(manifoldIn + '_' + mode + '_DFRollShFilteredPositive', conn, if_exists='replace')
    DFRollShFilteredNegative.to_sql(manifoldIn + '_' + mode + '_DFRollShFilteredNegative', conn, if_exists='replace')

    fig = plt.figure()
    #fig, ax = plt.subplots()
    allProjectionsDF.plot(ax=fig.add_subplot(241), title='Projections Returns')
    sl.cs(allProjectionsDF).plot(ax=fig.add_subplot(242), title='Projections Cumulative Returns')
    sl.cs(pnl).plot(ax=fig.add_subplot(243), title='EMA Trading PnL with Lag = '+str(L))
    RollShProjections.plot(ax=fig.add_subplot(244), title='Rolling Sharpes of Projections Returns')
    sl.cs(DFRollShFilteredPnl).plot(ax=fig.add_subplot(245), title='Rolling Sharpe Filtered Projections')
    sl.cs(rsDFRollShFilteredPnl).plot(ax=fig.add_subplot(246), title='Portfolio of RollSharpe Projections')
    sl.cs(semaDFRollShFilteredPnl).plot(ax=fig.add_subplot(247), title='EMA on RollSharpe Projections')
    sl.cs(rsSemaDFRollShFilteredPnl).plot(ax=fig.add_subplot(248), title='Portfolio of EMA on RollSharpe Projections')
    plt.legend(); plt.show()

def StationarityOnProjections(manifoldIn, mode):
    allProjections = []
    for pr in range(5):
        exPostProjections = pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)
        allProjections.append(sl.rs(exPostProjections))
    allProjectionsDF = pd.concat(allProjections, axis=1, ignore_index=True)
    allProjectionsDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    if mode == 'build':

        out = sl.Stationarity(allProjectionsDF, 49, 'roll', multi=1)
        out[0].to_sql('ADF_Test_'+manifoldIn, conn, if_exists='replace')
        out[1].to_sql('Pval_Test_'+manifoldIn, conn, if_exists='replace')
        out[2].to_sql('critVal_Test_'+manifoldIn, conn, if_exists='replace')

    elif mode == 'filter':
        #adf = pd.read_sql('SELECT * FROM ADF_Test_' + manifoldIn, conn)
        adf = pd.read_sql('SELECT * FROM Pval_Test_' + manifoldIn, conn).set_index('Dates', drop=True)
        #adf = pd.read_sql('SELECT * FROM critVal_Test_' + manifoldIn, conn)
        #fig, ax = plt.subplots()
        #adf.plot(ax=ax, title='critVal_Test_' + manifoldIn)
        #plt.show()

        ADFfilteredDF = adf[adf > 0.05].fillna(0)
        ADFfilteredDF.columns = allProjectionsDF.columns
        ADFfilteredDF = ADFfilteredDF * allProjectionsDF
        ADFfilteredDF.to_sql('ADF_pvalue_Filtered_' + manifoldIn + '_Projections', conn, if_exists='replace')
        fig, ax = plt.subplots()
        sl.cs(ADFfilteredDF).plot(ax=ax, title='ADF pvalue Filtered ' + manifoldIn + ' Projections')
        plt.show()

def CustomPortfolioOfProjections(manifoldIn, weights):
    EMApnl = []
    for pr in range(5):
        exPostProjections = pd.read_sql('SELECT * FROM '+manifoldIn+'_exPostProjections_'+str(pr), conn).set_index('Dates', drop=True) * weights[pr]
        EMApnl.append(exPostProjections)

    EMApnlDF = sl.rs(pd.concat(EMApnl, axis=1, ignore_index=True))
    print((np.sqrt(252) * sl.sharpe(EMApnlDF)).round(4))

    #fig, ax = plt.subplots(figsize=(19.2, 10.8))
    fig, ax = plt.subplots()
    #sl.cs(exPostProjections).plot(ax=ax)
    #sl.cs(sl.rs(exPostProjections)).plot(ax=ax)
    sl.cs(EMApnlDF).plot(ax=ax)
    plt.legend(); plt.show()

def ProjectionsPlots(manifoldIn):
        list = []
        for c in range(5):
            list.append(sl.rs(pd.read_sql('SELECT * FROM '+manifoldIn+'_exPostProjections_'+str(c), conn).set_index('Dates', drop=True).fillna(0)))
        exPostProjections = pd.concat(list, axis=1, ignore_index=True)
        exPostProjections.columns = ['P0', 'P1', 'P2', 'P3', 'P4']
        exPostProjections.to_sql(manifoldIn + '_RsExPostProjections', conn, if_exists='replace')

        sl.cs(exPostProjections).plot(title=manifoldIn+' Projections')
        #sl.cs(rsDf).plot(title='Equally Weighted Portfolio')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

def ARIMAonPCAProjections(manifoldIn):
    allProjections = []
    for pr in range(5):
        exPostProjections = pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_' + str(pr), conn).set_index('Dates', drop=True)
        allProjections.append(sl.rs(exPostProjections))
    allProjectionsDF = pd.concat(allProjections, axis=1, ignore_index=True)
    allProjectionsDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']

    orderList = [(5, 0, 0), (5, 1, 1), (10, 1, 1)]
    Arima_Results = sl.ARIMA_predictions(allProjectionsDF, start=49, mode='roll', opt='BIC', orderList=orderList, multi=0, indextype=1)
    Arima_Results[0].to_sql(manifoldIn+'_ARIMA_Predictions_OrderList0', conn, if_exists='replace')
    Arima_Results[1].to_sql(manifoldIn+'_ARIMA_stderrDF_OrderList0', conn, if_exists='replace')
    Arima_Results[2].to_sql(manifoldIn+'_ARIMA_errDF_OrderList0', conn, if_exists='replace')
    Arima_Results[3].to_sql(manifoldIn+'_ARIMA_confDF_OrderList0', conn, if_exists='replace')

    #Backtest = sl.BacktestPnL.ModelPnL(sl.Models(exPostProjections.copy()).ARIMA_signal(start=49, mode='roll', opt=(3,1,0), multi=0, indextype=1), retmode=1)
    #print(Backtest)
    #fig, ax = plt.subplots(figsize=(19.2, 10.8))
    #sl.cs(Backtest).plot(ax=ax, c='green')
    #exPostProjections.plot(ax=ax, c='blue')
    #plt.show()

def getProjectionsAngles(manifoldIn):
    df = pd.read_sql('SELECT * FROM df', conn).set_index('Dates', drop=True)
    angles = [[] for j in range(5)]
    for pr in range(5):
        print(pr)
        pr0 = pd.read_sql('SELECT * FROM ' + manifoldIn + '_principalCompsDf_' + str(pr), conn).set_index('Dates',
                                                                                                          drop=True)
        for idx, row in pr0.iterrows():
            # indAngle = np.degrees(sl.py_ang(pr0.loc[idx].values, df.loc[idx].values, 0))
            indAngle = np.degrees(sl.py_ang(pr0.loc[idx].values, df.loc[idx].values, 2))
            # indAngle = sl.angle_clockwise(pr0.loc[idx].values, df.loc[idx].values)
            # indAngle = sl.py_ang(pr0.loc[idx].values, df.loc[idx].values, 2)
            angles[pr].append(indAngle)
        # angles.append(np.degrees(sl.py_ang(pr0.iloc[3250].values, df.iloc[3250].values, 2)))
    anglesDF = pd.DataFrame(angles).T
    anglesDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']
    anglesDF.index = df.index
    anglesDF.to_sql(manifoldIn + '_Projections_RollingDotProds_to_df', conn, if_exists='replace')
    anglesDF.plot();
    plt.show()

    # .apply(lambda x: str(x).rjust(ASINcharLimit, '0'))

def SignCorrectionProjections(manifoldIn):
    df0 = pd.read_sql('SELECT * FROM df', conn).set_index('Dates', drop=True)
    signAngles = pd.read_sql('SELECT * FROM ' + manifoldIn + '_Projections_RollingDotProds_to_df', conn).set_index('Dates', drop=True)
    #signFilter = sl.rs(signAngles)
    #signFilter = sl.rs(sl.sign(signAngles))
    #signFilter = sl.sign(sl.rs(sl.sign(df)))
    #signFilter = sl.sign(sl.E(signAngles))
    #sumDF.plot(); plt.show()

    l = []
    for pr in range(5):
        print(pr)
        pr0 = pd.read_sql('SELECT * FROM ' + manifoldIn + '_principalCompsDf_' + str(pr), conn).set_index('Dates',
                                                                                                          drop=True)
        # sl.cs(sl.rs(sl.S(pr0) * df0)).plot(); plt.show()
        for col in pr0.columns:
            pr0[col] = pr0[col] * signFilter
        projection = sl.S(pr0) * df0
        projection.to_sql(manifoldIn + '_SignCorrected_E_exPostProjections_' + str(pr), conn, if_exists='replace')
        l.append(sl.rs(projection))
        # sl.cs(sl.rs(sl.S(pr0) * df0)).plot(title=str(pr)); plt.show()

    signfillDF = pd.concat(l, axis=1, ignore_index=True)
    signfillDF.columns = ['P0', 'P1', 'P2', 'P3', 'P4']
    signfillDF.to_sql(manifoldIn + '_SignCorrected_E_exPostProjections', conn, if_exists='replace')

#DataBuilder()

#CorrMatrix('BasketGekko')

#RunRollManifoldOnFXPairs('BasketGekko', 'PCA')
# RunRollManifoldOnFXPairs('BasketGekko', 'LLE')

# plotData('raw', 'BasketGekko', 'rv')
#plotData('projections', 'LLE', 'raw')

#LongOnly('BasketGekko')
#RiskParity('BasketGekko')

#semaOnProjections('PCA', 3, mode='classic', shThr=0, prIn=0)
#semaOnProjections('LLE', 3, mode='ADF', shThr=0, prIn=0)
#semaOnProjections('PCA', 500, mode='enhanced', shThr=1, prIn=4)
#semaOnProjections('LLE', 3, mode='classic', shThr=0, prIn=0)
#semaOnProjections('LLE', 3, mode='enhanced', shThr=0, prIn=1)
# semaOnProjections('PCA', 3, mode='signCorrection', shThr=0, prIn=0)
# semaOnProjections('LLE', 3, mode='signCorrection', shThr=0, prIn=0)
# semaOnProjections('PCA', 3, mode='enhanced_signCorrected', shThr=0, prIn=0)
#semaOnProjections('LLE', 3, mode='enhanced_signCorrected', shThr=0, prIn=0)

#ProjectionsPlots('PCA')
#ProjectionsPlots('LLE')

#CustomPortfolioOfProjections('PCA', [1, -1, 0,0,0])

#StationarityOnProjections('LLE', 'filter')
#StationarityOnProjections('LLE')

#ARIMAonPCAProjections('PCA')

# getProjectionsAngles('PCA')
# getProjectionsAngles('LLE')

SignCorrectionProjections('PCA')
#SignCorrectionProjections('LLE')
