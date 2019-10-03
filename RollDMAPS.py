from Slider import Slider as sl
import numpy as np
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodDataDMAPS.db')

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
    specNames = pd.read_sql('SELECT * FROM ' + set, conn)['Names'].tolist()
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
            exPostProjections = pd.read_sql('SELECT * FROM ' + set + '_exPostProjections_' + str(pr), conn).set_index(
                'Dates', drop=True)
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
        threedee.set_xlabel(cols[0]);
        threedee.set_ylabel(cols[1]);
        threedee.set_zlabel(cols[2])
        plt.show()

    elif mode == 'rv':
        sl.cs(sl.RV(df)).plot(ax=ax, title=set + ' Relative Values')
    plt.show()

def RunRollManifoldOnFXPairs(set, manifoldIn):
    df = DataSelect(set, 'rets')

    out = sl.AI.gRollingManifold(manifoldIn, df, 50, 5, [0, 1, 2, 3, 4], ProjectionMode='Transpose')

    out[0].to_sql('df', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql(manifoldIn+'_principalCompsDf_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql(manifoldIn+'_exPostProjections_'+str(k), conn, if_exists='replace')

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
        # allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_SignCorrected_P0_exPostProjections', conn).set_index('Dates', drop=True)
        # allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_SignCorrected_E_ofWX_exPostProjections', conn).set_index('Dates', drop=True)
        # allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_SignCorrected_P0_ofWX_exPostProjections', conn).set_index('Dates', drop=True)

    elif mode == 'ADF':
        allProjectionsDF = pd.read_sql('SELECT * FROM ADF_pvalue_Filtered_' + manifoldIn + '_Projections', conn).set_index('Dates', drop=True)

    elif mode == 'enhanced':
        allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_exPostProjections_' + str(prIn),
                                       conn).set_index('Dates', drop=True)

    elif mode == 'enhanced_signCorrected':
        # allProjectionsDF = pd.read_sql('SELECT * FROM ' + manifoldIn + '_SignCorrected_E_exPostProjections_' + str(prIn), conn).set_index('Dates', drop=True)
        allProjectionsDF = pd.read_sql(
            'SELECT * FROM ' + manifoldIn + '_SignCorrected_P0_exPostProjections_' + str(prIn), conn).set_index('Dates',
                                                                                                                drop=True)

    allProjectionsDF = allProjectionsDF.fillna(0)
    allProjectionsDF.to_sql(manifoldIn+'_'+mode+'_allProjectionsDF', conn, if_exists='replace')

    ######### Raw Projections Trading #########
    print('ProjectionsSharpes')
    print((np.sqrt(252) * sl.sharpe(allProjectionsDF)).round(4))
    print('rsProjectionsSharpes')
    print((np.sqrt(252) * sl.sharpe(sl.rs(allProjectionsDF))).round(4))
    sl.cs(sl.rs(allProjectionsDF)).plot();
    plt.show()

    pnl = sl.S(sl.ema(allProjectionsDF, nperiods=L)) * allProjectionsDF
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

    pnl = sl.ExPostOpt(sl.S(sl.ema(allProjectionsDF, nperiods=L)) * allProjectionsDF)[0]
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

    semaDFRollShFilteredPnl = sl.ExPostOpt(sl.S(sl.ema(DFRollShFilteredPnl, nperiods=L)) * DFRollShFilteredPnl)[0]
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

#DataBuilder()

#RunRollManifoldOnFXPairs('BasketGekko', 'DMAPS')

#plotData('projections', 'DMAPS', 'raw')
semaOnProjections('DMAPS', 3, mode='classic', shThr=0, prIn=0)

