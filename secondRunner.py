from Slider import Slider as sl
import numpy as np, pickle, json
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.manifold import LocallyLinearEmbedding

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodDataDMAPS.db')

def ContractiveOrbserverBuilder(modeIn):
    df = pd.read_sql('SELECT * FROM dataAdj', conn).set_index('Dates', drop=True)
    psi = pd.read_sql('SELECT * FROM RollDMAPSpsi', conn).set_index('index', drop=True)
    lDF = pd.read_sql('SELECT * FROM RollDMAPSlambdasDF', conn).set_index('Dates', drop=True)

    if modeIn == 0:

        for pr in range(5):
            psiTerm = ((1 - 0.5) * (-1) * lDF.iloc[:, pr]) * psi.iloc[:, pr]
            kTerm = (-1) * pd.read_sql('SELECT * FROM RollDMAPSComps' + str(pr), conn).set_index('index', drop=True)

            kDMShifted = sl.S(kTerm) * df
            kDMShifted.fillna(0).to_sql('kDMEmbeddedPortfolio' + str(pr), conn, if_exists='replace')

            cObserver = pd.concat([sl.S(psiTerm + sl.rs(kTerm * df)).fillna(0)] * len(df.columns), axis=1)
            cObserver.columns = df.columns
            # sl.cs(cObserver).plot()
            # plt.show()
            PnLcObserver = cObserver * df
            PnLcObserver.fillna(0).to_sql('PnLcObserver' + str(pr), conn, if_exists='replace')

    elif modeIn == 1:

        kDM0 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio0', conn).set_index('index', drop=True)
        kDM1 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio1', conn).set_index('index', drop=True)
        kDM2 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio2', conn).set_index('index', drop=True)
        kDM3 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio3', conn).set_index('index', drop=True)
        kDM4 = pd.read_sql('SELECT * FROM kDMEmbeddedPortfolio4', conn).set_index('index', drop=True)
        print((np.sqrt(252) * sl.sharpe(kDM0)).round(4))

        kDMall = pd.concat([sl.rs(kDM0), sl.rs(kDM1), sl.rs(kDM2), sl.rs(kDM3), sl.rs(kDM4)], axis=1, ignore_index=True)
        kDMall.fillna(0).to_sql('kDMEmbeddedPortfolioAll', conn, if_exists='replace')
        print((np.sqrt(252) * sl.sharpe(kDMall)).round(4))
        print((np.sqrt(252) * sl.sharpe(sl.rs(kDMall))).round(4))

        sl.cs(kDM0).plot(title='1st k-DM Embedded Portfolios Contributions')

        fig, axes = plt.subplots(nrows=2, ncols=1)
        sl.cs(kDMall).plot(ax=axes[0], title='k-DM Embedded Portfolios Cumulative Returns')
        sl.cs(sl.rs(kDMall)).plot(ax=axes[1], title='EWP of all k-DM Embedded Portfolios Cumulative Returns')
        plt.show()

    elif modeIn == 2:

        cObs0 = pd.read_sql('SELECT * FROM PnLcObserver0', conn).set_index('index', drop=True)
        cObs1 = pd.read_sql('SELECT * FROM PnLcObserver1', conn).set_index('index', drop=True)
        cObs2 = pd.read_sql('SELECT * FROM PnLcObserver2', conn).set_index('index', drop=True)
        cObs3 = pd.read_sql('SELECT * FROM PnLcObserver3', conn).set_index('index', drop=True)
        cObs4 = pd.read_sql('SELECT * FROM PnLcObserver4', conn).set_index('index', drop=True)
        print((np.sqrt(252) * sl.sharpe(cObs0)).round(4))

        cObsall = pd.concat([sl.rs(cObs0), sl.rs(cObs1), sl.rs(cObs2), sl.rs(cObs3), sl.rs(cObs4)], axis=1,
                            ignore_index=True)
        print((np.sqrt(252) * sl.sharpe(cObsall)).round(4))
        print((np.sqrt(252) * sl.sharpe(sl.rs(cObsall))).round(4))

        sl.cs(cObs0).plot(title='1st cObserver Embedded Portfolios Contributions')

        fig, axes = plt.subplots(nrows=2, ncols=1)
        sl.cs(cObsall).plot(ax=axes[0], title='cObserver Embedded Portfolios Cumulative Returns')
        sl.cs(sl.rs(cObsall)).plot(ax=axes[1], title='EWP of all cObserver Embedded Portfolios Cumulative Returns')
        plt.show()

    elif modeIn == 3:

        rollCo = pd.read_sql('SELECT * FROM RollDMAPScObserverDF', conn).set_index('index', drop=True)
        # sl.cs(rollCo).plot()
        # plt.show()
        for pr in range(5):
            rollCo = pd.concat([sl.S(rollCo.iloc[:, pr])] * len(df.columns), axis=1)
            rollCo.columns = df.columns
            rollCoPnl = rollCo * df
            rollCoPnl.fillna(0).to_sql('rollPnLcObserver' + str(pr), conn, if_exists='replace')

    elif modeIn == 4:

        rPnLcObs0 = pd.read_sql('SELECT * FROM rollPnLcObserver0', conn).set_index('index', drop=True)
        rPnLcObs1 = pd.read_sql('SELECT * FROM rollPnLcObserver1', conn).set_index('index', drop=True)
        rPnLcObs2 = pd.read_sql('SELECT * FROM rollPnLcObserver2', conn).set_index('index', drop=True)
        rPnLcObs3 = pd.read_sql('SELECT * FROM rollPnLcObserver3', conn).set_index('index', drop=True)
        rPnLcObs4 = pd.read_sql('SELECT * FROM rollPnLcObserver4', conn).set_index('index', drop=True)
        print((np.sqrt(252) * sl.sharpe(rPnLcObs0)).round(4))

        rPnLcObsAll = pd.concat(
            [sl.rs(rPnLcObs0), sl.rs(rPnLcObs1), sl.rs(rPnLcObs2), sl.rs(rPnLcObs3), sl.rs(rPnLcObs4)], axis=1,
            ignore_index=True)
        print((np.sqrt(252) * sl.sharpe(rPnLcObsAll)).round(4))
        print((np.sqrt(252) * sl.sharpe(sl.rs(rPnLcObsAll))).round(4))

        sl.cs(rPnLcObs0).plot(title='1st Rolling cObserver Embedded Portfolios Contributions')

        fig, axes = plt.subplots(nrows=2, ncols=1)
        sl.cs(rPnLcObsAll).plot(ax=axes[0], title='Rolling cObserver Embedded Portfolios Cumulative Returns')
        sl.cs(sl.rs(rPnLcObsAll)).plot(ax=axes[1],
                                       title='EWP of all Rolling cObserver Embedded Portfolios Cumulative Returns')
        plt.show()

    elif modeIn == 5:
        PnLeigCoeffsDF0 = sl.S(
            pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF0', conn).set_index('index', drop=True).loc[df.index]) * df
        PnLeigCoeffsDF1 = sl.S(
            pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF1', conn).set_index('index', drop=True).loc[df.index]) * df
        PnLeigCoeffsDF2 = sl.S(
            pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF2', conn).set_index('index', drop=True).loc[df.index]) * df
        PnLeigCoeffsDF3 = sl.S(
            pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF3', conn).set_index('index', drop=True).loc[df.index]) * df
        PnLeigCoeffsDF4 = sl.S(
            pd.read_sql('SELECT * FROM RollDMAPSeigCoeffsDF4', conn).set_index('index', drop=True).loc[df.index]) * df

        print((np.sqrt(252) * sl.sharpe(PnLeigCoeffsDF0)).round(4))

        PnLeigCoeffsDFAll = pd.concat(
            [sl.rs(PnLeigCoeffsDF0), sl.rs(PnLeigCoeffsDF1), sl.rs(PnLeigCoeffsDF2), sl.rs(PnLeigCoeffsDF3),
             sl.rs(PnLeigCoeffsDF4)], axis=1,
            ignore_index=True)
        print((np.sqrt(252) * sl.sharpe(PnLeigCoeffsDFAll)).round(4))
        print((np.sqrt(252) * sl.sharpe(sl.rs(PnLeigCoeffsDFAll))).round(4))

        sl.cs(PnLeigCoeffsDF0).plot(title='1st LR-Coeffs-cObserver Embedded Portfolios Contributions')

        fig, axes = plt.subplots(nrows=2, ncols=1)
        sl.cs(PnLeigCoeffsDFAll).plot(ax=axes[0],
                                      title='Rolling LR-Coeffs-cObserver Embedded Portfolios Cumulative Returns')
        sl.cs(sl.rs(PnLeigCoeffsDFAll)).plot(ax=axes[1],
                                             title='EWP of all Rolling LR-Coeffs-cObserver Embedded Portfolios Cumulative Returns')
        plt.show()

################################################ RNN TRADER ###################################################

def cObserverRnn(modeRun, whatToRun, exPostMode, magicNum):
    def Architecture(magicNum):

        magicNum = int(magicNum)

        if magicNum == 1:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 5,
                "epochsIn": 100,
                "batchSIzeIn": 5,
                "LearningMode": 'static',
                "LSTMmedSpecs": [{"units": 'xShape1', "RsF": False, "Dropout": 0.1}],
                "modelNum": magicNum,
                "TrainEndPct": 0.4,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                # optimizer='rmsprop', loss='categorical_crossentropy'
                "writeLearnStructure": 1
            }

        elif magicNum == 2:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 25,
                "epochsIn": 100,
                "batchSIzeIn": 1,
                "LearningMode": 'static',
                "LSTMmedSpecs": [{"units": 'xShape1', "RsF": True, "Dropout": 0.1},
                                 {"units": 'xShape1', "RsF": False, "Dropout": 0.05}],
                "modelNum": magicNum,
                "TrainEndPct": 0.4,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                # optimizer='rmsprop', loss='categorical_crossentropy'
                "writeLearnStructure": 1
            }

        elif magicNum == 3:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 5,
                "epochsIn": 100,
                "batchSIzeIn": 50,
                "LearningMode": 'static',
                "LSTMmedSpecs": [{"units": 'xShape1', "RsF": True, "Dropout": 0.1},
                                 {"units": 'xShape1', "RsF": False, "Dropout": 0.05}],
                "modelNum": magicNum,
                "TrainEndPct": 0.4,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                # optimizer='rmsprop', loss='categorical_crossentropy'
                "writeLearnStructure": 1
            }

        elif magicNum == 4:

            paramsSetup = {
                "HistLag": 0,
                "TrainWindow": 25,
                "epochsIn": 50,
                "batchSIzeIn": 10,
                "LearningMode": 'static',
                "LSTMmedSpecs": [{"units": 10, "RsF": True, "Dropout": 0.1},
                                 {"units": 10, "RsF": True, "Dropout": 0.1},
                                 {"units": 10, "RsF": False, "Dropout": 0.1}],
                "modelNum": magicNum,
                "TrainEndPct": 0.4,
                "CompilerSettings": ['adam', 'mean_squared_error'],
                # optimizer='rmsprop', loss='categorical_crossentropy'
                "writeLearnStructure": 1
            }

        return paramsSetup

    params = Architecture(magicNum)

    if modeRun == 'Run':
        df = pd.read_sql('SELECT * FROM ' + whatToRun + exPostMode, conn).set_index('index', drop=True)

        out = sl.AI.gRNN(df, params)
        out[0].to_sql('df_real_price_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn,
                      if_exists='replace')
        out[1].to_sql('df_predicted_price_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn,
                      if_exists='replace')
        out[2].to_sql('scoreList_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn,
                      if_exists='replace')

        pickle.dump(out[3], open("RNN_Classifier" + str(params['modelNum']) + ".p", "wb"))

    elif modeRun == 'BackTest':
        df_real_price = pd.read_sql(
            'SELECT * FROM df_real_price_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)

        df_predicted_price = pd.read_sql(
            'SELECT * FROM df_predicted_price_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum,
            conn).set_index('index', drop=True)

        df_predicted_price_errors = pd.read_sql(
            'SELECT * FROM scoreList_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn)

        print(df_predicted_price_errors)

        df_predicted_price.columns = df_real_price.columns

        # df_real_price.plot()
        # df_predicted_price.plot(title='RNN : Real vs Predicted Dynamics')
        # plt.show()

        sig = sl.S(sl.sign(df_predicted_price), nperiods=-1)
        # sig = sl.S(sl.sign(df_predicted_price))

        # pnl = sig * df_real_price.iloc[0:round(0.5*6110),:]
        # pnl = sl.ExPostOpt(sig * df_real_price.iloc[0:round(0.5*6110),:])[0]
        if exPostMode == 'exPost':
            pnl = sl.ExPostOpt(sig * df_real_price)[0]
        else:
            pnl = sig * df_real_price
        # pnl = sl.ExPostOpt(sig * allProjectionsDF)[0] # From ARIMA ...

        pnl.to_sql(
            'pnl_RNN_' + whatToRun + exPostMode + '_Projections_' + magicNum, conn, if_exists='replace')
        # pnl = sl.ExPostOpt(sig * sl.d(df_real_price))[0]
        rsPnL = sl.rs(pnl)
        print((np.sqrt(252) * sl.sharpe(pnl)).round(4))
        print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

        fig = plt.figure()
        sl.cs(pnl).plot(ax=fig.add_subplot(121), title='RNN Trading on Projections')
        sl.cs(rsPnL).plot(ax=fig.add_subplot(122), title='EWP of RNN Projections')

        # df_predicted_price_errors.plot(ax=fig.add_subplot(121), title='RNN Trading on Projections Error')
        plt.show()

    elif modeRun == 'microMergePnls':

        pnl0 = pd.read_sql(
            'SELECT * FROM pnl_RNN_' + whatToRun + str(0) + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)
        pnl1 = pd.read_sql(
            'SELECT * FROM pnl_RNN_' + whatToRun + str(1) + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)
        pnl2 = pd.read_sql(
            'SELECT * FROM pnl_RNN_' + whatToRun + str(2) + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)
        pnl3 = pd.read_sql(
            'SELECT * FROM pnl_RNN_' + whatToRun + str(3) + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)
        pnl4 = pd.read_sql(
            'SELECT * FROM pnl_RNN_' + whatToRun + str(4) + exPostMode + '_Projections_' + magicNum, conn).set_index(
            'index', drop=True)

        # pnlAll = pnl0
        # pnlAll = pnl0 + pnl1
        # pnlAll = pnl0 + pnl1+ pnl2
        # pnlAll = pnl0 + pnl1+ pnl2+ pnl3
        pnlAll = pnl0 + pnl1 + pnl2 + pnl3 + pnl4
        # pnlAll = sl.ExPostOpt(pd.concat(pnllist, axis=1, ignore_index=True))[0]

        rsPnL = sl.rs(pnlAll)
        shAll = pd.DataFrame(np.sqrt(252) * sl.sharpe(pnlAll), columns=['Sharpe']).round(4)
        shAll['Sign'] = 'NEU'
        shAll['Sign'][shAll['Sharpe'] > 0] = 'POS'
        shAll['Sign'][shAll['Sharpe'] < 0] = 'NEG'
        shAll[shAll['Sign'] == 'POS'].sort_values(by=['Sharpe']).to_sql('shAllPOS', conn, if_exists='replace')
        shAll[shAll['Sign'] == 'NEG'].sort_values(by=['Sharpe']).to_sql('shAllNEG', conn, if_exists='replace')
        print((np.sqrt(252) * sl.sharpe(rsPnL)).round(4))

        fig = plt.figure()
        # dfIRD.loc[psi.index, 'USDNOK'].plot(ax=fig.add_subplot(221), title='dfIRD_USDNOK')
        # lDF.iloc[:,0].plot(ax=fig.add_subplot(222), title='FirstlDf')
        # sl.cs(compsDF['USDNOK']).plot(ax=fig.add_subplot(223), title='compsDF[USDNOK]')
        # sl.cs(pnlAll['USDNOK']).plot(ax=fig.add_subplot(224), title='RNN Trading on USDNOK')

        sl.cs(pnlAll).plot(ax=fig.add_subplot(121), title='RNN Trading on all Projections combined')
        sl.cs(rsPnL).plot(ax=fig.add_subplot(122), title='EWP of RNN PnLs')
        # df_predicted_price_errors.plot(ax=fig.add_subplot(121), title='RNN Trading on Projections Error')
        plt.show()


# for i in range(6):
#    print("Running Contractive Observer with input : ", str(i))
#    ContractiveOrbserverBuilder(i)

# for pr in range(5):
for mNum in ['1', '2', '3', '4']:
    # print('Training RNN model '+ mNum +' for kDM embedded portfolio'+str(pr))
    # cObserverRnn('Run', 'kDMEmbeddedPortfolio'+str(pr), '', mNum)
    # cObserverRnn('BackTest', 'kDMEmbeddedPortfolio'+str(pr), '', mNum)
    # print('microMergePnlsof RNN model ' + mNum)
    # cObserverRnn('microMergePnls', 'kDMEmbeddedPortfolio', '', mNum)
    print('kDMEmbeddedPortfolioAll RNN model ' + mNum)
    cObserverRnn('Run', 'kDMEmbeddedPortfolioAll', '', mNum)
# cObserverRnn('BackTest', 'kDMEmbeddedPortfolioAll', '', mNum)

# for pr in range(5):
#    print('Training RNN for PnLcObserver ' + str(pr))
#    cObserverRnn('Run', 'PnLcObserver'+str(pr), '')
#    cObserverRnn('BackTest', 'PnLcObserver'+str(pr), '')
# cObserverRnn('microMergePnls', 'PnLcObserver', '')

# for pr in range(5):
#    print('Training RNN for rollPnLcObserver ' + str(pr))
#    cObserverRnn('Run', 'rollPnLcObserver'+str(pr), '')
#    cObserverRnn('BackTest', 'rollPnLcObserver'+str(pr), '')
#cObserverRnn('microMergePnls', 'rollPnLcObserver', '')
