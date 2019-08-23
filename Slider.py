import pandas as pd, numpy as np, matplotlib.pyplot as plt, multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import History
import warnings, os, tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

class Slider:

    # Operators

    def d(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        return df.diff(nperiods)

    def E(df):
        return df.mean(axis=1)

    def cs(df):
        return df.cumsum()

    def rs(df):
        return df.sum(axis=1)

    def dlog(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        return df.pct_change(nperiods, fill_method='ffill')

    def sign(df):
        df[df > 0] = 1
        df[df < 0] = -1
        df[df == 0] = 0
        return df

    def S(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 1
        return pd.DataFrame.shift(df, nperiods)

    def fd(df, **kwargs):
        if 'mode' in kwargs:
            mode = kwargs['mode']
        else:
            mode = 0
        if mode == 1:
            df = df.fillna(df.mean())
        elif mode == 0:
            df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

    def roller(df, func, n):
        ROLL = df.rolling(window=n, center=False).apply(lambda x: func(x))
        return ROLL

    def rollerVol(df, rvn):
        return Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def rollVol(df, rvn):
        return np.sqrt(len(df) / 14) * Slider.roller(df, np.std, rvn).replace(np.nan, 1).replace(0, 1)

    def rollSh(df):
        return np.sqrt(len(df) / 14) * Slider.roller(df, Slider.sharpe, 100)

    def sharpe(df):
        return df.mean() / df.std()

    def CorrMatrix(df):
        return df.corr()

    def sma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        MA = pd.DataFrame(df.rolling(nperiods, min_periods=nperiods).mean()).fillna(0)
        return MA

    def sema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = pd.DataFrame(df.ewm(span=nperiods, min_periods=nperiods).mean()).fillna(0)
        return EMA

    'ARIMA Operators'
    'Optimize order based on AIC or BIC fit'

    def get_optModel(data, opt):
        best_score, best_cfg = float("inf"), None
        if opt == 'AIC':
            for p in range(0, 15, 5):
                for d in [0, 1]:
                    for q in range(0, 2):
                        order = (p, d, q)
                        try:
                            model = ARIMA(data, order=(p, d, q))
                            model_fit = model.fit(disp=0)
                            aicVal = model_fit.aic
                            if aicVal < best_score:
                                best_score, best_cfg = aicVal, order
                            # print('ARIMA%s AIC=%.3f' % (order, aicVal))
                        except:
                            continue
        elif opt == 'BIC':
            for p in range(0, 15, 5):
                for d in [0, 1]:
                    for q in range(0, 2):
                        order = (p, d, q)
                        if p != q:
                            try:
                                model = ARIMA(data, order=(p, d, q))
                                model_fit = model.fit(disp=0)
                                aicVal = model_fit.bic
                                if aicVal < best_score:
                                    best_score, best_cfg = aicVal, order
                                # print('ARIMA%s AIC=%.3f' % (order, aicVal))
                            except:
                                continue
        return best_cfg

    'Arima Dataframe process'
    'Input: list of Dataframe, start: start/window, mode: rolling/expanding, opt: AIC, BIC, (p,d,q)'

    def ARIMA_process(datalist):
        Asset = datalist[0];
        start = datalist[1];
        mode = datalist[2];
        opt = datalist[3]
        AssetName = Asset.columns[0]

        X = Asset.to_numpy()
        predictions = []
        stderrList = []
        err = []
        confList = []
        for t in range(start, len(X)):

            if mode == 'roll':
                history = X[t - start:t]
            elif mode == 'exp':
                history = X[0:t]
            try:
                if opt == 'AIC' or opt == 'BIC':
                    Orders = Slider.get_optModel(history, opt)
                else:
                    Orders = opt
                model = ARIMA(history, order=Orders)
                model_fit = model.fit(disp=0)
                # print(model_fit.resid)
                forecast, stderr, conf = model_fit.forecast()
                yhat = forecast
                c1 = conf[0][0]
                c2 = conf[0][1]
            except Exception as e:
                print(e)
                yhat = np.zeros(1) + X[t - 1]
                stderr = np.zeros(1)
                c1 = np.nan;
                c2 = np.nan

            predictions.append(yhat)
            stderrList.append(stderr)
            obs = X[t]
            err.append((yhat - obs)[0])
            confList.append((c1, c2))

        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderrList, index=Asset[start:].index, columns=[AssetName + '_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName + '_err'])
        confDF = pd.DataFrame(confList, index=Asset[start:].index, columns=[AssetName + '_conf1', AssetName + '_conf2'])

        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_static(datalist):
        Asset = datalist[0];
        start = datalist[1];
        opt = datalist[3]
        AssetName = Asset.columns[0]
        X = Asset.to_numpy()
        history = X[start:-1]
        try:
            if opt == 'AIC' or opt == 'BIC':
                Orders = Slider.get_optModel(history, opt)
            else:
                Orders = opt
            model = ARIMA(history, order=Orders)
            model_fit = model.fit(disp=0)
            forecast, stderr, conf = model_fit.forecast()
            yhat = forecast
            c1 = conf[0][0]
            c2 = conf[0][1]
        except Exception as e:
            print(e)
            yhat = np.zeros(1) + history[-1]
            stderr = np.zeros(1)
            c1 = np.nan;
            c2 = np.nan

        obs = history[-1]
        err = (yhat - obs)
        predictions = yhat[0]
        PredictionsDF = pd.DataFrame(columns=[AssetName], index=Asset.index)
        stderrDF = pd.DataFrame(columns=[AssetName + '_stderr'], index=Asset.index)
        errDF = pd.DataFrame(columns=[AssetName + '_err'], index=Asset.index)
        confDF = pd.DataFrame(columns=[AssetName + '_conf1', AssetName + '_conf2'], index=Asset.index)

        PredictionsDF.loc[Asset.index[-1], AssetName] = predictions
        stderrDF.loc[Asset.index[-1], AssetName + '_stderr'] = stderr[0]
        errDF.loc[Asset.index[-1], AssetName + '_err'] = err[0]
        confDF.loc[Asset.index[-1], AssetName + '_conf1'] = c1;
        confDF.loc[Asset.index[-1], AssetName + '_conf2'] = c2

        return [PredictionsDF.iloc[-1, :], stderrDF.iloc[-1, :], errDF.iloc[-1, :], confDF.iloc[-1, :]]

    def ARIMA_predictions(df, mode, opt, **kwargs):
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0
        if 'multi' in kwargs:
            multi = kwargs['multi']
        else:
            multi = 0
        if 'indextype' in kwargs:
            indextype = kwargs['indextype']
        else:
            indextype = 0

        if indextype == 1:
            frequency = pd.infer_freq(df.index)
            df.index.freq = frequency
            print(frequency)
            df.loc[df.index.max() + pd.to_timedelta(frequency)] = None
        else:
            df.loc[df.index.max() + 1] = None
            print(df)

        Assets = df.columns.tolist()
        dflist = [[pd.DataFrame(df[Asset]), start, mode, opt] for Asset in Assets]
        if multi == 0:
            pool = multiprocessing.Pool(processes=len(df.columns.tolist()))
            if mode != 'static':
                resultsDF = pool.map(Slider.ARIMA_process, dflist)
                predictions = [x[0] for x in resultsDF]
                stderr = [x[1] for x in resultsDF]
                err = [x[2] for x in resultsDF]
                conf = [x[3] for x in resultsDF]
                PredictionsDF = pd.concat(predictions, axis=1, sort=True)
                stderrDF = pd.concat(stderr, axis=1, sort=True)
                errDF = pd.concat(err, axis=1, sort=True)
                confDF = pd.concat(conf, axis=1, sort=True)
            else:
                resultsDF = pool.map(Slider.ARIMA_static, dflist)
                PredictionsDF = [x[0] for x in resultsDF]
                stderrDF = [x[1] for x in resultsDF]
                errDF = [x[2] for x in resultsDF]
                confDF = [x[3] for x in resultsDF]

        elif multi == 1:
            resultsDF = []
            for i in range(len(Assets)):
                resultsDF.append(Slider.ARIMA_multiprocess(dflist[i]))

            predictions = [x[0] for x in resultsDF]
            stderr = [x[1] for x in resultsDF]
            err = [x[2] for x in resultsDF]
            conf = [x[3] for x in resultsDF]
            PredictionsDF = pd.concat(predictions, axis=1, sort=True)
            stderrDF = pd.concat(stderr, axis=1, sort=True)
            errDF = pd.concat(err, axis=1, sort=True)
            confDF = pd.concat(conf, axis=1, sort=True)

        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_predict(historyOpt):
        history = historyOpt[0];
        opt = historyOpt[1]
        if opt == 'AIC' or opt == 'BIC':
            Orders = Slider.get_optModel(history, opt)
        else:
            Orders = opt
        try:
            print(Orders)
            model = ARIMA(history, order=Orders)
            model_fit = model.fit(disp=0)
            forecast, stderr, conf = model_fit.forecast()
            yhat = forecast
            c1 = conf[0][0]
            c2 = conf[0][1]
        except Exception as e:
            print(e)
            yhat = np.zeros(1) + history[-1]
            stderr = np.zeros(1)
            c1 = np.nan;
            c2 = np.nan
        obs = history[-1]
        err = (yhat - obs)

        return [yhat, stderr, err, (c1, c2)]

    def ARIMA_multiprocess(df):
        print('Running Multiprocess Arima')

        Asset = df[0];
        start = df[1];
        mode = df[2];
        opt = df[3]
        AssetName = Asset.columns[0]

        X = Asset.to_numpy()

        if mode == 'roll':
            history = [[X[t - start:t], opt] for t in range(start, len(X))]
        elif mode == 'exp':
            history = [[X[0:t], opt] for t in range(start, len(X))]

        p = multiprocessing.Pool(processes=12)
        results = p.map(Slider.ARIMA_predict, history)
        p.close()
        p.join()

        predictions = [x[0] for x in results]
        stderr = [x[1] for x in results]
        err = [x[2] for x in results]
        conf = [x[3] for x in results]
        PredictionsDF = pd.DataFrame(predictions, index=Asset[start:].index, columns=[AssetName])
        stderrDF = pd.DataFrame(stderr, index=Asset[start:].index, columns=[AssetName + '_stderr'])
        errDF = pd.DataFrame(err, index=Asset[start:].index, columns=[AssetName + '_err'])
        confDF = pd.DataFrame(conf, index=Asset[start:].index, columns=[AssetName + '_conf1', AssetName + '_conf2'])
        return [PredictionsDF, stderrDF, errDF, confDF]

    def ARIMA_multipredictions(df, mode, opt, **kwargs):
        if 'start' in kwargs:
            start = kwargs['start']
        else:
            start = 0
        if 'multi' in kwargs:
            multi = kwargs['multi']
        else:
            multi = 0
        if 'indextype' in kwargs:
            indextype = kwargs['indextype']
        else:
            indextype = 0

        if indextype == 1:
            frequency = pd.infer_freq(df.index)
            df.index.freq = frequency
            print(frequency)
            df.loc[df.index.max() + pd.to_timedelta(frequency)] = None

        Assets = df.columns.tolist()
        dflist = [[pd.DataFrame(df[Asset]), start, mode, opt] for Asset in Assets]
        resultsDF = []
        for i in range(len(Assets)):
            resultsDF.append(Slider.ARIMA_multiprocess(dflist[i]))

        predictions = [x[0] for x in resultsDF]
        stderr = [x[1] for x in resultsDF]
        err = [x[2] for x in resultsDF]
        conf = [x[3] for x in resultsDF]
        PredictionsDF = pd.concat(predictions, axis=1, sort=True)
        stderrDF = pd.concat(stderr, axis=1, sort=True)
        errDF = pd.concat(err, axis=1, sort=True)
        confDF = pd.concat(conf, axis=1, sort=True)

        return [PredictionsDF, stderrDF, errDF, confDF]

    # Machine Learning Functions
    class AI:

        def Pca(df, **kwargs):

            Dates = df.index

            if 'nD' not in kwargs:
                nD = 2
            else:
                nD = kwargs['nD']

            features = df.columns.values
            # Separating out the features
            x = df.loc[:, features].values
            # Separating out the target
            # y = df.loc[:, ['target']].values
            # Standardizing the features
            x = StandardScaler().fit_transform(x)

            pca = PCA(n_components=nD)
            principalComponents = pca.fit_transform(x)
            principalDf = pd.DataFrame(data=principalComponents)

            principalDf['Date'] = Dates

            principalDf = principalDf.set_index('Date', drop=True)

            return principalDf

        def gRollingManifold(manifold, df0, st, NumProjections, eigsPC, **kwargs):
            if 'RollMode' in kwargs:
                RollMode = kwargs['RollMode']
            else:
                RollMode = 'RollWindow'

            if 'Scaler' in kwargs:
                Scaler = kwargs['Scaler']
            else:
                Scaler = 'Standard'

            if 'ProjectionMode' in kwargs:
                ProjectionMode = kwargs['ProjectionMode']
            else:
                ProjectionMode = 'Transpose'

            if 'LLE_n_neighbors' in kwargs:
                n_neighbors = kwargs['LLE_n_neighbors']
            else:
                n_neighbors = 2

            Comps = [[] for j in range(len(eigsPC))]
            # st = 50; pcaN = 5; eigsPC = [0];
            for i in range(st, len(df0)+1):
                try:

                    print("Step:", i, " of ", len(df0))
                    if RollMode == 'RollWindow':
                        df = df0.iloc[i - st:i, :]
                    else:
                        df = df0.iloc[0:i, :]

                    if ProjectionMode == 'Transpose':
                        df = df.T

                    features = df.columns.values
                    x = df.loc[:, features].values

                    if Scaler == 'Standard':
                        x = StandardScaler().fit_transform(x)

                    if manifold == 'PCA':
                        pca = PCA(n_components=NumProjections)
                        X_pca = pca.fit_transform(x)
                        # if i == st:
                        #    print(pca.explained_variance_ratio_)
                        for eig in eigsPC:
                            #print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                            Comps[eig].append(list(pca.components_[eig]))

                    elif manifold == 'LLE':
                        lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections, method='standard')
                        X_lle = lle.fit_transform(x)
                        for eig in eigsPC:
                            #print(eig, ',', len(pca.components_[eig]), ',', len(pca.components_)) # 0 , 100 , 5
                            Comps[eig].append(list(X_lle[:, eig]))

                except Exception as e:
                    print(e)
                    for c in len(eigsPC):
                        Comps[c].append(list(np.zeros(len(df0.columns), 1)))

            print(len(Comps[eig]))
            principalCompsDf = [[] for j in range(len(Comps))]
            exPostProjections = [[] for j in range(len(Comps))]
            for k in range(len(Comps)):
                #principalCompsDf[k] = pd.DataFrame(pcaComps[k], columns=df0.columns, index=df1.index)

                principalCompsDf[k] = pd.concat([pd.DataFrame(np.zeros((st-1, len(df0.columns))), columns=df0.columns),
                                                 pd.DataFrame(Comps[k], columns=df0.columns)], axis=0, ignore_index=True)
                principalCompsDf[k].index = df0.index
                principalCompsDf[k] = principalCompsDf[k].fillna(0).replace(0, np.nan).ffill()

                exPostProjections[k] = df0 * Slider.S(principalCompsDf[k]) #exPostProjections[k] = df0.mul(Slider.S(principalCompsDf[k]), axis=0)

            return [df0, principalCompsDf, exPostProjections]

        def gRNN(dataset_all, params):
            history = History()

            HistLag = 0;
            TrainWindow = 50;
            TrainEnd = 0.3 * len(dataset_all)
            InputArchitecture = pd.DataFrame([['LSTM', 0.2], ['LSTM', 0.2], ['LSTM', 0.2], ['Dense', 0.2]],
                                             columns=['LayerType', 'LayerDropout'])
            OutputArchitecture = pd.DataFrame([['Dense', 0.2]], columns=['LayerType', 'LayerDropout'])
            CompilerParams = pd.DataFrame([['adam', 'mean_squared_error']], columns=['optimizer', 'loss'])
            epochsIn = 50;
            batchSIzeIn = 49;
            medBatchTrain = 49;
            HistoryPlot = 0;
            PredictionsPlot = 0

            training_set = dataset_all.iloc[:TrainEnd].values

            # Feature Scaling
            sc = MinMaxScaler(feature_range=(0, 1))
            training_set_scaled = sc.fit_transform(training_set)
            # training_set_scaled = training_set

            # Creating a data structure with N timesteps and 1 output
            X_train = [];
            y_train = []
            for i in range(TrainWindow, TrainEnd):
                X_train.append(training_set_scaled[i - TrainWindow:i - HistLag])
                y_train.append(training_set_scaled[i])
            X_train, y_train = np.array(X_train), np.array(y_train)

            # Reshaping
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(dataset_all.columns)))

            # Part 2 - Building the RNN
            history = History()

            # Initialising the RNN
            regressor = Sequential()
            # Adding the first LSTM layer and some Dropout regularisation
            for idx, row in InputArchitecture.iterrows():
                if idx == 0:
                    if row['LayerType'] == 'LSTM':
                        LayerIn = LSTM(units=X_train.shape[1], return_sequences=True,
                                       input_shape=(X_train.shape[1], len(dataset_all.columns)))
                    elif row['LayerType'] == 'Dense':
                        LayerIn = Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu',
                                        input_dim=X_train.shape[1])

                elif idx < len(InputArchitecture):
                    if row['LayerType'] == 'LSTM':
                        LayerIn = LSTM(units=X_train.shape[1], return_sequences=True)
                    elif row['LayerType'] == 'Dense':
                        LayerIn = Dense(units=X_train.shape[1])

                elif idx == len(InputArchitecture):
                    if row['LayerType'] == 'LSTM':
                        LayerIn = LSTM(units=X_train.shape[1])
                    elif row['LayerType'] == 'Dense':
                        LayerIn = Dense(units=X_train.shape[1])

                regressor.add(LayerIn)
                regressor.add(Dropout(float(row['LayerDropout'])))

            # Adding the output layer
            for idx, row in OutputArchitecture.iterrows():
                if row['LayerType'] == 'LSTM':
                    LayerOut = LSTM(units=y_train.shape[1])
                elif row['LayerType'] == 'Dense':
                    LayerOut = Dense(units=y_train.shape[1])
                regressor.add(LayerOut)

            # Compiling the RNN
            regressor.compile(optimizer=CompilerParams['optimizer'][0], loss=CompilerParams['loss'][0])
            # regressor.compile(optimizer='rmsprop', loss='mean_squared_error')
            # regressor.compile(optimizer='adam', loss='categorical_crossentropy')

            # Fitting the RNN to the Training set
            regressor.fit(X_train, y_train, epochs=epochsIn, batch_size=batchSIzeIn, callbacks=[history])

            # Part 3 - Making the predictions and visualising the results
            # Getting the test set
            dataset_test = dataset_all.iloc[TrainEnd:].values
            inputs = sc.transform(dataset_test.reshape(-1, len(dataset_all.columns)))
            # inputs = dataset_test.reshape(-1, len(dataset_all.columns))

            # Getting the predicted prices
            dataset_total = pd.concat([pd.DataFrame(training_set_scaled), pd.DataFrame(inputs)], axis=0).reset_index(
                drop=True).values

            X_test = [];
            y_test = []
            for i in range(TrainEnd, len(dataset_total)):  # TrainEnd+100):
                print('Calculating: ' + str(round(i / len(dataset_total) * 100, 2)) + '%')
                # Formalize the rolling input prices
                X_test.append(dataset_total[i - TrainWindow:i - HistLag])
                X_test_array = np.array(X_test)
                X_test_array = np.reshape(X_test_array,
                                          (X_test_array.shape[0], X_test_array.shape[1], len(dataset_all.columns)))

                # Formalize the rolling output prices
                y_test.append(dataset_total[i])
                y_test_array = np.array(y_test)
                # Get the out-of-sample predicted prices
                predicted_price = sc.inverse_transform(regressor.predict(X_test_array))

                if len(X_test) > medBatchTrain:
                    X_test_Batch = X_test[-medBatchTrain:]
                    X_test_array_Batch = np.array(X_test_Batch)
                    X_test_array_Batch = np.reshape(X_test_array_Batch, (
                        X_test_array_Batch.shape[0], X_test_array_Batch.shape[1], len(dataset_all.columns)))
                    y_test_Batch = y_test[-medBatchTrain:]
                    y_test_array_Batch = np.array(y_test_Batch)
                try:
                    # Retrain the RNN on batch as new info comes into play
                    regressor.train_on_batch(X_test_array_Batch, y_test_array_Batch)
                    # Final evaluation of the model
                    scores = regressor.evaluate(X_test_array_Batch, y_test_array_Batch, verbose=0)
                    print(scores)
                except Exception as e:
                    print(e)
                    print("Breaking on the Training ...")

            # df_real_price = pd.DataFrame(dataset_test)
            df_real_price = dataset_all.iloc[TrainEnd - 1:]
            # df_predicted_price = pd.DataFrame(predicted_price)
            df_predicted_price = pd.DataFrame(predicted_price, index=dataset_all.iloc[TrainEnd - 1:-1].index)

            if HistoryPlot == 1:
                print(history.history.keys())
                plt.plot(history.history['loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')
                plt.show()

            if PredictionsPlot == 1:
                plt.plot(df_real_price, color='red', label='Real Price')
                plt.plot(df_predicted_price, color='blue', label='Predicted Price')
                plt.show()

            return [df_real_price, df_predicted_price]

    class Models:

        def __init__(self, df, **kwargs):
            'The initial data input in the Models Class '
            self.df = df

            'The meta-data on which the model calculates the betting signal matrix : d, dlog, raw data'
            if 'retmode' in kwargs:
                self.retmode = kwargs['retmode']
            else:
                self.retmode = 2

            if self.retmode == 0:
                self.ToPnL = Slider.d(self.df)
            elif self.retmode == 1:
                self.ToPnL = Slider.dlog(self.df)
            else:
                self.ToPnL = self.df

            self.sig = float('nan')

        def ARIMA_signal(self, **kwargs):
            if 'start' in kwargs:
                start = kwargs['start']
            else:
                start = 0
            if 'mode' in kwargs:
                mode = kwargs['mode']
            else:
                mode = 'exp'
            if 'opt' in kwargs:
                opt = kwargs['opt']
            else:
                opt = (1,0,0)
            if 'multi' in kwargs:
                multi = kwargs['multi']
            else:
                multi = 0
            if 'indextype' in kwargs:
                indextype = kwargs['indextype']
            else:
                indextype = 0
            print(multi)
            self.sig = Slider.sign(Slider.ARIMA_predictions(self.ToPnL, start=start, mode=mode, opt=opt, multi=multi, indextype=indextype)[0] - Slider.S(self.ToPnL))
            self.ToPnL = self.ToPnL
            return self

    class BacktestPnL:

        def ModelPnL(model, **kwargs):

            if 'retmode' in kwargs:
                retmode = kwargs['retmode']
            else:
                retmode = 0

            if retmode == 1:
                #model.ToPnL = Torus.d(model.ToPnL)
                return model.sig * model.ToPnL.diff()
            else:
                return model.sig * model.ToPnL