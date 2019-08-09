import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
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

    def bma(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        MA = pd.DataFrame(df.rolling(nperiods, min_periods=nperiods).mean()).fillna(0)
        return MA

    def bema(df, **kwargs):
        if 'nperiods' in kwargs:
            nperiods = kwargs['nperiods']
        else:
            nperiods = 3
        EMA = pd.DataFrame(df.ewm(span=nperiods, min_periods=nperiods).mean()).fillna(0)
        return EMA

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

        def gRollingPca(df0, st, pcaN, eigsPC, **kwargs):
            if 'RollMode' in kwargs:
                RollMode = kwargs['RollMode']
            else:
                RollMode = 'RollWindow'

            # st = 50; pcaN = 5; eigsPC = [0];
            pcaComps = []
            for i in range(st, len(df0)):
                print("Step:", i, " of ", len(df0))
                if RollMode == 'RollWindow':
                    df = df0.iloc[i - st:i, :]
                else:
                    df = df0.iloc[0:i, :]

                features = df.columns.values
                x = df.loc[:, features].values
                x = StandardScaler().fit_transform(x)

                pca = PCA(n_components=pcaN)
                principalComponents = pca.fit_transform(x)
                # if i == st:
                #    print(pca.explained_variance_ratio_)
                wContribs = pd.DataFrame(pca.components_[eigsPC])
                pcaComps.append(wContribs.sum().tolist())

            df1 = df0.iloc[st:, :]
            principalCompsDf = pd.DataFrame(pcaComps, columns=df0.columns, index=df1.index).abs()
            exPostProjections = df1 * Slider.S(principalCompsDf)

            return [df1, principalCompsDf, exPostProjections]

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
