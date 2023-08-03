from datetime import datetime
import pandas as pd, numpy as np, blpapi, sqlite3, time, matplotlib.pyplot as plt, itertools, types, multiprocessing, ta, sqlite3, xlrd
from sklearn import (manifold, datasets, decomposition, ensemble, discriminant_analysis, random_projection, neighbors)
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.ensemble import RandomForestClassifier
from pyerb import pyerb as pe
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.metrics import adjusted_mutual_info_score, mutual_info_score
from scipy.stats import norm, t
from scipy.stats import skew, kurtosis, entropy
from sklearn import linear_model
from sklearn.decomposition import PCA
from itertools import combinations
from ta.volume import *
from optparse import OptionParser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import History
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy import sparse
from scipy.linalg import svd
from scipy.stats.stats import pearsonr
from tqdm import tqdm
import warnings, os, tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection,
)

class ML:

    ############################################################################################################

    def binarize(df, targetColumns):

        for c in targetColumns:
            df.loc[df[c] > 0, c] = 1
            df.loc[df[c] < 0, c] = -1
            df.loc[df[c] == 0, c] = 0

        out = df

        return out

    def gReshape(data, Features):

        samples = data.shape[0]
        if len(data.shape) == 1:
            TimeSteps = 1
        else:
            TimeSteps = data.shape[1]

        data = data.reshape((samples, TimeSteps, Features))

        return data

    def reframeData(dataIn, reframeStep, varSelect, **kwargs):
        """
        Function to reframe a dataset into lagged instances of the input matrix :
        ####################
        dataIn : the input matrix
        reframeStep : up to which lag to create the instances
        varSelect (int) : which variable to return as 'Y' for any potential regression using the x, or if 'all' return all vars
        return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr]
        X_all_gpr : the predictors (lagged instances)
        Y_all_gpr : the targets Y (as per varselect)
        lastY_test_point_gpr : the last point to be the next input (test point) for an online ML rolling prediction framework
        """
        if "frameConstructor" in kwargs:
            frameConstructor = kwargs["frameConstructor"]
        else:
            frameConstructor = "ascending"

        if "returnMode" in kwargs:
            returnMode = kwargs["returnMode"]
        else:
            returnMode = "ML"

        baseDF = pd.DataFrame(dataIn)

        if frameConstructor == "ascending":
            looperRange = range(reframeStep + 1)
        elif frameConstructor == "descending":
            looperRange = range(reframeStep, -1, -1)

        df_List = []
        for i in looperRange:
            if i == 0:
                subDF_i0 = baseDF.copy()
                subDF_i0.columns = ["base_" + str(x) for x in subDF_i0.columns]
                df_List.append(subDF_i0)
            else:
                subDF = baseDF.shift(i)  # .fillna(0)
                subDF.columns = ["delay_" + str(i) + "_" + str(x) for x in subDF.columns]
                df_List.append(subDF)

        df = pd.concat(df_List, axis=1).dropna()

        if returnMode == "ML":

            if varSelect == "all":
                Y_DF = df.loc[:, [x for x in df.columns if "base_" in x]]
            else:
                Y_DF = df.loc[:, "base_" + str(varSelect)]
            X_DF = df.loc[:, [x for x in df.columns if "delay_" in x]]
            lastY_test_point = df.loc[df.index[-1], [x for x in df.columns if "base_" in x]]

            X_all_gpr = X_DF.values
            if isinstance(Y_DF, pd.Series) == 1:
                Y_all_gpr = Y_DF.values.reshape(-1, 1)
            else:
                Y_all_gpr = Y_DF.values

            lastY_test_point_gpr = lastY_test_point.values.reshape(1, -1)

            return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr]

        elif returnMode == "Takens":

            return df

    def gANN(X_train, X_test, y_train, params):
        epochsIn = params[0]
        batchSIzeIn = params[1]

        history = History()
        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Part 2 - Now let's make the ANN!
        # Importing the Keras libraries and packages

        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu',
                             input_dim=X_train.shape[1]))
        # Adding a Dropout
        classifier.add(Dropout(0.25))

        # Adding the second hidden layer
        classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
        # Adding a Dropout
        classifier.add(Dropout(0.25))

        # Adding the third hidden layer
        classifier.add(Dense(units=X_train.shape[1], kernel_initializer='uniform', activation='relu'))
        # Adding a Dropout
        classifier.add(Dropout(0.25))

        # Adding the output layer
        try:
            shapeIn = y_train.shape[1]
        except Exception as e:
            print(e)
            shapeIn = 1

        classifier.add(Dense(units=shapeIn, kernel_initializer='uniform', activation='sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(X_train, y_train, batch_size=batchSIzeIn, epochs=epochsIn, callbacks=[history])

        # Part 3 - Making predictions and evaluating the model

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        return [y_pred, classifier]

    def gClassification(dataset_all, params):

        ################################### Very First Data Preprocessing #########################################

        dataVals = dataset_all.values

        if isinstance(dataset_all, pd.Series):  # Single Sample
            FeatSpaceDims = 1
            outNaming = [dataset_all.name]
            print(outNaming)
        else:
            FeatSpaceDims = len(dataset_all.columns)
            outNaming = dataset_all.columns

        #################################### Feature Scaling #####################################################
        if params['Scaler'] == "Standard":
            sc_X = StandardScaler()
        elif params['Scaler'] == 'MinMax':
            sc_X = MinMaxScaler()  # feature_range=(0, 1)

        #################### Creating a data structure with N timesteps and 1 output #############################
        X = []
        y = []
        real_y = []
        for i in range(params["InputSequenceLength"], len(dataset_all)):
            X.append(dataVals[i - params["InputSequenceLength"]:i - params["HistLag"]])
            y.append(np.sign(dataVals[i]))
            real_y.append(dataVals[i])
        X, y, real_y = np.array(X), np.array(y), np.array(real_y)
        y[y == -1] = 2
        idx = dataset_all.iloc[params["InputSequenceLength"]:].index

        # print("X.shape=", X.shape, ", y.shape=", y.shape,", real_y.shape=", real_y.shape, ", FeatSpaceDims=", FeatSpaceDims,
        #      ", InputSequenceLength=", params["InputSequenceLength"],
        #      ", SubHistoryLength=", params["SubHistoryLength"],
        #      ", SubHistoryTrainingLength=", params["SubHistoryTrainingLength"], ", len(idx) = ", len(idx))

        stepper = params["SubHistoryLength"] - params["SubHistoryTrainingLength"]

        df_predicted_price_train_List = []
        df_real_price_class_train_List = []
        df_real_price_train_List = []
        df_predicted_price_test_List = []
        df_real_price_class_test_List = []
        df_real_price_test_List = []

        breakFlag = False
        megaCount = 0
        scoreList = []
        for i in range(0, X.shape[0], stepper):
            print("i = ", i, "/", X.shape[0])
            subProcessingHistory_X, subProcessingHistory_y, subProcessingHistory_real_y = X[i:i + params["SubHistoryLength"]], \
                                                                                          y[i:i + params["SubHistoryLength"]], \
                                                                                          real_y[i:i + params["SubHistoryLength"]]
            subIdx = idx[i:i + params["SubHistoryLength"]]
            if len(subProcessingHistory_X) < params["SubHistoryLength"]:
                subProcessingHistory_X, subProcessingHistory_y, subProcessingHistory_real_y = X[i:], y[i:], real_y[i:]
                subIdx = idx[i:]
                breakFlag = True
            X_train, y_train, real_y_train = subProcessingHistory_X[:params["SubHistoryTrainingLength"]], \
                                             subProcessingHistory_y[:params["SubHistoryTrainingLength"]], \
                                             subProcessingHistory_real_y[:params["SubHistoryTrainingLength"]],
            subIdx_train = subIdx[:params["SubHistoryTrainingLength"]]
            X_test, y_test, real_y_test = subProcessingHistory_X[params["SubHistoryTrainingLength"]:], \
                                          subProcessingHistory_y[params["SubHistoryTrainingLength"]:], \
                                          subProcessingHistory_real_y[params["SubHistoryTrainingLength"]:]
            subIdx_test = subIdx[params["SubHistoryTrainingLength"]:]

            # Enable Scaling
            if params['Scaler'] is not None:
                X_train = sc_X.fit_transform(X_train)
                try:
                    X_test = sc_X.transform(X_test)
                except Exception as e:
                    print(e)

            # print("Data subHistories Set : i = ", i, ", len(subProcessingHistory_X) = ", len(subProcessingHistory_X),
            #      ", len(subProcessingHistory_y) = ", len(subProcessingHistory_y),
            #      ", X_train.shape = ", X_train.shape, ", y_train = ", y_train.shape, ", X_test.shape = ", X_test.shape,
            #      ", y_test.shape = ", y_test.shape)

            ####################################### Reshaping ########################################################
            "Samples : One sequence is one sample. A batch is comprised of one or more samples."
            "Time Steps : One time step is one point of observation in the sample."
            "Features : One feature is one observation at a time step."

            ################################### Build the Systems #####################################
            # print("megaCount = ", megaCount)

            if params["model"] == "RNN":
                X_train, X_test = ML.gReshape(X_train, FeatSpaceDims), \
                                  ML.Reshape(X_test, FeatSpaceDims)
                if megaCount == 0:
                    # print("After Reshaping : X_train.shape = ", X_train.shape,
                    #      ", y_train = ", y_train.shape, ", X_test.shape = ", X_test.shape,
                    #      ", y_test.shape = ", y_test.shape)

                    ########################################## RNN #############################################
                    print("Recurrent Neural Networks Classification...", outNaming)
                    model = Sequential()
                    # Adding the first LSTM layer and some Dropout regularisation
                    for layer in range(len(params["medSpecs"])):
                        if params["medSpecs"][layer]["units"] == 'xShape1':
                            unitsIn = X_train.shape[1]
                            if params["medSpecs"][layer]["LayerType"] == "LSTM":
                                model.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                               unit_forget_bias=True, bias_initializer='ones',
                                               input_shape=(X_train.shape[1], FeatSpaceDims)))
                            elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                                model.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"],
                                                    unit_forget_bias=True, bias_initializer='ones',
                                                    input_shape=(X_train.shape[1], FeatSpaceDims)))
                            model.add(Dropout(params["medSpecs"][layer]["Dropout"]))
                        else:
                            unitsIn = params["medSpecs"][layer]["units"]
                            if params["medSpecs"][layer]["LayerType"] == "LSTM":
                                model.add(LSTM(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                            elif params["medSpecs"][layer]["LayerType"] == "SimpleRNN":
                                model.add(SimpleRNN(units=unitsIn, return_sequences=params["medSpecs"][layer]["RsF"]))
                            model.add(Dropout(params["medSpecs"][layer]["Dropout"]))
                    # Adding the output layer
                    if len(y_train.shape) == 1:
                        model.add(Dense(units=1))
                    else:
                        model.add(Dense(units=y_train.shape[1]))
                    ####
                    my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=params["EarlyStopping_patience_Epochs"])]
                    model.compile(optimizer=params["CompilerSettings"][0], loss=params["CompilerSettings"][1])
                # Fitting the RNN Model to the Training set
                try:
                    model.fit(X_train, y_train, epochs=params["epochsIn"], batch_size=params["batchSIzeIn"],
                              verbose=0, callbacks=my_callbacks)
                except Exception as e:
                    print(e)
                ############################################
                try:
                    trainScore = model.evaluate(X_train, y_train, verbose=0)
                except Exception as e:
                    print(e)
                    trainScore = np.nan
            elif params["model"] == "GPC":
                ########################################## GPC #############################################
                # Define model
                if megaCount == 0:
                    print("Gaussian Process Classification...", outNaming, ", Kernel = ", params['Kernel'])
                    if params['Kernel'] == '0':
                        mainKernel = 1 ** 2 * Matern(length_scale=1, nu=0.5) + WhiteKernel()
                    elif params['Kernel'] == '1':
                        mainKernel = 1 ** 2 * Matern(length_scale=1, nu=0.5) + 1 ** 2 * DotProduct(sigma_0=1) + \
                                     1 ** 2 * RationalQuadratic(alpha=1, length_scale=1) + 1 ** 2 * ConstantKernel()
                    ##################### Running with Greedy Search Best Model ##################
                    model = GaussianProcessClassifier(kernel=mainKernel, random_state=0)
                # Fitting the GPC Model to the Training set
                try:
                    model.fit(X_train, y_train)
                except Exception as e:
                    print(e)
                ###########################################
                try:
                    trainScore = model.score(X_train, y_train)
                except:
                    trainScore = np.nan
            # print(trainScore)
            scoreList.append(trainScore)
            ############################### TRAIN PREDICT #################################
            try:
                predicted_price_train = model.predict(X_train)
            except:
                predicted_price_train = np.zeros(X_train.shape[0])
            if params["model"] == "GPC":
                try:
                    predicted_price_proba_train = model.predict_proba(X_train)
                except:
                    predicted_price_proba_train = np.zeros((X_train.shape[0], len(model.classes_)))

            # print("predicted_price_train = ", predicted_price_train, ", predicted_price_proba_train = ", predicted_price_proba_train)
            # print(predicted_price_train.shape, ", ", (np.zeros(X_train.shape[0])).shape)
            # print(predicted_price_proba_train.shape, ", ", (np.zeros((X_train.shape[0], len(model.classes_)))).shape)

            df_predicted_price_train = pd.DataFrame(predicted_price_train, index=subIdx_train,
                                                    columns=['Predicted_Train_' + str(x) for x in outNaming])
            if params["model"] == "GPC":
                df_predicted_price_proba_train = pd.DataFrame(predicted_price_proba_train, index=subIdx_train,
                                                              columns=['Predicted_Proba_Train_' + str(x) for x in
                                                                       model.classes_])
                df_predicted_price_train = pd.concat([df_predicted_price_train, df_predicted_price_proba_train], axis=1)
            df_real_price_class_train = pd.DataFrame(y_train, index=subIdx_train,
                                                     columns=['Real_Train_Class_' + str(x) for x in outNaming])
            df_real_price_train = pd.DataFrame(real_y_train, index=subIdx_train,
                                               columns=['Real_Train_' + str(x) for x in outNaming])
            ############################### TEST PREDICT ##################################
            if params["LearningMode"] == 'static':
                try:
                    predicted_price_test = model.predict(X_test)
                except:
                    predicted_price_test = np.zeros(X_test.shape[0])
                if params["model"] == "GPC":
                    try:
                        predicted_price_proba_test = model.predict_proba(X_test)
                    except:
                        predicted_price_proba_test = np.zeros((X_test.shape[0], len(model.classes_)))

                # print("predicted_price_test = ", predicted_price_test, ", y_test = ", y_test, ", predicted_price_proba_test = ", predicted_price_proba_test)
                df_predicted_price_test = pd.DataFrame(predicted_price_test, index=subIdx_test,
                                                       columns=['Predicted_Test_' + x for x in outNaming])
                if params["model"] == "GPC":
                    df_predicted_price_proba_test = pd.DataFrame(predicted_price_proba_test, index=subIdx_test,
                                                                 columns=['Predicted_Proba_Test_' + str(x) for x in
                                                                          model.classes_])
                    df_predicted_price_test = pd.concat([df_predicted_price_test, df_predicted_price_proba_test],
                                                        axis=1)
                df_real_price_class_test = pd.DataFrame(y_test, index=subIdx_test,
                                                        columns=['Real_Test_Class_' + x for x in outNaming])
                df_real_price_test = pd.DataFrame(real_y_test, index=subIdx_test,
                                                  columns=['Real_Test_' + x for x in outNaming])

            df_predicted_price_train_List.append(df_predicted_price_train)
            df_real_price_class_train_List.append(df_real_price_class_train)
            df_real_price_train_List.append(df_real_price_train)
            df_predicted_price_test_List.append(df_predicted_price_test)
            df_real_price_class_test_List.append(df_real_price_class_test)
            df_real_price_test_List.append(df_real_price_test)

            megaCount += 1

            if breakFlag:
                break

        df_predicted_price_train_DF = pd.concat(df_predicted_price_train_List, axis=0)
        df_real_price_class_train_DF = pd.concat(df_real_price_class_train_List, axis=0)
        df_real_price_train_DF = pd.concat(df_real_price_train_List, axis=0)
        df_predicted_price_test_DF = pd.concat(df_predicted_price_test_List, axis=0)
        df_real_price_class_test_DF = pd.concat(df_real_price_class_test_List, axis=0)
        df_real_price_test_DF = pd.concat(df_real_price_test_List, axis=0)

        dfScore = pd.DataFrame(scoreList)

        return [df_predicted_price_train_DF, df_real_price_class_train_DF, df_real_price_train_DF,
                df_predicted_price_test_DF, df_real_price_class_test_DF, df_real_price_test_DF, dfScore, model]

    ############################################################################################################

    def Roll_SML_Predict(df, runMode, **kwargs):
        print("df.shape = ", df.shape)
        def rollPreds(roll_rets, assetSel):
            try:
                roll_reframedData = ML.reframeData(roll_rets.values, 1, assetSel)
                X_train = roll_reframedData[0]
                Y_train = np.sign(roll_reframedData[1])
                Y_train[Y_train > 0] = 1
                Y_train[Y_train < 0] = -1
                model.fit(X_train, Y_train)
                predicted_price_train = model.predict(roll_reframedData[2])
                predicted_price_train_prob = model.predict_proba(roll_reframedData[2])
                probsList = list(predicted_price_train_prob[0])
                if len(model.classes_) == 2:
                    probsList = [probsList[0], 0, probsList[1]]
                subDataList = [roll_rets.index[-1], predicted_price_train[0]]
                for x in probsList:
                    subDataList.append(x)
            except Exception as e:
                print(e)
                subDataList = [roll_rets.index[-1], None]

            return subDataList

        def FeaturesExtract(data, targetAsset, mode, topNum):
            feature_names = data.columns

            X_train = data.values
            # X_train = (X_train-np.mean(X_train))/np.std(X_train)
            X_test = data.values

            y_train = data.loc[:, targetAsset].values
            # y_train = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test = data.loc[:, targetAsset].values

            params = {
                "n_estimators": 500,
                "max_depth": 4,
                "min_samples_split": 5,
                'max_features': int(1),  # MLdP
                "warm_start": True,
                "oob_score": True,
                "random_state": 42,
            }
            reg = RandomForestRegressor(**params)
            reg.fit(X_train, y_train)

            if mode == "MDI":
                y_pred = reg.predict(X_test)
                #print(mean_squared_error(y_test, y_pred, squared=False))
                feature_importance = reg.feature_importances_
                sorted_idx = np.argsort(feature_importance)
                #pos = np.arange(sorted_idx.shape[0])
                FeaturesOut = pd.Series(np.array(feature_names)[sorted_idx]).tolist()

            elif mode == "MDA":
                result = permutation_importance(
                    reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
                )
                tree_importances = pd.Series(result.importances_mean, index=feature_names)
                sorted_idx = np.argsort(tree_importances)
                FeaturesOut = pd.Series(np.array(feature_names)[sorted_idx]).tolist()

            return FeaturesOut[:topNum]

        if "workConn" in kwargs:
            workConn = sqlite3.connect(kwargs["workConn"],detect_types=sqlite3.PARSE_DECLTYPES)
        else:
            workConn = sqlite3.connect("pyerbML.db",detect_types=sqlite3.PARSE_DECLTYPES)

        if "MLModel" in kwargs:
            MLModel = kwargs['MLModel']
        else:
            MLModel = ["GPC", {"mainConfig":"GekkoMainKernel"}]

        if "targetAssetList" in kwargs:
            targetAssetList = kwargs['targetAssetList']
        else:
            targetAssetList = df.columns

        if "MemoryDepth" in kwargs:
            MemoryDepth = kwargs['MemoryDepth']
        else:
            MemoryDepth = 25

        if "FeaturesRegulators" in kwargs:
            FeaturesRegulators = kwargs['FeaturesRegulators']
        else:
            FeaturesRegulators = ["SelfRegulate", 10]

        if "RetsSmoothed" in kwargs:
            RetsSmoothed = kwargs['RetsSmoothed']
        else:
            RetsSmoothed = False

        modelList = []
        for i in range(len(targetAssetList)):
            if MLModel[0] == "GPC":
                if MLModel[1]['mainConfig'] == "GekkoMainKernel":
                    mainKernel = 1 * ConstantKernel() + 1 * RBF() + 1 * RationalQuadratic() + 1 * Matern(nu=0.5) \
                                 + 1 * Matern(nu=2.5) #  + 1*DotProduct() + 1*ExpSineSquared()
                elif MLModel[1]['mainConfig'] == "MRMaternKernel":
                    mainKernel = 1 * Matern(nu=0.5)
                elif MLModel[1]['mainConfig'] == "TrendMaternKernel":
                    mainKernel = 1 * Matern(nu=2.5)
                if RetsSmoothed == False:
                    mainKernel += 1 * WhiteKernel()
                sub_model = GaussianProcessClassifier(kernel=mainKernel, random_state=0)
            elif MLModel[0] == "RFC":
                if MLModel[1]['mainConfig'] == "MLdP0":
                    sub_model = RandomForestClassifier(n_estimators=1000, class_weight='balanced_subsample', criterion='entropy')
            modelList.append(sub_model)

        PredsMat = []
        model_c = 0
        for asset in targetAssetList:
            subAssetName = asset.replace(" ", "")
            model = modelList[model_c]
            print(subAssetName, ", model = ", model)

            if runMode == 'train':
                startPeriod = MemoryDepth + 1  # , rets.shape[0] - 750
            elif runMode == 'update':
                startPeriod = df.shape[0] - 5  # last 'x' days

            print("startPeriod = ", startPeriod)

            PredDataList = []
            FeaturesList = []
            for i in tqdm(range(startPeriod, df.shape[0] + 1)):
                med_rets = df.iloc[i - MemoryDepth:i, :]
                #if str(med_rets.index[-1]) == "2023-05-11 00:00:00": #Debug on index
                #    print("got here...")
                #    rolling_Predictions = rollPreds(med_rets, list(med_rets.columns).index(asset))
                #else:
                #    pass
                #"""
                if FeaturesRegulators[0] != "SelfRegulate":
                    DrivingFeatures = sorted(FeaturesExtract(med_rets, asset, "MDI", FeaturesRegulators[1]))
                    med_rets = med_rets[[asset]+DrivingFeatures]
                    FeaturesList.append([med_rets.index[-1]] + DrivingFeatures)
                "Here I build the regression problem"
                rolling_Predictions = rollPreds(med_rets, list(med_rets.columns).index(asset))
                PredDataList.append(rolling_Predictions)
                #"""
            try:
                PredsDF = pd.DataFrame(PredDataList,columns=["date", asset] + [str(x) for x in list(model.classes_)])
            except Exception as e:
                print(e)
                PredsDF = pd.DataFrame(PredDataList)
                print(PredsDF)
                PredsDF.columns = ["date", asset] + ["-1.0","0.0", "1.0"]

            if FeaturesRegulators[0] != "SelfRegulate":
                FeaturesDF = pd.DataFrame(FeaturesList,columns=["date"] + ["Feature_"+str(x) for x in range(len(DrivingFeatures))])
                try:
                    FeaturesDF = FeaturesDF.set_index("date", drop=True)
                except Exception as e:
                    print(e)

            PredsDF = PredsDF.set_index('date', drop=True)
            PredsDF = PredsDF.astype(float)
            PredsDF[[x for x in PredsDF.columns if (x == "-1.0") | (x == "0.0") | (x == "1.0")]] *= 100

            if runMode == 'train':
                PredsDF.to_sql(kwargs['StrategyName'] + "_" + subAssetName + '_PredsDF_FirstRun',
                               workConn, if_exists='replace')
                if FeaturesRegulators[0] != "SelfRegulate":
                    FeaturesDF.to_sql(kwargs['StrategyName'] + "_" + subAssetName + '_FeaturesDF_FirstRun',
                               workConn, if_exists='replace')
            if runMode == 'update':
                prev_PredsDF = pd.read_sql(
                    "SELECT * FROM " + kwargs['StrategyName'] + "_" + subAssetName + "_PredsDF",
                    workConn).set_index('date', drop=True)
                PredsDF = pd.concat([prev_PredsDF, PredsDF])
                PredsDF = PredsDF[~PredsDF.index.duplicated(keep='last')]
                #############################################################################################
                if FeaturesRegulators[0] != "SelfRegulate":
                    prev_FeaturesDF = pd.read_sql(
                        "SELECT * FROM " + kwargs['StrategyName'] + "_" + subAssetName + "_FeaturesDF",
                        workConn).set_index('date', drop=True)
                    FeaturesDF = pd.concat([prev_FeaturesDF, FeaturesDF])
                    FeaturesDF = FeaturesDF[~FeaturesDF.index.duplicated(keep='last')]

            PredsDF.to_sql(kwargs['StrategyName'] + "_" + subAssetName + '_PredsDF', workConn,
                           if_exists='replace')
            if FeaturesRegulators[0] != "SelfRegulate":
                FeaturesDF.to_sql(kwargs['StrategyName'] + "_" + subAssetName + '_FeaturesDF', workConn,
                           if_exists='replace')

            PredsMat.append(PredsDF[asset])

            model_c += 1

        PredsAllDF = pd.concat(PredsMat, axis=1).sort_index()
        PredsAllDF.fillna(0).to_sql(kwargs['StrategyName'] + '_PredsMat', workConn,if_exists='replace')

    def RollDecisionTree(df0, **kwargs):
        if 'RollMode' in kwargs:
            RollMode = kwargs['RollMode']
        else:
            RollMode = 'ExpWindow'

        if 'X' in kwargs:
            X = kwargs['X']
        else:
            X = df0.columns[:-1]

        if 'Y' in kwargs:
            Y = kwargs['Y']
        else:
            Y = df0.columns[-1]

        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = 25

        splitThresholdList = []
        for i in range(st, len(df0) + 1):

            if RollMode == 'RollWindow':
                df = df0.iloc[i - st:i, :]
            else:
                df = df0.iloc[0:i, :]

            # Create the decision tree classifier
            clf = DecisionTreeClassifier(
                criterion='gini',
                splitter='best',
                max_depth=None,
                min_samples_split=2,#2, 10
                min_samples_leaf=1,#1, 5
                min_weight_fraction_leaf=0.0,
                max_features=None,
                random_state=None,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                class_weight=None,
                ccp_alpha=0.0)

            trainX = df[X].values
            if trainX.shape[1] == 1:
                trainX = trainX.reshape(-1, 1)
            trainY = df[Y].values

            # Train the classifier
            clf.fit(trainX, trainY)
            # Print the split level of the tree
            tree_threshold_Info = clf.tree_.threshold
            splitThresholdList.append([df.index[-1], tree_threshold_Info[0]])

        splitThresholdDF = pd.DataFrame(splitThresholdList, columns=["date", "TreeThreshold"]).set_index("date", drop=True)

        return splitThresholdDF

    def DecisionTrees_Signal_Filter(df, filteringIndicatorsDF, DecisionTrees_RV, targetFilters, **kwargs):
        if "DT_Aggregate_Mode" in kwargs:
            DT_Aggregate_Mode = kwargs['DT_Aggregate_Mode']
        else:
            DT_Aggregate_Mode = "Additive"

        if "BarrierDirections" in kwargs:
            BarrierDirections = kwargs['BarrierDirections']
        else:
            BarrierDirections = ["Under" for x in targetFilters]

        FilterPairsList = []
        c = 0
        for filter in targetFilters:
            for asset in df.columns:
                FilterPairsList.append([filter, asset, BarrierDirections[c]])
            c += 1
        DT_FilterList = []
        for FilterPair in FilterPairsList:
            DT_SubFilter = pe.sign(
                filteringIndicatorsDF[FilterPair[0]] - DecisionTrees_RV[FilterPair[0] + "_" + FilterPair[1]])
            if FilterPair[2] == "Under":
                DT_SubFilter[DT_SubFilter < 0] = 0
            elif FilterPair[2] == "Upper":
                DT_SubFilter[DT_SubFilter > 0] = 0
            elif FilterPair[2] == "UnderReverse":
                DT_SubFilter[DT_SubFilter < 0] *= -1
            elif FilterPair[2] == "UpperReverse":
                DT_SubFilter[DT_SubFilter > 0] *= -1
            DT_SubFilter.name = FilterPair[0] + "_" + FilterPair[1]
            DT_FilterList.append(DT_SubFilter)

        DT_FilterDF = pd.concat(DT_FilterList, axis=1).sort_index().fillna(1)

        "Aggregate DT Filters"
        DT_Aggregated_SigDF = pd.DataFrame(0, index=df.index, columns=df.columns)
        for c in DT_FilterDF.columns:
            TrAsset = c.split("_")[1]
            if DT_Aggregate_Mode == "Additive":
                DT_Aggregated_SigDF[TrAsset] += DT_FilterDF[c]
            elif DT_Aggregate_Mode == "Multiplicative":
                DT_Aggregated_SigDF[TrAsset] *= DT_FilterDF[c]

        DT_Aggregated_SigDF = pe.sign(DT_Aggregated_SigDF)

        return [DT_FilterDF, DT_Aggregated_SigDF]

class ManSee:

    def gRollingManifold(manifoldIn, df0, **kwargs):
        if 'st' in kwargs:
            st = kwargs['st']
        else:
            st = df0.shape[1] + 1
        if st < df0.shape[1] + 1:
            st = df0.shape[1] + 1

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
            ProjectionMode = 'NoTranspose'

        "CALCULATE ROLLING STATISTIC"
        if manifoldIn == 'CustomMetric':
            if 'CustomMetricStatistic' in kwargs:
                CustomMetricStatistic = kwargs['CustomMetricStatistic']
                metaDF_Rolling = pe.rollStatistics(df0.copy(), CustomMetricStatistic)
            else:
                CustomMetricStatistic = None
                metaDF_Rolling = df0.copy()

            if 'CustomMetric' in kwargs:
                CustomMetric = kwargs['CustomMetric']
            else:
                CustomMetric = "euclidean"

        EmbeddingPackList = []
        for i in tqdm(range(st, len(df0) + 1)):
            try:

                #print("Step:", i, " of ", len(df0) + 1)
                if RollMode == 'RollWindow':
                    df = df0.iloc[i - st:i, :]
                else:
                    df = df0.iloc[0:i, :]

                latestIndex = df.index[-1]

                NumProjections = len(df.columns)
                if ProjectionMode == 'Transpose':
                    df = df.T

                x = df.values

                if Scaler == 'Standard':
                    x = StandardScaler().fit_transform(x)

                if manifoldIn == 'CustomMetric':
                    customMetric = pe.Metric(metaDF_Rolling, statistic=CustomMetricStatistic, metric=CustomMetric)
                    EmbeddingPackList.append({"latestIndex":latestIndex,"ModelObj":customMetric, "Projections":[], "ExtraData":[]})

                elif manifoldIn == 'PCA':
                    pca = PCA(n_components=NumProjections)
                    X_pca = pca.fit_transform(x)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": pca, "Projections": X_pca, "ExtraData": []})

                elif manifoldIn == 'PCA_Correlations':
                    cor_mat = pd.DataFrame(np.corrcoef(x.T)).abs().fillna(0).values
                    eig_vecs, eig_vals, vh = np.linalg.svd(cor_mat, full_matrices=True)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": cor_mat, "Projections": eig_vecs, "ExtraData": [eig_vals, vh]})

                elif manifoldIn == 'BetaRegressV':
                    BetaKernelDF = pe.BetaKernel(df)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": []})

                elif manifoldIn == 'BetaProject':
                    BetaKernelDF = pe.BetaKernel(df)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": []})

                elif manifoldIn == 'BetaRegressH':
                    BetaKernelDF = pe.BetaKernel(df)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": []})

                elif manifoldIn == 'BetaRegressC':
                    BetaKernelDF = pe.BetaKernel(df)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": []})

                elif manifoldIn == 'BetaDiffusion':
                    BetaKernelDF = pe.BetaKernel(df)
                    BetaKernelDF *= 1/BetaKernelDF.median()
                    U, s, VT = svd(BetaKernelDF.values)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": [U, s, VT ]})

                elif manifoldIn == 'Beta':
                    BetaKernelDF = pe.BetaKernel(df).fillna(0)
                    U, s, VT = svd(BetaKernelDF.values)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": BetaKernelDF, "ExtraData": [U, s, VT ]})

                elif manifoldIn == 'MultipleRegress':

                    MRKernelDF = pe.MultiRegressKernel(df)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": [], "Projections": MRKernelDF, "ExtraData": []})

                elif manifoldIn == 'DMAPS':
                    pcm = pfold.PCManifold(x)
                    pcm.optimize_parameters(random_state=1)
                    dmap = dfold.DiffusionMaps(
                        pfold.GaussianKernel(epsilon=pcm.kernel.epsilon),
                        n_eigenpairs=NumProjections-1,
                        dist_kwargs=dict(cut_off=pcm.cut_off),
                    )

                    dmap = dmap.fit(pcm, store_kernel_matrix=True)
                    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": dmap, "Projections": evecs, "ExtraData": [evals]})

                elif manifoldIn == 'LLE':
                    n_neighbors = int(np.sqrt(x.shape[0]))
                    lle = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=NumProjections-1, method="standard", n_jobs=-1)
                    X_lle = lle.fit_transform(x)
                    EmbeddingPackList.append({"latestIndex": latestIndex, "ModelObj": lle, "Projections": X_lle, "ExtraData": [n_neighbors]})

            except Exception as e:
                print(e)
                EmbeddingPackList.append([latestIndex, []])

        return [df0, EmbeddingPackList]

    def ManifoldPackUnpack(ID, ManifoldPackList, **kwargs):

        if "TemporalExtraction" in kwargs:
            TemporalExtraction = kwargs['TemporalExtraction']
        else:
            TemporalExtraction = "LastValue"

        InputData = ManifoldPackList[0]
        print(InputData)
        EmbeddingDataPack = ManifoldPackList[1]

        if "NoTranspose_" in ID:
            ManifoldTS = [pd.DataFrame(None, index=InputData.index, columns=InputData.columns) for i in range(EmbeddingDataPack[0]["Projections"].shape[1])]
        else:
            ManifoldTS = [pd.DataFrame(None, index=InputData.index, columns=[InputData.columns[i]]) for i in range(EmbeddingDataPack[0]["Projections"].shape[1])]

        for pack in tqdm(EmbeddingDataPack):
            for c in range(pack["Projections"].shape[1]):
                #########################################################################################
                if "PCA_" in ID:
                    ModelData = pack["ModelObj"].components_[c]
                elif "LLE_" in ID:
                    ModelData = []
                #########################################################################################
                if "NoTranspose_" in ID:
                    ManifoldTS[c].loc[pack["latestIndex"]] = ModelData
                else:
                    if TemporalExtraction == "LastValue":
                        ManifoldTS[c].loc[pack["latestIndex"]] = ModelData[-1]
                    elif TemporalExtraction in ["PearsonCorrelationVal", "PearsonCorrelationPVal", "AdjMI", "DecisionTree"]:
                        ModelDataSeries = pd.Series(ModelData)
                        InputDataRefIndex = InputData.index.get_loc(pack["latestIndex"])+1
                        InputDataRefSeries = InputData[InputData.columns[c]].iloc[InputDataRefIndex-ModelDataSeries.shape[0]:InputDataRefIndex]
                        if TemporalExtraction == "PearsonCorrelationVal":
                            ManifoldTS[c].loc[pack["latestIndex"]] = pearsonr(ModelDataSeries.values, InputDataRefSeries.values)[0]
                        elif TemporalExtraction == "PearsonCorrelationPVal":
                            ManifoldTS[c].loc[pack["latestIndex"]] = pearsonr(ModelDataSeries.values, InputDataRefSeries.values)[1]
                        elif TemporalExtraction == "AdjMI":
                            ManifoldTS[c].loc[pack["latestIndex"]] = adjusted_mutual_info_score(ModelDataSeries.values, InputDataRefSeries.values)
                        elif TemporalExtraction == "DecisionTree":
                            InputDataRefSeries[InputDataRefSeries > 0] = 1
                            InputDataRefSeries[InputDataRefSeries < 0] = -1
                            InputDataRefSeries[InputDataRefSeries == 0] = 0
                            # Create the decision tree classifier
                            clf = DecisionTreeClassifier(
                                criterion='gini',
                                splitter='best',
                                max_depth=None,
                                min_samples_split=2,  # 2, 10
                                min_samples_leaf=1,  # 1, 5
                                min_weight_fraction_leaf=0.0,
                                max_features=None,
                                random_state=None,
                                max_leaf_nodes=None,
                                min_impurity_decrease=0.0,
                                class_weight=None,
                                ccp_alpha=0.0)

                            trainX = ModelDataSeries.values.reshape(-1, 1)
                            trainY = InputDataRefSeries.values

                            # Train the classifier
                            clf.fit(trainX, trainY)
                            # Print the split level of the tree
                            tree_threshold_Info = clf.tree_.threshold
                            ManifoldTS[c].loc[pack["latestIndex"]] = tree_threshold_Info[0]

        if "_Transpose_" in ID:
            Out = pd.concat(ManifoldTS, axis=1).sort_index()
            print(Out)
        else:
            Out = ManifoldTS

        return Out
