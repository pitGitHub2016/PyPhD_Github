from Slider import Slider as sl
import scipy.io, glob
from scipy.interpolate import NearestNDInterpolator
import itertools, math
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.ar_model import AutoReg, AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.interpolate import RBFInterpolator
from keras.models import Sequential
from keras.layers import Dense
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from datafold.dynfold import LocalRegressionSelection

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return [mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))]

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def reframeData(dataIn, reframeStep, varSelect, **kwargs):
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

def Embed(method, X_train_local, target_intrinsic_dim, **kwargs):
    if "LLE_neighbors" in kwargs:
        LLE_neighbors = kwargs["LLE_neighbors"]
    else:
        LLE_neighbors = 50

    if "DM" in method:
        X_pcm = pfold.PCManifold(X_train_local)
        X_pcm.optimize_parameters()

        if target_intrinsic_dim >= 10:
            n_eigenpairsIn = target_intrinsic_dim+1
        else:
            n_eigenpairsIn = 10

        dmap_local = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
            n_eigenpairs=n_eigenpairsIn,
            dist_kwargs=dict(cut_off=X_pcm.cut_off),
        )
        dmap_local = dmap_local.fit(X_pcm)
        # evecs_raw, evals_raw = dmap.eigenvectors_, dmap.eigenvalues_

        if X_train_local.shape[0] < 500:
            n_subsampleIn = X_train_local.shape[0] - 1
        else:
            n_subsampleIn = 500

        selection = LocalRegressionSelection(
            intrinsic_dim=target_intrinsic_dim, n_subsample=n_subsampleIn, strategy="dim"
        ).fit(dmap_local.eigenvectors_)

        # print("selection.evec_indices_ = ", selection.evec_indices_)
        parsimoniousEigs = ",".join([str(x) for x in selection.evec_indices_])

        target_mapping = selection.transform(dmap_local.eigenvectors_)
        # print("target_mapping.shape = ", target_mapping.shape)

        out = [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, dmap_local.eigenvalues_[selection.evec_indices_]]
    elif "LLE" in method:
        lle = manifold.LocallyLinearEmbedding(n_neighbors=LLE_neighbors, n_components=target_intrinsic_dim,
                                              method="standard", n_jobs=-1)
        target_mapping = lle.fit_transform(X_train_local)

        out = [target_mapping, "none", 1, []]
    elif "PCA" in method:
        pca = PCA(n_components=target_intrinsic_dim)
        evecs = pca.fit_transform(X_train_local.T)
        evals = pca.singular_values_
        explainedVarianceRatio = pca.explained_variance_ratio_
        target_mapping = pca.components_.T

        out = [target_mapping, ",".join(evecs), explainedVarianceRatio, evals]

    return out

def TradePreds(X_Preds, X_test):
    trSig = pd.DataFrame(X_Preds)
    X_test_df = pd.DataFrame(X_test)

    pnl = sl.rs(sl.sign(trSig) * X_test_df)
    pnl_sh = np.sqrt(252) * sl.sharpe(pnl)

    return pnl_sh

def PaperizePreds(X_Preds, X_test, **kwargs):  # CI_Lower_Band, CI_Upper_Band

    if 'outputFormat' in kwargs:
        outputFormat = kwargs['outputFormat']
    else:
        outputFormat = "RMSE"

    if 'roundingFlag' in kwargs:
        roundingFlag = kwargs['roundingFlag']
    else:
        roundingFlag = False

    localVarColumns = ["x" + str(x) for x in range(X_Preds.shape[1])]

    X_test_df = pd.DataFrame(X_test, columns=localVarColumns)
    X_Preds_df = pd.DataFrame(X_Preds, columns=localVarColumns)

    outList = []
    for col in X_test_df.columns: #[:7]
        if outputFormat == "RMSE":
            submetric = np.round(np.sqrt((1 / X_test_df.shape[0]) * np.sum((X_Preds_df[col].values - X_test_df[col].values) ** 2)),4)  # SIETTOS
        elif outputFormat == "Sharpe":
            sub_pnl = sl.sign(X_Preds_df[col]) * X_test_df[col]
            submetric = np.sqrt(252) * sl.sharpe(sub_pnl)
        outList.append(submetric)

    if outputFormat == "Sharpe":
        totalPnl = sl.rs(sl.sign(X_Preds_df) * X_test_df)
        totalSharpe = np.sqrt(252) * sl.sharpe(totalPnl)
        outList.append(totalSharpe)

    if roundingFlag == True:
        outList = [np.round(x, 3) for x in outList]

    if 'returnData' in kwargs:
        return outList
    else:
        return ' & '.join([str(x) for x in outList])

def get_ML_Predictions(mode, MLmethod, predictorsData, y_shifted, forecastHorizon):
    MLmethodSplit = MLmethod.split(",")
    if MLmethodSplit[1] == "GPR_Single":
        mainKernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()  # Official (29/8/2021)
        model_List = [
            GaussianProcessRegressor(kernel=mainKernel, alpha=0.01, n_restarts_optimizer=2, random_state=random_state)
            for var in range(predictorsData.shape[1])]
    elif MLmethod[0] == "ANN_Single":
        model_List = []
        for var in range(predictorsData.shape[1]):
            ANN_model = Sequential()
            # ANN_model.add(Dense(2, input_dim=predictorsData.shape[1], activation='sigmoid'))
            ANN_model.add(Dense(3, input_dim=predictorsData.shape[1], activation='sigmoid'))
            ANN_model.add(Dense(1, activation='sigmoid'))
            ANN_model.compile(loss='mse', optimizer='adam')
            model_List.append(ANN_model)

    if mode == "Main":
        Preds_List = []
        for step_i in tqdm(range(forecastHorizon)):

            models_preds_list = []
            for modelIn in range(len(model_List)):
                if step_i == 0:
                    roll_reframedData = reframeData(predictorsData, MLmethodSplit[2], modelIn)
                    model_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1])
                    # print("model_List[modelIn].score = ", model_List[modelIn].score(roll_reframedData[0], roll_reframedData[1]))
                    try:
                        print("model_List[", modelIn, "].kernel = ", model_List[modelIn].kernel_)
                    except:
                        pass
                    sub_row_Preds = model_List[modelIn].predict(roll_reframedData[2])
                else:
                    sub_row_Preds = model_List[modelIn].predict(total_row_subPred.reshape(roll_reframedData[2].shape))

                models_preds_list.append(sub_row_Preds[0][0])

            total_row_subPred = np.array(models_preds_list)
            # print("step_i = ", step_i, ", MLmethod = ", MLmethod, ", total_row_subPred = ", total_row_subPred)
            Preds_List.append(total_row_subPred)
    elif mode == "HardCoded":

        mainKernel1 = 1 * RBF() + 1 * WhiteKernel()  # "fixed", length_scale_bounds=(0,1)
        mainKernel2 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel3 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel4 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel5 = 1 * RBF() + 1 * WhiteKernel()
        gprModel1 = GaussianProcessRegressor(kernel=mainKernel1,
                                             random_state=random_state)  # n_restarts_optimizer=10, normalize_y=True
        gprModel2 = GaussianProcessRegressor(kernel=mainKernel2, random_state=random_state)
        gprModel3 = GaussianProcessRegressor(kernel=mainKernel3, random_state=random_state)
        gprModel4 = GaussianProcessRegressor(kernel=mainKernel4, random_state=random_state)
        gprModel5 = GaussianProcessRegressor(kernel=mainKernel5, random_state=random_state)

        # kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        # kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        # kernel3 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        # kernel4 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        # kernel5 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        # scale = StandardScaler()
        # predictorsData = scale.fit_transform(predictorsData)

        d1 = predictorsData[:predictorsData.shape[0] - 1]
        d2 = predictorsData[1:predictorsData.shape[0]]

        # SKLEARN
        gprModel1.fit(d1, d2[:, 0])
        gprModel2.fit(d1, d2[:, 1])
        gprModel3.fit(d1, d2[:, 2])
        gprModel4.fit(d1, d2[:, 3])
        gprModel5.fit(d1, d2[:, 4])

        # GPY
        # gprModel1 = GPy.models.GPRegression(d1, d2[:, 0].reshape(d2.shape[0], 1), kernel1)
        # gprModel2 = GPy.models.GPRegression(d1, d2[:, 1].reshape(d2.shape[0], 1), kernel2)
        # gprModel3 = GPy.models.GPRegression(d1, d2[:, 2].reshape(d2.shape[0], 1), kernel3)
        # gprModel4 = GPy.models.GPRegression(d1, d2[:, 3].reshape(d2.shape[0], 1), kernel4)
        # gprModel5 = GPy.models.GPRegression(d1, d2[:, 4].reshape(d2.shape[0], 1), kernel5)
        # gprModel1.optimize(messages=True)
        # gprModel2.optimize(messages=True)
        # gprModel3.optimize(messages=True)
        # gprModel4.optimize(messages=True)
        # gprModel5.optimize(messages=True)

        lastPreds = d1[-1].reshape(1, -1)
        print("lastPreds = ", lastPreds)
        print("lastPreds.shape = ", lastPreds.shape)

        Preds_List = []
        for step_i in range(500):
            gprModel1_Preds = gprModel1.predict(lastPreds)
            try:
                print(gprModel1.kernel_)
            except Exception as e:
                print(e)
            gprModel2_Preds = gprModel2.predict(lastPreds)
            gprModel3_Preds = gprModel3.predict(lastPreds)
            gprModel4_Preds = gprModel4.predict(lastPreds)
            gprModel5_Preds = gprModel5.predict(lastPreds)

            lastPreds = np.array(
                [gprModel1_Preds[0], gprModel2_Preds[0], gprModel3_Preds[0], gprModel4_Preds[0], gprModel5_Preds[0]])
            Preds_List.append(lastPreds)
            print("step_i = ", step_i, ", lastPreds = ", lastPreds)
            # print("lastPreds.shape = ", lastPreds.shape)
            lastPreds = lastPreds.reshape(1, -1)
            # print("lastPreds.shape = ", lastPreds.shape)
    # pd.DataFrame(Preds_List).plot()
    # plt.show()
    # time.sleep(3000)

    return Preds_List

random_state = 0
modelParamsList = []

def Lift(method, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation, knn, **kwargs):
    if method == 'GH':
        pcm = pfold.PCManifold(eig_trainingSet)
        pcm.optimize_parameters(random_state=random_state, k=knn)
        opt_epsilon = pcm.kernel.epsilon
        opt_cutoff = pcm.cut_off
        opt_n_eigenpairs = eig_trainingSet.shape[1]
        gh_interpolant_psi_to_X = GHI(pfold.GaussianKernel(epsilon=opt_epsilon),
                                      n_eigenpairs=opt_n_eigenpairs, dist_kwargs=dict(cut_off=opt_cutoff), )
        gh_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = gh_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gh_interpolant_psi_to_X.predict(eig_Simulation)
        # print("opt_epsilon = ", opt_epsilon)
        # print("opt_cutoff = ", opt_cutoff)
        modelParamsList.append([knn, opt_cutoff, opt_epsilon])

        "Optimize Parameters using BayesianCV"
        """
        n_iters = 5
        np.random.seed(random_state)

        train_indices, test_indices = train_test_split(
            np.random.permutation(X_trainingSet.shape[0]), train_size=2 / 3, test_size=1 / 3
        )

        class GHIGauss(GHI):
            def __init__(self, epsilon=1, n_eigenpairs=2, cut_off=np.inf):
                self.epsilon = epsilon
                self.n_eigenpairs = n_eigenpairs
                self.cut_off = cut_off

                super(GHIGauss, self).__init__(
                    kernel=pfold.GaussianKernel(self.epsilon),
                    n_eigenpairs=self.n_eigenpairs,
                    is_stochastic=False,
                    dist_kwargs=dict(cut_off=self.cut_off),
                )

        opt = BayesSearchCV(
            GHIGauss(),
            {
                "epsilon": Real(
                    pcm.kernel.epsilon / 2, pcm.kernel.epsilon * 2, prior="log-uniform"
                ),
                "cut_off": Real(pcm.cut_off / 2, pcm.cut_off * 2, prior="uniform"),
                "n_eigenpairs": Integer(10, 1000, prior="uniform"),
            },
            n_iter=n_iters,
            random_state=0,
            scoring=lambda estimator, x, y: estimator.score(
                x, y, multioutput="uniform_average"
            ),  # is to be maximized
            cv=[[train_indices, test_indices]],
            refit=False,  # we cannot refit to the entire dataset because this would alter the optimal kernel scale
        )

        # run the Bayesian optimization
        opt.fit(eig_trainingSet, X_trainingSet)

        # get best model and results from parameter search

        # refit best parameter set on training set (not entire dataset - the parameters are optimized for the training set!)
        optimal_GHI = GHIGauss(**opt.best_params_).fit(
            eig_trainingSet[train_indices, :], X_trainingSet[train_indices, :]
        )

        print(
            f"Previous epsilon: {pcm.kernel.epsilon}, cut-off: {pcm.cut_off}, #eigenpairs: {opt_n_eigenpairs}"
        )
        print(
            f"Optimal epsilon: {optimal_GHI.epsilon}, cut-off: {optimal_GHI.cut_off}, #eigenpairs: {optimal_GHI.n_eigenpairs}"
        )
        extrapolatedPsi_to_X = optimal_GHI.predict(eig_Simulation)
        """
    elif method == 'LP':
        lpyr_interpolant_psi_to_X = LPI(auto_adaptive=True)
        lpyr_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = lpyr_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = lpyr_interpolant_psi_to_X.predict(eig_Simulation)
    elif method == 'KR':
        mainKernel_Kriging_GP = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()  # Official (29/8/2021)
        gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP, normalize_y=True)
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        residual = gpr_model_fit.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gpr_model_fit.predict(eig_Simulation)
    elif method == 'SI':  # Simple Linear ND Interpolator
        knn_interpolator = NearestNDInterpolator(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = knn_interpolator(eig_Simulation)
        residual = extrapolatedPsi_to_X - X_testSet
    elif method == "RBF":
        extrapolatedPsi_to_X = RBFInterpolator(eig_trainingSet, X_trainingSet, kernel="linear", degree=1, neighbors=knn,
                                               epsilon=1)(eig_Simulation)
        residual = extrapolatedPsi_to_X - X_testSet

    try:
        mse_psnr_val = mse_psnr(extrapolatedPsi_to_X, X_testSet)
        mse = mse_psnr_val[0]
        # psnr = mse_psnr_val[1]
        # r2_score_val = r2_score(X_testSet, extrapolatedPsi_to_X)
        rmse = np.sqrt(mse)
        # nrmse = rmse / (np.amax(extrapolatedPsi_to_X)-np.amin(extrapolatedPsi_to_X))
        # mape = mean_absolute_percentage_error(X_testSet, extrapolatedPsi_to_X)
    except Exception as e:
        # print(e)
        mse_psnr_val = np.nan
        mse = np.nan
        # psnr = np.nan
        # r2_score_val = np.nan
        rmse = np.nan
        # nrmse = np.nan
        # mape = np.nan

    return [extrapolatedPsi_to_X, mse, rmse, residual]

def Test():
    pass

def RollingRunProcess(params):
    liftMethod = params['liftMethod']
    trainSetLength = params['trainSetLength']
    data = params['data']
    data_embed = params['data_embed']
    rolling_Embed_Memory = params['rolling_Embed_Memory']
    target_intrinsic_dim = params['target_intrinsic_dim']
    mode = params['mode']
    modeSplit = mode.split(',')
    rolling_Predict_Memory = params['rolling_Predict_Memory']
    embedMethod = params['embedMethod']
    processName = '_'.join([str(x) for x in list(params.values())[3:]])
    print(processName)
    # time.sleep(3000)

    roll_parsimoniousEigs_List = []
    PredsList = []
    for i in tqdm(range(trainSetLength, data.shape[0], 1)):

        try:
            if liftMethod == 'FullModel':
                roll_X_test = data[i]

                roll_target_mapping = data[i - rolling_Embed_Memory:i]
                roll_parsimoniousEigs = ""
                roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
                modelListSpectrum = roll_target_mapping.shape[1]
            else:

                roll_X_train_embed = data_embed[i - rolling_Embed_Memory:i]
                roll_X_train_lift = data[i - rolling_Embed_Memory:i]
                roll_X_test = data[i]

                roll_target_mapping_List = Embed(embedMethod, roll_X_train_embed, target_intrinsic_dim)
                roll_target_mapping = roll_target_mapping_List[0]
                roll_parsimoniousEigs = roll_target_mapping_List[1]
                roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
                modelListSpectrum = roll_target_mapping.shape[1]

            if i == trainSetLength:
                mainRolling_kernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()
                model_GPR_List = [GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0) for var in range(modelListSpectrum)]

            ##############################################################################################################
            if modeSplit[1].strip() == "Single_AR":
                # print("roll_target_mapping.shape[1] = ", roll_target_mapping.shape[1])
                subPredsList = []
                for varNum in range(roll_target_mapping.shape[1]):
                    AR_submodel = AR(roll_target_mapping[-rolling_Predict_Memory:, varNum])
                    bestLag = AR_submodel.select_order(int(modeSplit[2]), 'aic', 'nc')
                    arima_model = ARIMA(roll_target_mapping[-rolling_Predict_Memory:, varNum], order=(bestLag, 0, 0))
                    arima_model_fit = arima_model.fit(disp=0)
                    FullModel_Single_AR_Preds = arima_model_fit.forecast(steps=1, alpha=0.05)
                    subPredsList.append(FullModel_Single_AR_Preds[0][0])
                row_Preds = np.array(subPredsList).reshape(1, roll_target_mapping.shape[1])
            elif modeSplit[1].strip() == 'VAR':
                roll_forecasting_model = VAR(roll_target_mapping[-rolling_Predict_Memory:])
                roll_model_fit = roll_forecasting_model.fit(int(modeSplit[2]))
                roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=1, alpha=0.05)
                row_Preds = roll_target_mapping_Preds_All[0]
            elif modeSplit[1].strip() == 'GPR_Single':
                models_preds_list = []
                for modelIn in range(len(model_GPR_List)):
                    roll_reframedData = reframeData(roll_target_mapping[-rolling_Predict_Memory:], int(modeSplit[2]), modelIn)
                    try:
                        model_GPR_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1])
                        sub_row_Preds, sub_roll_target_mapping_Preds_Std = model_GPR_List[modelIn].predict(roll_reframedData[2], return_std=True)
                        models_preds_list.append(sub_row_Preds[0][0])
                    except:
                        models_preds_list.append(0)
                row_Preds = np.array(models_preds_list).reshape(1, roll_target_mapping.shape[1])

            if liftMethod == 'FullModel':
                PredsList.append(row_Preds[0])
            else:
                # Lifted_X_Preds = Lift(liftMethod, X_train_lift, X_test, target_mapping, mapped_Preds, knn)[0]
                single_lifted_Preds = Lift("GH", roll_X_train_lift, roll_X_test, roll_target_mapping, row_Preds, params["neighborsIn"])[0]
                PredsList.append(single_lifted_Preds)
        except Exception as e:
            print(e)
            PredsList.append([0 for var in range(len(PredsList[-1]))])

    Preds = np.array(PredsList)

    print(Preds.shape)
    pickle.dump(Preds, open(RollingRunnersPath + processName + ".p", "wb"))

def RunPythonDM(paramList):
    label = paramList[0]
    mode = paramList[1]
    simulationNumber = paramList[2]
    datasetsPath = paramList[3]
    RollingRunnersPath = paramList[4]
    TakensSpace = paramList[6]
    embedMethod = paramList[5] + "_" + TakensSpace
    TakensSpaceSpecs = TakensSpace.split(",")
    target_intrinsic_dim = paramList[7]
    forecastHorizon = paramList[8]

    matFileName = datasetsPath + label + "_" + str(simulationNumber) + '.mat'

    mat = scipy.io.loadmat(matFileName)
    modeSplit = mode.split(',')

    if TakensSpaceSpecs[0] == "no":

        data = mat['Ytrain']
        data_embed = data
        trainSetLength = data.shape[0] - forecastHorizon

        X_train = data[:trainSetLength]
        X_test = data[trainSetLength:]
        X_test_shifted = sl.S(pd.DataFrame(X_test)).values
        X_test_shifted[0] = X_train[-1]

        X_train_embed = X_train
        X_train_predict = X_train
        X_train_lift = X_train
    elif TakensSpaceSpecs[0] == "yes":

        data = mat['Ytrain1']
        if TakensSpaceSpecs[2] == "extended":
            data_embed = mat['Ytrain']
        else:
            data_embed = data

        if TakensSpaceSpecs[1] == "plain":
            pass
        elif TakensSpaceSpecs[1] == "shift_var1_3":

            data_df = pd.DataFrame(data)
            data_df.iloc[:, 0] = sl.S(data_df.iloc[:, 0], nperiods=3)
            data_df = data_df.dropna()
            data = data_df.values

        trainSetLength = data.shape[0] - forecastHorizon
        X_train = data[:trainSetLength]
        X_test = data[trainSetLength:]
        X_test_shifted = sl.S(pd.DataFrame(X_test)).values
        X_test_shifted[0] = X_train[-1]
        X_train_predict = X_train
        trainEmbedSetLength = data_embed.shape[0] - forecastHorizon
        X_train_embed = data_embed[:trainEmbedSetLength]
        X_train_lift = mat['YtrainLift'][:trainEmbedSetLength]

    if modeSplit[0] == 'FullModel_Static':
        if modeSplit[1] == "Single_AR":
            print("X_train_predict.shape[1] = ", X_train_predict.shape[1])
            Preds = []
            for varNum in range(X_train_predict.shape[1]):
                print("varNum = ", varNum)
                AR_submodel = AR(X_train_predict[:, varNum])
                bestLag = AR_submodel.select_order(int(modeSplit[2]), 'aic', 'nc')
                arima_model = ARIMA(X_train_predict[:, varNum], order=(bestLag, 0, 0))
                arima_model_fit = arima_model.fit()
                FullModel_Single_AR_Preds = arima_model_fit.forecast(steps=forecastHorizon, alpha=0.05)
                print("bestLag = ", bestLag)
                # print(FullModel_Single_AR_Preds)
                # time.sleep(300)
                Preds.append([FullModel_Single_AR_Preds, bestLag])
        elif modeSplit[1] == "VAR":
            forecasting_model = VAR(X_train_predict)
            model_fit = forecasting_model.fit(int(modeSplit[2]))
            FullModel_VAR_Preds = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
            Preds = FullModel_VAR_Preds[0]
        elif modeSplit[1] in ["GPR_Single", "ANN_Single"]:
            Preds_List = get_ML_Predictions("Main", modeSplit, X_train_predict, X_test_shifted, forecastHorizon)
            Preds = pd.DataFrame(Preds_List).values

        try:
            FullModel_Static_PaperText = PaperizePreds(Preds, X_test)
            print("FullModel_Static_PaperText = ", FullModel_Static_PaperText)
        except Exception as e:
            print(e)
        pickle.dump(Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + embedMethod + ".p","wb"))
    elif modeSplit[0] == "FullModel_Rolling":

        rolling_Embed_Memory = data.shape[0] - forecastHorizon
        rolling_Predict_Memory = rolling_Embed_Memory

        paramsProcess = {
            'data': data,
            'data_embed': data_embed,
            'X_test': X_test,
            'label': label,
            'simulationNumber': simulationNumber,
            'liftMethod': 'FullModel',
            'trainSetLength': trainSetLength,
            'rolling_Embed_Memory': rolling_Embed_Memory,
            'target_intrinsic_dim': target_intrinsic_dim, 'mode': mode,
            'rolling_Predict_Memory': rolling_Predict_Memory,
            'embedMethod': embedMethod,
            'kernelIn': np.nan, 'degreeIn': np.nan, 'neighborsIn': np.nan, 'epsilonIn': np.nan
        }

        RollingRunProcess(paramsProcess)
    ###################################################################################################################
    elif modeSplit[0] == 'Static_run':

        ###################################### REDUCED TARGET SPACE #######################################
        "Check if exists already"
        allProcessedAlready = [f for f in glob.glob(RollingRunnersPath + '*.p')]
        knn = 50  # (neighbors used across analysis)

        "Lift the embedded space predictions to the Original Space"
        for liftMethod in ["GH"]:  # , "KR", "SI", "RBF"

            "Embed"
            target_mapping_List = Embed(embedMethod, X_train_embed, target_intrinsic_dim)
            target_mapping = target_mapping_List[0]
            parsimoniousEigs = target_mapping_List[1]
            target_mapping_EigVals = target_mapping_List[3]

            lift_outputFile = RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + embedMethod + "_" + liftMethod + "_" + str(target_intrinsic_dim) + ".p"

            if lift_outputFile not in allProcessedAlready:

                "Forecast and get CIs for the embedded space"
                if modeSplit[1] == "VAR":
                    forecasting_model = VAR(target_mapping)
                    model_fit = forecasting_model.fit(int(modeSplit[2]))
                    target_mapping_Preds_All = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
                    mapped_Preds = target_mapping_Preds_All[0]
                elif modeSplit[1] in ["GPR_Single", "ANN_Single"]:
                    Preds_List = get_ML_Predictions("Main", modeSplit[1], target_mapping, [], forecastHorizon)
                    mapped_Preds = pd.DataFrame(Preds_List).values

                try:
                    Lifted_X_Preds = Lift(liftMethod, X_train_lift, X_test, target_mapping, mapped_Preds, knn)[0]
                except Exception as e:
                    print(str(simulationNumber) + "_" + mode.replace(",", "_") + "_" + embedMethod + "_" + liftMethod)
                    print(e)

                pickle.dump([Lifted_X_Preds, mapped_Preds, parsimoniousEigs], open(lift_outputFile, "wb"))
            else:
                print(lift_outputFile, " already exists!")
    ###################################################################################################################
    elif modeSplit[0] == 'Rolling_run':

        rolling_Embed_Memory = data.shape[0] - forecastHorizon
        rolling_Predict_Memory = rolling_Embed_Memory

        paramsProcess = {
            'data': data,
            'data_embed': data_embed,
            'X_test': X_test,
            'label': label,
            'simulationNumber': simulationNumber,
            'liftMethod': "GH",
            'trainSetLength': trainSetLength,
            'rolling_Embed_Memory': rolling_Embed_Memory,
            'target_intrinsic_dim': target_intrinsic_dim, 'mode': mode,
            'rolling_Predict_Memory': rolling_Predict_Memory,
            'embedMethod': embedMethod,
            'kernelIn': "linear", 'degreeIn': 1, 'neighborsIn': 50, 'epsilonIn': 1
        }

        RollingRunProcess(paramsProcess)
    ###################################################################################################################
    elif mode == 'StaticFullEigvalPlot':

        ###################################### FULL EIGENSPACE #######################################
        X_pcm_full = pfold.PCManifold(X_train)
        X_pcm_full.optimize_parameters()

        dmap_full = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=X_pcm_full.kernel.epsilon),
            n_eigenpairs=X_train.shape[0] - 1,
            dist_kwargs=dict(cut_off=X_pcm_full.cut_off),
        )
        dmap_full = dmap_full.fit(X_pcm_full)
        evecs_full, evals_full = dmap_full.eigenvectors_, dmap_full.eigenvalues_

        full_eigenval_df = pd.DataFrame(evals_full)
        full_eigenval_df.to_csv(label + "_" + mode + ".csv")

def Reporter(mode, datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim, **kwargs):
    if 'reportPercentilesFlag' in kwargs:
        reportPercentilesFlag = kwargs['reportPercentilesFlag']
    else:
        reportPercentilesFlag = True

    if 'outputFormatReporter' in kwargs:
        outputFormatReporter = kwargs['outputFormatReporter']
    else:
        outputFormatReporter = "RMSE"

    if 'RiskParityFlag' in kwargs:
        RiskParityFlag = kwargs['RiskParityFlag']
    else:
        RiskParityFlag = "NO"

    if mode == "RunRandomWalks":

        allDataListRW = []
        for dataset_total_name in glob.glob(datasetsPath + '*.mat'):
            mat_dataset = scipy.io.loadmat(dataset_total_name)
            # dataRW = mat_dataset['y']
            if "Delay" in dataset_total_name:
                dataRW = mat_dataset['Ytrain1']
            else:
                dataRW = mat_dataset['Ytrain']

            if RiskParityFlag == "Yes":
                dataRW_df = pd.DataFrame(dataRW)
                riskParityVol = np.sqrt(252) * sl.S(sl.rollerVol(dataRW_df, 250)) * 100
                dataRW_df = (dataRW_df / riskParityVol).replace([np.inf, -np.inf], 0)
                dataRW = dataRW_df.values

            trainSetLengthRW = dataRW.shape[0] - forecastHorizon
            X_testRW = dataRW[trainSetLengthRW:]

            PredsRW = sl.S(pd.DataFrame(X_testRW)).fillna(0).values
            metricsVarsRW = PaperizePreds(PredsRW, X_testRW, returnData='yes', outputFormat=outputFormatReporter)
            allDataListRW.append([dataset_total_name.split("\\")[-1] + "_" + "-" + "_" + "-" + "_" + "-" + "_" + "-", metricsVarsRW])

        dataDFRW = pd.DataFrame(allDataListRW, columns=["file_name", "metricsVars"]).set_index("file_name", drop=True)
        dataDFRW.to_excel("Reporter_dataDF_RW_raw_"+RiskParityFlag+".xlsx")
    elif mode == "Run":

        allDataList = []
        for file_total_name in tqdm(glob.glob(RollingRunnersPath + '\\*.p')):
            file_name = file_total_name.split("\\")[-1]
            file_name_split = file_name.split("_")
            label = file_name_split[0]
            simulationNumber = file_name_split[1]

            matFileName = datasetsPath + label + "_" + str(simulationNumber) + '.mat'
            mat = scipy.io.loadmat(matFileName)

            if "Delay" in matFileName:
                data = mat['Ytrain1']
            else:
                data = mat['Ytrain']

            if RiskParityFlag == "Yes":
                data_df = pd.DataFrame(data)
                riskParityVol = np.sqrt(252) * sl.S(sl.rollerVol(data_df, 250)) * 100
                data_df = (data_df / riskParityVol).replace([np.inf, -np.inf], 0)
                data = data_df.values

            PredsData = pickle.load(open(file_total_name, "rb"))
            if "Single_AR_1" in file_total_name:
                print(file_total_name)
                Preds = np.empty(shape=(len(PredsData[0][0][0]), len(PredsData)))
                bestLagList = []
                c = 0
                for elem in PredsData:
                    subPredsData = elem[0]
                    subMean_Preds = subPredsData[0]
                    # sub_CI5_Preds = subPredsData[1]
                    # sub_CI95_Preds = subPredsData[2]
                    Preds[:, c] = subMean_Preds
                    bestLag = elem[1]
                    bestLagList.append(bestLag)
                    c += 1
                pd.DataFrame(bestLagList).to_excel(RollingRunnersPath + file_name + "_bestLag.xlsx")
            else:

                if type(PredsData) == list:
                    Preds = PredsData[0]
                    parsimoniousEigs = PredsData[2]
                else:
                    Preds = PredsData
                    parsimoniousEigs = ""

            trainSetLength = data.shape[0] - Preds.shape[0]
            X_test = data[trainSetLength:]

            if writeResiduals == 1:
                pd.DataFrame(Preds - X_test).to_excel(
                    "D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\ResidualsAnalysis\\" + file_name + "_Residuals.xlsx")

            if "FOREX" in datasetsPath:
                roundingFlagIn = True
            else:
                roundingFlagIn = False

            metricsVars = PaperizePreds(Preds, X_test, returnData='yes', outputFormat=outputFormatReporter,
                                     roundingFlag=roundingFlagIn)

            allDataList.append([file_name, metricsVars, parsimoniousEigs, target_intrinsic_dim, Preds.shape[0]])

        dataDF = pd.DataFrame(allDataList,
                              columns=["file_name", "metricsVars", "parsimoniousEigs", "target_intrinsic_dim",
                                       "Preds.shape[0]"]).set_index("file_name", drop=True)
        dataDF.to_excel("Reporter_dataDF_raw_"+RiskParityFlag+".xlsx")

    elif mode == "Read":

        dataDF = pd.concat([pd.read_excel("Reporter_dataDF_RW_raw_"+RiskParityFlag+".xlsx"), pd.read_excel("Reporter_dataDF_raw_"+RiskParityFlag+".xlsx")])
        dataDF['Dataset'] = dataDF["file_name"].str.split("_").str[0]
        dataDF['SimulationNumber'] = dataDF["file_name"].str.split("_").str[1]
        dataDF['ID'] = dataDF["file_name"].str.split("_", n=2).str[2:].apply(lambda x: x[0])
        dataDF['metricsVars'] = dataDF['metricsVars'].str.replace("[", "").str.replace("]", "")
        dataDF["Dataset_ID"] = dataDF['Dataset'] + "_" + dataDF["ID"]

        ReportList = []
        for elem in tqdm(set(dataDF["Dataset_ID"].tolist())):

            subDF = dataDF[dataDF["Dataset_ID"] == elem]

            "Calculate RMSE(variables), Medians and CIs"
            rmseVarsDF = subDF['metricsVars'].apply(lambda x: x.split(",")).apply(pd.Series).reset_index(
                drop=True).astype(float)
            rmseVarsDF_median = rmseVarsDF.median().tolist()

            percentile5 = np.percentile(rmseVarsDF.values, 5, axis=0)
            percentile95 = np.percentile(rmseVarsDF.values, 95, axis=0)

            if reportPercentilesFlag == True:
                reportText = ' & '.join([str(np.round(rmseVarsDF_median[c], 3)) + ' (' + str(
                    np.round(percentile5[c], 3)) + ',' + str(np.round(percentile95[c], 3)) + ')' for c in
                                         range(len(rmseVarsDF_median))])
            else:
                reportText = ' & '.join([str(np.round(rmseVarsDF_median[c], 3)) for c in range(len(rmseVarsDF_median))])

            ReportList.append([elem, reportText, subDF.shape[0]])

        ReportDF = pd.DataFrame(ReportList, columns=["Dataset_ID", "reportText", "#simulations"]).set_index(
            "Dataset_ID", drop=True)
        ReportDF.to_excel("Reporter_"+RiskParityFlag+".xlsx")

if __name__ == '__main__':

    label = "FxDataAdjRetsDelay"  # EEGsynthetic2, EEGsynthetic2nonlin, EEGsynthetic2nonlinDelay, FxDataAdjRets, FxDataAdjRetsDelay, FxDataAdjRetsMAJORS, FxDataAdjRetsMAJORSDelay
    embedMethod = "DM"  # PCA, LLE, DM
    #simulNum = 100
    simulNum = 1000

    pcFolderRoot = 'D:\Dropbox\VM_Backup\\'
    #pcFolderRoot = 'E:\Dropbox\Dropbox\VM_Backup\\'

    if label == "EEGsynthetic2":
        target_intrinsic_dim = 2
        TakensSpace = "no,plain"
        reportPercentilesFlagIn = True
        forecastHorizon = 500
        outputFormat = "RMSE"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\\'
    elif label == "EEGsynthetic2nonlin":
        target_intrinsic_dim = 3
        TakensSpace = "no,plain"
        reportPercentilesFlagIn = True
        forecastHorizon = 500
        outputFormat = "RMSE"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\\'
    elif label == "EEGsynthetic2nonlinDelay":
        TakensSpace = "yes,plain,non-extended"
        #TakensSpace = "yes,plain,extended"
        #TakensSpace = "yes,shift_var1_3"

        if TakensSpace == "yes,plain,non-extended":
            target_intrinsic_dim = 2
            #target_intrinsic_dim = 3
            #target_intrinsic_dim = 4
        elif TakensSpace == "yes,plain,extended":
            target_intrinsic_dim = 4

        reportPercentilesFlagIn = True
        forecastHorizon = 500
        outputFormat = "RMSE"
        RiskParityFlagIn = "No"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_DelayApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners_Delay\\'
    elif label in ["FxDataAdjRets", "FxDataAdjRetsDelay", "FxDataAdjRetsMAJORS", "FxDataAdjRetsMAJORSDelay"]:
        if label in ["FxDataAdjRets", "FxDataAdjRetsMAJORS"]:
            TakensSpace = "no,plain"
        else:
            TakensSpace = "yes,plain,non-extended"
            #TakensSpace = "yes,plain,extended"

        target_intrinsic_dim = 3
        #target_intrinsic_dim = 3
        reportPercentilesFlagIn = False
        # forecastHorizon = 500 # Static
        forecastHorizon = 5002 - 300 # Rolling 3rd
        #forecastHorizon = 5002 - 500 # Rolling Main
        #forecastHorizon = 5002 - 1000  # Rolling 2nd
        outputFormat = "Sharpe"
        RiskParityFlagIn = "Yes" # "Yes", "No"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_FOREX\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners_FOREX\\'

    ##### STATIC ####
    #processToRun = "FullModel_Static,Single_AR,1"
    #processToRun = "FullModel_Static,VAR,1"
    #processToRun = "FullModel_Static,VAR,3"
    #processToRun = "FullModel_Static,GPR_Single"
    # processToRun = "FullModel_Static,ANN_Single"
    #processToRun = "Static_run,VAR,1"
    #processToRun = "Static_run,VAR,3"
    # processToRun = "Static_run,GPR_Single" # Siettos
    # processToRun = "Static_run,ANN_Single"

    ##### ROLLING ####
    #processToRun = "FullModel_Rolling,Single_AR,5"
    #processToRun = "FullModel_Rolling,VAR,3"
    #processToRun = "FullModel_Rolling,VAR,10"
    #processToRun = "Rolling_run,VAR,1"
    processToRun = "Rolling_run,VAR,3"
    #processToRun = "Rolling_run,VAR,10"
    #processToRun = "Rolling_run,GPR_Single,1"
    #processToRun = "Rolling_run,GPR_Single,3"
    #processToRun = "Rolling_run,GPR_Single,5"
    #processToRun = "Rolling_run,GPR_Single,7"

    "Run Single Process (for testing)"
    RunPythonDM([label, processToRun, 0, datasetsPath, RollingRunnersPath, embedMethod, TakensSpace, target_intrinsic_dim,forecastHorizon])

    # Test()

    runProcessesFlag = 1
    writeResiduals = 0

    if runProcessesFlag == 0:

        simulList = []
        for simul in range(simulNum):
            try:
                simulList.append([label, processToRun, simul, datasetsPath, RollingRunnersPath, embedMethod, TakensSpace,target_intrinsic_dim, forecastHorizon])
            except Exception as e:
                print(simul)
                print(e)
        p = mp.Pool(mp.cpu_count())
        result = p.map(RunPythonDM, tqdm(simulList))
        p.close()
        p.join()
    elif runProcessesFlag == 1:
        print("REPORTER ......... ")
        " REPORT RESULTS "

        Reporter("RunRandomWalks", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                 reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)
        Reporter("Run", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                 reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)
        Reporter("Read", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                 reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)

### NOTES ###
"""
"Build Combinations of the several CIs"
varSet = [[] for varN in range(Preds.shape[1])]
for varN in range(Preds.shape[1]):
    sub_pred_df = pd.DataFrame(Preds.iloc[:, varN])
    sub_varName = list(sub_pred_df.columns)[0]
    sub_pred_df.columns = ["Mean_"+sub_varName]
    sub_pred_df["CI_lower_"+sub_varName] = pd.DataFrame(CI_Lower_Band.iloc[:, varN])
    sub_pred_df["CI_upper_"+sub_varName] = pd.DataFrame(CI_Upper_Band.iloc[:, varN])
    varSet[varN] = sub_pred_df

allCombosColumns = [x for x in list(itertools.product(*varSet))]
varSet_DF = pd.concat(varSet, axis=1)

statsList = []
for combo in tqdm(allCombosColumns):
    target_mapping_Preds_df = varSet_DF.loc[:,combo]
    target_mapping_Preds = target_mapping_Preds_df.values
    for liftMethod in ["RBF", "GH"]:
        if liftMethod == "RBF":
            for kernelIn in ['linear', 'gaussian']:
                for degreeIn in [1, 2, 3]:
                    for neighborsIn in [20, 50, 100]:
                        for epsilonIn in [0.01, 0.1, 1, 10]:
                            Lifted_X_Preds_RBF = RBFInterpolator(target_mapping, X_train, kernel=kernelIn, degree=degreeIn, neighbors=neighborsIn, epsilon=epsilonIn)(target_mapping_Preds)
                            mse_RBF = mse_psnr(Lifted_X_Preds_RBF, X_test)[0]
                            rmse_RBF = np.sqrt(mse_RBF)

                            Lifted_X_Preds = [Lifted_X_Preds_RBF]
                            Lifted_X_Preds.append(mse_RBF)
                            Lifted_X_Preds.append(rmse_RBF)

                            # TRADING
                            trade_pnl_sh = TradePreds(Lifted_X_Preds[0], X_test)

                            modelID = str(kernelIn)+'_'+str(degreeIn)+'_'+str(neighborsIn)+'_'+str(epsilonIn)+'_'+str(liftMethod)
                            statsList.append([str(combo), modelID, Lifted_X_Preds[1], Lifted_X_Preds[2], trade_pnl_sh]) #, pnl_sh
        else:
            for knn in [20, 50, 100]:
                # TRADING
                trade_pnl_sh = TradePreds(Lifted_X_Preds[0], X_test)

                Lifted_X_Preds = Lift(liftMethod, X_train, X_test, target_mapping, target_mapping_Preds, knn)

                modelID = str(knn) + '_' + str(liftMethod)
                statsList.append([str(combo), modelID, Lifted_X_Preds[1], Lifted_X_Preds[2], trade_pnl_sh]) #, pnl_sh

statsDF = pd.DataFrame(statsList, columns=["combo", "modelID", "MSE", "RMSE", "Sharpe"]).set_index("modelID", drop=True)
statsDF["parsimoniousEigs"] = parsimoniousEigs
pickle.dump(statsDF, open(mode.replace(',', '_') + "_" + label + ".p", "wb"))
print(statsDF)
"""
"""
Lifted_Data_List = pickle.load(open("Rolling_run_VAR_EEGdatanew.p", "rb"))
CI_indices = [x for x in Lifted_Data_List[0].iloc[:,0]]
print(CI_indices)
print(Lifted_Data_List[0].columns)

RBF_lifted_List = [[] for varN in range(len(CI_indices))]
GH_lifted_List = [[] for varN in range(len(CI_indices))]
#LP_lifted_List = [[] for varN in range(len(CI_indices))]
for step_data in tqdm(Lifted_Data_List): #[:10]
    subdata = step_data.set_index("roll_target_mapping_Names", drop=True)
    #print(subdata)
    #time.sleep(3000)
    for ci_count in range(len(CI_indices)):
        RBF_lifted_List[ci_count].append(subdata.loc[CI_indices[ci_count], "single_RBF_lifted"])
        GH_lifted_List[ci_count].append(subdata.loc[CI_indices[ci_count], "single_GH_lifted"])
        #LP_lifted_List[ci_count].append(subdata.loc[CI_indices[ci_count],"single_GH_lifted"])

RBF_lifted_Stats_List = []
GH_lifted_Stats_List = []
for j in tqdm(range(len(RBF_lifted_List))):
    Lifted_RBF_Preds = pd.DataFrame(RBF_lifted_List[j]).values
    Lifted_GH_Preds = pd.DataFrame(GH_lifted_List[j]).values

    #print("X_test.shape = ", X_test.shape)
    #print("Lifted_RBF_Preds.shape = ", Lifted_RBF_Preds.shape)
    #print("Lifted_GH_Preds.shape = ", Lifted_GH_Preds.shape)

    mse_RBF = mse_psnr(Lifted_RBF_Preds, X_test)[0]
    rmse_RBF = np.sqrt(mse_RBF)
    mse_GH = mse_psnr(Lifted_GH_Preds, X_test)[0]
    rmse_GH = np.sqrt(mse_GH)

    RBF_lifted_Stats_List.append([j, mse_RBF, rmse_RBF])
    GH_lifted_Stats_List.append([j, mse_GH, rmse_GH])

RBF_lifted_Stats_df = pd.DataFrame(RBF_lifted_Stats_List, index=CI_indices, columns=["j", "MSE", "RMSE"])
GH_lifted_Stats_df = pd.DataFrame(GH_lifted_Stats_List, index=CI_indices, columns=["j", "MSE", "RMSE"])

RBF_lifted_Stats_df.to_excel(mode.replace(',', '_') + "_RBF_lifted_Stats_df_" + label + ".xlsx")
GH_lifted_Stats_df.to_excel(mode.replace(',', '_') + "_GH_lifted_Stats_df_" + label + ".xlsx")

outList = [RBF_lifted_Stats_df, GH_lifted_Stats_df]

pickle.dump(outList, open(mode.replace(',', '_') + "_" + label + ".p", "wb"))
#print(outList)
"""
