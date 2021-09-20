from Slider import Slider as sl
import scipy.io, glob, os
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

"Simple function to test sth"
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def Test():
    pass

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

def Embed(method, X_train_local, target_intrinsic_dim, **kwargs):
    """
    Function to embed input data using a specific embedding method

    :param method: DM, LLE or PCA
    :param X_train_local: X to embed
    :param target_intrinsic_dim: how many parsimonious coordinates
    :param kwargs: LLE_neighbors, dm_epsilon : either a specific value, or if zero it is internally optimized
    :return [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, eigValsOut]:
    target_mapping --> the mapped data
    parsimoniousEigs (str) --> the indexes of the parsimonious coordinates
    X_pcm.kernel.epsilon --> optimized epsilon value
    eigValsOut --> corresponding eigenvalues
    """
    if "LLE_neighbors" in kwargs:
        LLE_neighbors = kwargs["LLE_neighbors"]
    else:
        LLE_neighbors = 50

    if "dm_optParams_knn" in kwargs:
        dm_optParams_knn = kwargs["dm_optParams_knn"]
    else:
        dm_optParams_knn = 50 #previously default was 25 (the one in datafold module as well)

    if "dm_epsilon" in kwargs:
        dm_epsilon = kwargs["dm_epsilon"]
    else:
        dm_epsilon = "opt"

    if "cut_off" in kwargs:
        cut_off = kwargs["cut_off"]
    else:
        cut_off = "opt"

    if "DM" in method:
        X_pcm = pfold.PCManifold(X_train_local)
        X_pcm.optimize_parameters(random_state=random_state, k=dm_optParams_knn)
        if dm_epsilon == "opt":
            dm_epsilon = X_pcm.kernel.epsilon
        if cut_off == "opt":
            cut_off = X_pcm.kernel.cut_off

        if "ComputeParsimonious" in method:
            if target_intrinsic_dim >= 10:
                n_eigenpairsIn = target_intrinsic_dim+1
            else:
                n_eigenpairsIn = 10
        else:
            n_eigenpairsIn = target_intrinsic_dim

        dmap_local = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=dm_epsilon),
            n_eigenpairs=n_eigenpairsIn,
            dist_kwargs=dict(cut_off=cut_off))
        dmap_local = dmap_local.fit(X_pcm)
        # evecs_raw, evals_raw = dmap.eigenvectors_, dmap.eigenvalues_

        if "ComputeParsimonious" in method:
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
            eigValsOut = dmap_local.eigenvalues_[selection.evec_indices_]
        else:
            parsimoniousEigs = "first"
            target_mapping = dmap_local.eigenvectors_
            eigValsOut = dmap_local.eigenvalues_

        out = [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, eigValsOut]
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

def get_ML_Predictions(mode, MLmethod, predictorsData, y_shifted, forecastHorizon):
    """
    Function to perform prediction with ML methods : GPR or ANN, in Multi-step prediction schemes (static)

    :param mode:
        'Main' : identifying on the spot how many regressors objects are to be created and looping on this list of regressors (objects) in order to get the predictions
        'Hardcoded' : cross-checking-validating functionality of the above 'looping' rationale
    :param MLmethod: e.g. 'GPR_Single,3', or ANN_Single,3
        GPR_Single : creates a GPR for EACH variable in the input set and reframes them into instances up to lag (3)
        fixed kernel is used here
    :param predictorsData: input data
    :param y_shifted: IF inserted, those values are used as additional 'exogenous' variables for predictions (not in our case)
    :param forecastHorizon: how many steps to predict
    :return: Preds_List --> the output Predictions
    """
    MLmethodSplit = MLmethod.split(",")
    if MLmethodSplit[1] == "GPR_Single":
        mainKernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()  # Official (29/8/2021)
        model_List = [
            GaussianProcessRegressor(kernel=mainKernel, alpha=0.01, n_restarts_optimizer=2, random_state=random_state)
            for var in range(predictorsData.shape[1])]
    elif MLmethod[1] == "ANN_Single":
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
                    roll_reframedData = reframeData(predictorsData, int(MLmethodSplit[2]), modelIn)
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

def PaperizePreds(X_Preds, X_test, **kwargs):  # CI_Lower_Band, CI_Upper_Band
    """
    Function to 'pretify' output results for overleaf

    :param X_Preds: Predictions matrix
    :param X_test: corresponding Test Set Points
    :param kwargs:
        outputFormat : either RMSE or Sharpe
        roundingFlag : round or not (3 digits)
    :return: the results as 'list' to be consumed by other functions, or as text (str) to be pasted in overleaf
    """
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
    for col in X_test_df.columns:
        if outputFormat == "RMSE":
            submetric = np.round(np.sqrt((1 / X_test_df.shape[0]) * np.sum((X_Preds_df[col].values - X_test_df[col].values) ** 2)),4)  # SIETTOS
        elif outputFormat == "Sharpe":
            sub_pnl = sl.sign(X_Preds_df[col]) * X_test_df[col]
            submetric = np.sqrt(252) * sl.sharpe(sub_pnl)
        elif outputFormat == "RollingSharpe":
            sub_pnl = sl.sign(X_Preds_df[col]) * X_test_df[col]
            rollSharpe = np.sqrt(252) * sl.rollStatistics(sub_pnl, 'Sharpe', nIn=250).dropna()
            rollSharpeStats = [str(np.round(x, 3)) for x in [rollSharpe.mean(), rollSharpe.min(), rollSharpe.max()]]
            #rollSharpe_CIs = mean_confidence_interval(rollSharpe.values, confidence=0.95)
            #rollSharpeStats = [str(np.round(x, 3)) for x in [rollSharpe.mean(), rollSharpe_CIs[1], rollSharpe_CIs[2]]]
            submetric = rollSharpeStats[0] + '(' + rollSharpeStats[1] + "," + rollSharpeStats[2] + ")"
        outList.append(submetric)

    if outputFormat == "Sharpe":
        totalPnl = sl.rs(sl.sign(X_Preds_df) * X_test_df)
        totalSharpe = np.sqrt(252) * sl.sharpe(totalPnl)
        outList.append(totalSharpe)
    elif outputFormat == "RollingSharpe":
        totalPnl = sl.rs(sl.sign(X_Preds_df) * X_test_df)
        total_rollSharpe = np.sqrt(252) * sl.rollStatistics(totalPnl, 'Sharpe', nIn=250).dropna()
        total_rollSharpeStats = [str(np.round(x, 3)) for x in [total_rollSharpe.mean(), total_rollSharpe.min(), total_rollSharpe.max()]]
        #total_rollSharpe_CIs = mean_confidence_interval(total_rollSharpe.values, confidence=0.95)
        #total_rollSharpeStats = [str(np.round(x, 3)) for x in [total_rollSharpe.mean(), total_rollSharpe_CIs[1], total_rollSharpe_CIs[2]]]
        total_submetric = total_rollSharpeStats[0] + '(' + total_rollSharpeStats[1] + "," + total_rollSharpeStats[2] + ")"
        outList.append(total_submetric)
        roundingFlag = False

    if roundingFlag == True:
        outList = [np.round(x, 3) for x in outList]

    if 'returnData' in kwargs:
        return outList
    else:
        return ' & '.join([str(x) for x in outList])

random_state = 0
modelParamsList = []

def Lift(method, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation, **kwargs):
    """
     Function to perform lifting

     :param method: available methods are
         'GH' : Geometric Harmonics
         'LP' : Laplacial Pyramids
         'KR' : Kriging (GPRs)
         'SI' : Simple knn interpolation
         'RBF' : Radial Basis Functions interpolation
     :param X_trainingSet: input high-dimensional space data (X), training set
     :param X_testSet: high-dimensional space data (X), test set
     :param eig_trainingSet: low-dimensional (embedded) space parsimonious eigenvectors (Y), training set
     :param eig_Simulation: low-dimensional (embedded) space parsimonious eigenvectors (Y), predicted (by a specific forecasting methodogy, e.g. VAR(3)) set
     :param knn: input neighbors used for the lifting
     :return: [extrapolatedPsi_to_X, mse, rmse, residual]
         extrapolatedPsi_to_X : lifted data
         mse : mean-squared error between the [extrapolatedPsi_to_X and X_testSet]
         rmse : corresponding roor-mean-squared-error
         residual : residuals of the lifting process
     """

    if "lift_optParams_knn" in kwargs:
        lift_optParams_knn = kwargs["lift_optParams_knn"]
    else:
        lift_optParams_knn = 50 #previously default was 25 (the one in datafold module as well)

    if "GH_epsilon" in kwargs:
        GH_epsilon = kwargs["GH_epsilon"]
    else:
        GH_epsilon = "opt"

    if "GH_cut_off" in kwargs:
        GH_cut_off = kwargs["GH_cut_off"]
    else:
        GH_cut_off = "opt"

    def mse_psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return [mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))]

    if method == 'GH':
        pcm = pfold.PCManifold(eig_trainingSet)
        pcm.optimize_parameters(random_state=random_state, k=lift_optParams_knn)
        if GH_epsilon == "opt":
            GH_epsilon = pcm.kernel.epsilon
        if GH_cut_off == "opt":
            GH_cut_off = pcm.cut_off
        opt_n_eigenpairs = eig_trainingSet.shape[1]
        gh_interpolant_psi_to_X = GHI(pfold.GaussianKernel(epsilon=GH_epsilon),
                                      n_eigenpairs=opt_n_eigenpairs, dist_kwargs=dict(cut_off=GH_cut_off))
        gh_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = gh_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gh_interpolant_psi_to_X.predict(eig_Simulation)
        # print("opt_epsilon = ", opt_epsilon)
        # print("opt_cutoff = ", opt_cutoff)
        modelParamsList.append([lift_optParams_knn, GH_cut_off, GH_epsilon])

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
        extrapolatedPsi_to_X = RBFInterpolator(eig_trainingSet, X_trainingSet, kernel="linear", degree=1, neighbors=lift_optParams_knn, epsilon=1)(eig_Simulation)
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

def RollingRunProcess(params):
    """
    Function to perform a rolling window prediction scheme
        Either Full model analysis in the [Original Space (OS)]
        or the 'embed-predict-lift' prediction framework, [Embedded space (ES)]
    :param params: dict
        Example :
        paramsProcess = {
            'data': data, ---> entire dataset
            'data_embed': data_embed, --> data used for embedding, either the OS dataset or the ones used for embedding in the case of 'extended'-'Takens' like analysis
            'X_test': X_test, --> test set
            'label': label, --> dataset label
            'simulationNumber': simulationNumber, --> the number of a specific simulation (for the EEG multi-simulations-datasets cases) - forex has 1 dataset
            'liftMethod': 'FullModel', --> 'FullModel' in the case of OS prediction, or 'GH'-'RBF'-'KR' etc as per the Lift() function lifting process
            'trainSetLength': trainSetLength, ---> index (position) where the training set ends
            'rolling_Embed_Memory': data.shape[0] - forecastHorizon, --> #points used for embedding (represented as the difference of the entire dataset's length and the target number of forecasts = 'forecastHorizon'
            'target_intrinsic_dim': target_intrinsic_dim, --> #parsimonious coordinates
            'mode': mode, --> what model and framework is being processed, Example : "FullModel_Rolling,VAR,1" for a VAR(1) prediction framework in the OS or "Rolling_run,VAR,1" for the VAR(1) prediction framework in the embed-predict-lift case where embed and lift are defined in the other parameters of the dict object as well
            'rolling_Predict_Memory': data.shape[0] - forecastHorizon, ---> #points used for the prediction model training. could be the same as the 'rolling_Embed_Memory', or different for speedier (but less memory given) results
            'embedMethod': embedMethod, --> embedding method used. "LLE" for Local-Linear-Embedding. "DM" for Diffusion Maps with NO parsimonious selection as per the linear-fit approach. "DMComputeParsimonious" where the parsimonious eigs are identified at EACH time step
            'kernelIn': np.nan, 'degreeIn': np.nan, 'neighborsIn': np.nan, 'epsilonIn': np.nan --> parameters for the several embed/lifting approaches
        }

    :return: Pickle object written in the target folder, containing the predictions
    [Preds = np.array(PredsList)]
    """
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
    dm_epsilonIn = params["dm_epsilonIn"]
    cut_offIn = params["cut_offIn"]
    dm_optParams_knnIn = params["dm_optParams_knnIn"]
    LLE_neighborsIn = params["LLE_neighborsIn"]
    lift_optParams_knnIn = params["lift_optParams_knn"]
    GH_epsilonIn = params["GH_epsilon"]
    GH_cut_offIn = params["GH_cut_off"]

    processName = '_'.join([str(x) for x in list(params.values())[3:]])

    writeEmbeddingFile = False
    embeddingFileExists = False

    "Check if embedding File already exists and use this one instead of recalculating!"
    checkEmbeddingFilePath = RollingRunnersPath + "Embedding\\" + processName.replace(mode + "_", "") + "_EMBEDDINGFILE.p"
    if os.path.exists(checkEmbeddingFilePath):
        print("Embedding File Exists! : " + checkEmbeddingFilePath)
        Stored_Embedding_Data = pickle.load(open(checkEmbeddingFilePath, "rb"))
        #print("len(Stored_Embedding_Data) = ", len(Stored_Embedding_Data))
        #print("len(range(trainSetLength, data.shape[0], 1)) = ", len(range(trainSetLength, data.shape[0], 1)))
        #time.sleep(3000)
        storedCount = 0
        embeddingFileExists = True
    else:
        print("No related Embedding File exists. Calculating embedding as well .... ")

    PredsList = []
    embeddingList = []
    for i in tqdm(range(trainSetLength, data.shape[0], 1)): #
        #print(i)
        try:
            if liftMethod == 'FullModel':
                roll_X_test = data[i]
                roll_target_mapping = data[i - rolling_Embed_Memory:i]
                modelListSpectrum = roll_target_mapping.shape[1]
            else:
                roll_X_train_embed = data_embed[i - rolling_Embed_Memory:i]
                roll_X_train_lift = data[i - rolling_Embed_Memory:i]
                roll_X_test = data[i]

                if embeddingFileExists == True:
                    roll_target_mapping = Stored_Embedding_Data[storedCount][0]
                    modelListSpectrum = roll_target_mapping.shape[1]
                    #print("storedCount = ", storedCount, ", type(roll_target_mapping) = ", type(roll_target_mapping), ", roll_target_mapping.shape = ", roll_target_mapping.shape)
                    #time.sleep(3000)
                    storedCount += 1
                else:
                    try:
                        roll_target_mapping_List = Embed(embedMethod, roll_X_train_embed, target_intrinsic_dim, LLE_neighbors=LLE_neighborsIn, dm_epsilon=dm_epsilonIn, cut_off=cut_offIn, dm_optParams_knn=dm_optParams_knnIn)
                        roll_target_mapping = roll_target_mapping_List[0]
                        embeddingList.append(roll_target_mapping_List)
                    except Exception as e:
                        print("Embedding error ... ")
                        print(e)
                        roll_target_mapping = embeddingList[-1][0]
                        embeddingList.append(embeddingList[-1])

                    modelListSpectrum = roll_target_mapping.shape[1]
                    writeEmbeddingFile = True

            ###################################### PREDICT #######################################
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
                #print("roll_target_mapping[-rolling_Predict_Memory:].shape = ", roll_target_mapping[-rolling_Predict_Memory:].shape)
                roll_forecasting_model = VAR(roll_target_mapping[-rolling_Predict_Memory:])
                roll_model_fit = roll_forecasting_model.fit(int(modeSplit[2]))
                roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=1, alpha=0.05)
                row_Preds = roll_target_mapping_Preds_All[0]
                #print("row_Preds.shape = ", row_Preds.shape)
            elif modeSplit[1].strip() == 'GPR_Single':
                if i == trainSetLength:
                    mainRolling_kernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel()
                    model_GPR_List = [GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0) for var in range(modelListSpectrum)]

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
                single_lifted_Preds = Lift("GH", roll_X_train_lift, roll_X_test, roll_target_mapping, row_Preds, lift_optParams_knn=lift_optParams_knnIn, GH_epsilon=GH_epsilonIn, GH_cut_off=GH_cut_offIn)[0]
                PredsList.append(single_lifted_Preds)
        except Exception as e:
            print(e)
            PredsList.append([0 for var in range(len(PredsList[-1]))])

    Preds = np.array(PredsList)
    #sl.cs(pd.DataFrame(Preds)).plot()
    #plt.show()
    print("Preds.shape = ", Preds.shape)
    pickle.dump(Preds, open(RollingRunnersPath + processName + ".p", "wb"))
    if writeEmbeddingFile == True:
        try:
            pickle.dump(embeddingList, open(RollingRunnersPath + "Embedding\\" + processName.replace(mode+"_", "") + "_EMBEDDINGFILE.p", "wb"))
        except Exception as e:
            pickle.dump(embeddingList, open(RollingRunnersPath + processName.replace(mode + "_", "") + "_EMBEDDINGFILE.p", "wb"))

def RunPythonDM(paramList):
    label = paramList[0]
    mode = paramList[1]
    simulationNumber = paramList[2]
    datasetsPath = paramList[3]
    RollingRunnersPath = paramList[4]
    TakensSpace = paramList[6]
    embedMethod = paramList[5] + "_" + TakensSpace
    target_intrinsic_dim = paramList[7]
    forecastHorizon = paramList[8]
    rolling_Predict_Memory = paramList[9]
    LLE_neighborsIn = paramList[10]
    dm_epsilonIn = paramList[11]
    cut_offIn = paramList[12]
    dm_optParams_knnIn = paramList[13]
    lift_optParams_knnIn = paramList[14]
    GH_epsilonIn = paramList[15]
    GH_cut_offIn = paramList[16]

    "Read the mat file"
    matFileName = datasetsPath + label + "_" + str(simulationNumber) + '.mat'
    mat = scipy.io.loadmat(matFileName)

    modeSplit = mode.split(',')

    "Prepare the data per case : delay (extended) cases or not (straight-forward) ones"
    if "Delay" in label:
        data = mat['Ytrain1']
    else:
        data = mat['Ytrain']

    if TakensSpace == "extended":
        data_embed = mat['Ytrain']
    elif TakensSpace == "TakensDynFold":
        data_embed = TSCTakensEmbedding(data)
        print("TakensDynFold ... ", " data.shape = ", data.shape)
        print("data_embed.shape = ", data_embed.shape)
        time.sleep(3000)
    else:
        data_embed = data

    trainSetLength = data.shape[0] - forecastHorizon
    X_train = data[:trainSetLength]
    X_test = data[trainSetLength:]
    X_test_shifted = sl.S(pd.DataFrame(X_test)).values
    X_test_shifted[0] = X_train[-1]
    X_train_predict = X_train
    trainEmbedSetLength = data_embed.shape[0] - forecastHorizon
    X_train_embed = data_embed[:trainEmbedSetLength]

    if "Delay" in label:
        X_train_lift = mat['YtrainLift'][:trainEmbedSetLength]
    else:
        X_train_lift = mat['Ytrain'][:trainEmbedSetLength]

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
            Preds_List = get_ML_Predictions("Main", mode, X_train_predict, X_test_shifted, forecastHorizon)
            Preds = pd.DataFrame(Preds_List).values

        try:
            FullModel_Static_PaperText = PaperizePreds(Preds, X_test)
            print("FullModel_Static_PaperText = ", FullModel_Static_PaperText)
        except Exception as e:
            print(e)
        pickle.dump(Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + embedMethod + ".p","wb"))
    elif modeSplit[0] == "FullModel_Rolling":

        paramsProcess = {
            'data': data,
            'data_embed': data_embed,
            'X_test': X_test,
            'label': label,
            'simulationNumber': simulationNumber,
            'liftMethod': 'FullModel',
            'trainSetLength': trainSetLength,
            'rolling_Embed_Memory': data.shape[0] - forecastHorizon,
            'target_intrinsic_dim': target_intrinsic_dim, 'mode': mode,
            'rolling_Predict_Memory': data.shape[0] - forecastHorizon,
            'embedMethod': embedMethod,
            'LLE_neighborsIn': LLE_neighborsIn,
            'dm_epsilonIn': dm_epsilonIn,
            'cut_offIn': cut_offIn,
            'dm_optParams_knnIn': dm_optParams_knnIn,
            'lift_optParams_knn': lift_optParams_knnIn,
            'GH_epsilon': GH_epsilonIn,
            'GH_cut_off': GH_cut_offIn
        }

        RollingRunProcess(paramsProcess)
    ###################################################################################################################
    elif modeSplit[0] == 'Static_run':

        ###################################### REDUCED TARGET SPACE #######################################
        "Check if exists already"
        allProcessedAlready = [f for f in glob.glob(RollingRunnersPath + '*.p')]

        "Embed"
        target_mapping_List = Embed(embedMethod, X_train_embed, target_intrinsic_dim, LLE_neighbors=LLE_neighborsIn, dm_epsilon=dm_epsilonIn, cut_off=cut_offIn, dm_optParams_knn=dm_optParams_knnIn)
        target_mapping = target_mapping_List[0]
        parsimoniousEigs = target_mapping_List[1]
        target_mapping_EigVals = target_mapping_List[3]

        "Forecast and if needed get CIs for the embedded space"
        if modeSplit[1] == "VAR":
            forecasting_model = VAR(target_mapping)
            model_fit = forecasting_model.fit(int(modeSplit[2]))
            target_mapping_Preds_All = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
            mapped_Preds = target_mapping_Preds_All[0]
        elif modeSplit[1] in ["GPRSingle", "ANN_Single"]:
            Preds_List = get_ML_Predictions("Main", mode, target_mapping, [], forecastHorizon)
            mapped_Preds = pd.DataFrame(Preds_List).values

        "Lift the embedded space predictions to the Original Space"
        for liftMethod in ["GH", "SI", "RBF"]:  #"KR"
            lift_outputFile = RollingRunnersPath + label + "_" + str(simulationNumber) + '_' + mode + '_' + liftMethod + '_' + '_'.join([str(x) for x in paramList[5:]]) + ".p"
            if lift_outputFile not in allProcessedAlready:
                try:
                    Lifted_X_Preds = Lift(liftMethod, X_train_lift, X_test, target_mapping, mapped_Preds, lift_optParams_knn=lift_optParams_knnIn, GH_epsilon=GH_epsilonIn, GH_cut_off=GH_cut_offIn)[0]
                except Exception as e:
                    print(str(simulationNumber) + "_" + mode.replace(",", "_") + "_" + embedMethod + "_" + liftMethod)
                    print(e)

                pickle.dump([Lifted_X_Preds, mapped_Preds, parsimoniousEigs], open(lift_outputFile, "wb"))
            else:
                print(lift_outputFile, " already exists!")
    ###################################################################################################################
    elif modeSplit[0] == 'Rolling_run':

        paramsProcess = {
            'data': data,
            'data_embed': data_embed,
            'X_test': X_test,
            'label': label,
            'simulationNumber': simulationNumber,
            'liftMethod': "GH",
            'trainSetLength': trainSetLength,
            'rolling_Embed_Memory': data.shape[0] - forecastHorizon,
            'target_intrinsic_dim': target_intrinsic_dim,
            'mode': mode,
            'rolling_Predict_Memory': rolling_Predict_Memory,
            'embedMethod': embedMethod,
            'LLE_neighborsIn': LLE_neighborsIn,
            'dm_epsilonIn': dm_epsilonIn,
            'cut_offIn': cut_offIn,
            'dm_optParams_knnIn': dm_optParams_knnIn,
            'lift_optParams_knn': lift_optParams_knnIn,
            'GH_epsilon': GH_epsilonIn,
            'GH_cut_off': GH_cut_offIn
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
    """
    Function to get the reported results in Excel Files

    :param mode:
        RunRandomWalks --> report for the random walks for the datasets
        Run --> report for the predictions schemes applied in the datasets (raw)
        Read --> aggregated report for the predictions schemes applied in the datasets --> especially applied on the EEG datasets, where median and percentiles are reported
    :param datasetsPath: folder where the 'raw' datasets are stored
    :param RollingRunnersPath: folder where the output predictions Pickles to be processed, are stored
    :param writeResiduals: whether to write the prediction residuals or not (for e.g. matlab to consume and plot)
    :param target_intrinsic_dim: #parsimonious used
    :param kwargs:
        reportPercentilesFlag : True or False
        outputFormatReporter : RMSE or Sharpe
        RiskParityFlag : whether to perform risk parity portfolio construction or not (e.g. yes,250) means that 250 trading days are used for the rolling volatilities calculation
    :return: excel files written in the 'RollingRunnersPath' directory with the output results
        Example:
        'Reporter_dataDF_RW_raw_Yes,250' : this excel corresponds to the Random Walks
        'Reporter_dataDF_raw_Yes,250' : this excel corresponds to the several models predictions' results
        'Reporter_Yes,250' : the aggregated performance, i.e. median and percentiles in the case of EEG Datasets
    """
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

    RiskParityFlagSplit = RiskParityFlag.split(",")

    if mode == "RunRandomWalks":

        allDataListRW = []
        for dataset_total_name in glob.glob(datasetsPath + '*.mat'):
            mat_dataset = scipy.io.loadmat(dataset_total_name)
            # dataRW = mat_dataset['y']
            if "Delay" in dataset_total_name:
                dataRW = mat_dataset['Ytrain1']
            else:
                dataRW = mat_dataset['Ytrain']

            if RiskParityFlagSplit[0] == "Yes":
                dataRW_df = pd.DataFrame(dataRW)
                riskParityVol = np.sqrt(252) * sl.S(sl.rollerVol(dataRW_df, int(RiskParityFlagSplit[1]))) * 100 # 250
                dataRW_df = (dataRW_df / riskParityVol).replace([np.inf, -np.inf], 0)
                dataRW = dataRW_df.values

            trainSetLengthRW = dataRW.shape[0] - forecastHorizon
            X_testRW = dataRW[trainSetLengthRW:]

            PredsRW = sl.S(pd.DataFrame(X_testRW)).fillna(0).values
            metricsVarsRW = PaperizePreds(PredsRW, X_testRW, returnData='yes', outputFormat=outputFormatReporter)
            allDataListRW.append([dataset_total_name.split("\\")[-1] + "_" + "-" + "_" + "-" + "_" + "-" + "_" + "-", metricsVarsRW])

        dataDFRW = pd.DataFrame(allDataListRW, columns=["file_name", "metricsVars"]).set_index("file_name", drop=True)
        dataDFRW.to_excel(RollingRunnersPath+"Reporter_dataDF_RW_raw_"+RiskParityFlag+".xlsx")
    elif mode == "Run":

        allDataList = []
        for file_total_name in tqdm(glob.glob(RollingRunnersPath + '\\*.p')):
            try:
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

                if RiskParityFlagSplit[0] == "Yes":
                    data_df = pd.DataFrame(data)
                    riskParityVol = np.sqrt(252) * sl.S(sl.rollerVol(data_df, int(RiskParityFlagSplit[1]))) * 100
                    data_df = (data_df / riskParityVol).replace([np.nan, np.inf, -np.inf], 0)
                    data = data_df.values

                PredsData = pickle.load(open(file_total_name, "rb"))
                if "Single_AR_1" in file_total_name:
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
                #print("file_name = ", file_name, ", trainSetLength = ", trainSetLength)
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
            except Exception as e:
                print(e)
                print(file_total_name)

        dataDF = pd.DataFrame(allDataList,
                              columns=["file_name", "metricsVars", "parsimoniousEigs", "target_intrinsic_dim",
                                       "Preds.shape[0]"]).set_index("file_name", drop=True)
        dataDF.to_excel(RollingRunnersPath+"Reporter_dataDF_raw_"+RiskParityFlag+".xlsx")
    elif mode == "Read":

        dataDF = pd.concat([pd.read_excel(RollingRunnersPath+"Reporter_dataDF_RW_raw_"+RiskParityFlag+".xlsx"), pd.read_excel(RollingRunnersPath+"Reporter_dataDF_raw_"+RiskParityFlag+".xlsx")])
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
        ReportDF.to_excel(RollingRunnersPath+"Reporter_"+RiskParityFlag+".xlsx")

def ReportProcessingToOverleaf(RiskParityFlagIn):
    df = pd.read_excel(RollingRunnersPath + "Reporter_dataDF_raw_" + RiskParityFlagIn + ".xlsx")
    df = df[["file_name", "metricsVars"]]
    df["file_name_split"] = df["file_name"].str.split("_")
    df["fullOrEmbed"] = df["file_name_split"].str[2]

    df = df[df["fullOrEmbed"]!="FullModel"]
    #df = df[df["fullOrEmbed"]=="FullModel"]

    df["embedMethod"] = df["file_name_split"].str[9]
    for embedMethod in set(list(df["embedMethod"].values)):
        print(embedMethod)
        subdf = df[df["embedMethod"]==embedMethod]
        subdf["TableID"] = subdf["file_name_split"].str[3] + "," + df["file_name_split"].str[8]
        subdf["TotalSharpe"] = subdf["metricsVars"].str.split(",").str[-1].str.replace("]", "") + ", " + subdf["file_name_split"].str[7]
        subdf = subdf[["TableID", "TotalSharpe"]].groupby(['TableID']).max()
        subdf.to_excel(RollingRunnersPath+embedMethod+"_ReportProcessingToOverleaf.xlsx")

if __name__ == '__main__':

    label = "FxDataAdjRetsMAJORSDelay"  # EEGsynthetic2, EEGsynthetic2nonlin, EEGsynthetic2nonlinDelay, FxDataAdjRetsMAJORSDelay
    embedMethod = "DMComputeParsimonious"  # LLE, DMComputeParsimonious, DM

    pcFolderRoot = 'D:\Dropbox\VM_Backup\\'
    #pcFolderRoot = 'E:\Dropbox\Dropbox\VM_Backup\\'

    if label == "EEGsynthetic2":
        target_intrinsic_dim = 2 #2,3
        simulNum = 100
        TakensSpace = "NoTakens"
        reportPercentilesFlagIn = True
        forecastHorizon = 500
        Predict_Memory = 2000-forecastHorizon
        outputFormat = "RMSE"
        RiskParityFlagIn = "No"
        ######################
        LLE_neighborsIn = 50
        ######################
        dm_epsilonIn = "opt"
        cut_offIn = np.inf
        dm_optParams_knnIn = 50
        ######################
        GH_epsilonIn = "opt"
        GH_cut_offIn = "opt"
        lift_optParams_knnIn = 50
        ######################
        rollingSpecs = [[], [], 1] #1,

        runProcessesFlag = "MultipleProcesses"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\StaticRunners\\'
    elif label == "EEGsynthetic2nonlin":
        target_intrinsic_dim = 3 #2,3
        simulNum = 100
        TakensSpace = "NoTakens"
        reportPercentilesFlagIn = True
        forecastHorizon = 500
        Predict_Memory = 2000 - forecastHorizon
        outputFormat = "RMSE"
        RiskParityFlagIn = "No"
        ######################
        LLE_neighborsIn = 50
        ######################
        dm_epsilonIn = "opt"
        cut_offIn = np.inf
        dm_optParams_knnIn = 50
        ######################
        GH_epsilonIn = "opt"
        GH_cut_offIn = "opt"
        lift_optParams_knnIn = 50
        ######################
        rollingSpecs = [[], [], 1]

        runProcessesFlag = "MultipleProcesses"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\StaticRunners\\'
    elif label == "EEGsynthetic2nonlinDelay":
        TakensSpace = "NoTakens"

        if TakensSpace == "NoTakens":
            target_intrinsic_dim = 3
        elif TakensSpace == "yes,extended":
            target_intrinsic_dim = 3 # 3,4

        simulNum = 1000
        reportPercentilesFlagIn = True
        forecastHorizon = 500
        Predict_Memory = 2000 - forecastHorizon
        outputFormat = "RMSE"
        RiskParityFlagIn = "No"
        ######################
        LLE_neighborsIn = 50
        ######################
        dm_epsilonIn = "opt"
        cut_offIn = np.inf
        dm_optParams_knnIn = 50
        ######################
        GH_epsilonIn = "opt"
        GH_cut_offIn = "opt"
        lift_optParams_knnIn = 50
        ######################
        rollingSpecs = [[], [], 3]

        runProcessesFlag = "MultipleProcesses"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_DelayApproach\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\StaticRunners_Delay\\'
    elif label == "FxDataAdjRetsMAJORSDelay":
        TakensSpace = "NoTakens"
        #TakensSpace = "extended"
        #TakensSpace = "TakensDynFold"

        target_intrinsic_dim = 3 # 2, 3
        reportPercentilesFlagIn = False
        outputFormat = "Sharpe"
        #outputFormat = "RollingSharpe"
        RiskParityFlagIn = "Yes,250" # "Yes,250", "No"
        #RiskParityFlagIn = "No" # "Yes,250", "No"

        rollingSpecs = [1000, 1000, 10] # 20, 100, 250, 500, 1000

        forecastHorizon = 5002 - rollingSpecs[0]
        Predict_Memory = rollingSpecs[1]
        ######################
        LLE_neighborsIn = 50 #5
        ######################
        dm_epsilonIn = "opt"
        cut_offIn = np.inf
        dm_optParams_knnIn = 50 #5
        ######################
        GH_epsilonIn = "opt"
        GH_cut_offIn = "opt"
        lift_optParams_knnIn = 50 #5, 25, 50

        runProcessesFlag = "SingleProcess"

        datasetsPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_FOREX\\'
        RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners_FOREX\\'
        #RollingRunnersPath = pcFolderRoot + 'RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners_FOREX\\temp\\'

    ##### STATIC ####
    #processToRun = "FullModel_Static,Single_AR,"+str(rollingSpecs[2])
    #processToRun = "FullModel_Static,VAR,"+str(rollingSpecs[2])
    #processToRun = "FullModel_Static,GPRSingle,"+str(rollingSpecs[2])
    #processToRun = "FullModel_Static,ANN_Single"+str(rollingSpecs[2])
    #processToRun = "Static_run,VAR,"+str(rollingSpecs[2])
    #processToRun = "Static_run,GPRSingle,"+str(rollingSpecs[2])
    #processToRun = "Static_run,ANN_Single"

    ##### ROLLING ####
    #processToRun = "FullModel_Rolling,Single_AR,"+str(rollingSpecs[2])
    #processToRun = "FullModel_Rolling,VAR,"+str(rollingSpecs[2])
    #processToRun = "FullModel_Rolling,GPRSingle,"+str(rollingSpecs[2])
    processToRun = "Rolling_run,VAR,"+str(rollingSpecs[2])
    #processToRun = "Rolling_run,GPRSingle,"+str(rollingSpecs[2])

    #runProcessesFlag = "Report"
    runProcessesFlag = "ReportProcessingToOverleaf"
    writeResiduals = 0

    if runProcessesFlag == "SingleProcess":
        "Run Single Process (for the FOREX datasets)"
        RunPythonDM(
            [label, processToRun, 0, datasetsPath, RollingRunnersPath, embedMethod, TakensSpace, target_intrinsic_dim,
             forecastHorizon, Predict_Memory, LLE_neighborsIn, dm_epsilonIn, cut_offIn, dm_optParams_knnIn, lift_optParams_knnIn, GH_epsilonIn, GH_cut_offIn])
    elif runProcessesFlag == "MultipleProcesses":

        simulList = []
        for simul in range(simulNum):
            try:
                simulList.append([label, processToRun, simul, datasetsPath, RollingRunnersPath, embedMethod, TakensSpace, target_intrinsic_dim,
             forecastHorizon, Predict_Memory, LLE_neighborsIn, dm_epsilonIn, cut_offIn, dm_optParams_knnIn, lift_optParams_knnIn, GH_epsilonIn, GH_cut_offIn])
            except Exception as e:
                print(simul)
                print(e)
        p = mp.Pool(mp.cpu_count())
        result = p.map(RunPythonDM, tqdm(simulList))
        p.close()
        p.join()
    elif runProcessesFlag == "Report":
        print("REPORTER ......... ")
        " REPORT RESULTS "

        Reporter("RunRandomWalks", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                 reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)
        Reporter("Run", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                 reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)
        if outputFormat != "RollingSharpe":
            Reporter("Read", datasetsPath, RollingRunnersPath, writeResiduals, target_intrinsic_dim,
                     reportPercentilesFlag=reportPercentilesFlagIn, outputFormatReporter=outputFormat, RiskParityFlag=RiskParityFlagIn)
    elif runProcessesFlag == "ReportProcessingToOverleaf":
        ReportProcessingToOverleaf(RiskParityFlagIn)

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
"""
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

def TradePreds(X_Preds, X_test):
    trSig = pd.DataFrame(X_Preds)
    X_test_df = pd.DataFrame(X_test)

    pnl = sl.rs(sl.sign(trSig) * X_test_df)
    pnl_sh = np.sqrt(252) * sl.sharpe(pnl)

    return pnl_sh

"""