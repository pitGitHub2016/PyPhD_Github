from Slider import Slider as sl
import scipy.io, glob
from scipy.interpolate import NearestNDInterpolator
import itertools, math
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import GPy
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
    LocalRegressionSelection,
)
from datafold.dynfold import LocalRegressionSelection

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

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
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def reframeData(dataIn, reframeStep, varSelect):

    baseDF = pd.DataFrame(dataIn)

    df_List = []
    for i in range(reframeStep+1):
        if i == 0:
            subDF_i0 = baseDF.copy()
            subDF_i0.columns = ["base_" + str(x) for x in subDF_i0.columns]
            df_List.append(subDF_i0)
        else:
            subDF = baseDF.shift(i)#.fillna(0)
            subDF.columns = ["delay_"+str(i)+"_"+str(x) for x in subDF.columns]
            df_List.append(subDF)

    df = pd.concat(df_List, axis=1).dropna()

    if varSelect == "all":
        Y_DF = df.loc[:, [x for x in df.columns if "base_" in x]]
    else:
        Y_DF = df.loc[:, "base_"+str(varSelect)]
    X_DF = df.loc[:, [x for x in df.columns if "delay_" in x]]
    lastY_test_point = df.loc[df.index[-1], [x for x in df.columns if "base_" in x]]

    #print(df)
    #print(df.tail(5))
    #print("X_DF = ", X_DF.tail(5))
    #print("Y_DF = ", Y_DF.tail(5))
    #print(lastY_test_point.values)
    #time.sleep(3000)

    X_all_gpr = X_DF.values
    if isinstance(Y_DF, pd.Series) == 1:
        Y_all_gpr = Y_DF.values.reshape(-1,1)
    else:
        Y_all_gpr = Y_DF.values

    lastY_test_point_gpr = lastY_test_point.values.reshape(1,-1)

    return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr]

def Embed(method, X_train_local, target_intrinsic_dim, **kwargs):

    if "LLE_neighbors" in kwargs:
        LLE_neighbors = kwargs["LLE_neighbors"]
    else:
        LLE_neighbors = 50

    if method == "DM":

        X_pcm = pfold.PCManifold(X_train_local)
        X_pcm.optimize_parameters()

        dmap_local = dfold.DiffusionMaps(
            kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon),
            n_eigenpairs=10,
            dist_kwargs=dict(cut_off=X_pcm.cut_off),
        )
        dmap_local = dmap_local.fit(X_pcm)
        #evecs_raw, evals_raw = dmap.eigenvectors_, dmap.eigenvalues_

        selection = LocalRegressionSelection(
            intrinsic_dim=target_intrinsic_dim, n_subsample=500, strategy="dim"
        ).fit(dmap_local.eigenvectors_)

        #print("selection.evec_indices_ = ", selection.evec_indices_)
        parsimoniousEigs = ",".join([str(x) for x in selection.evec_indices_])

        target_mapping = selection.transform(dmap_local.eigenvectors_)
        #print("target_mapping.shape = ", target_mapping.shape)

        out = [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, dmap_local.eigenvalues_[selection.evec_indices_]]
    elif method == "LLE":
        lle = manifold.LocallyLinearEmbedding(n_neighbors=LLE_neighbors, n_components=target_intrinsic_dim, method="standard", n_jobs=-1)
        target_mapping = lle.fit_transform(X_train_local)

        out = [target_mapping, "none", 1, []]
    elif method == "PCA":
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

def PaperizePreds(X_Preds, X_test, **kwargs): # CI_Lower_Band, CI_Upper_Band

    localVarColumns = ["x"+str(x) for x in range(X_Preds.shape[1])]

    X_test_df = pd.DataFrame(X_test, columns=localVarColumns)
    X_Preds_df = pd.DataFrame(X_Preds, columns=localVarColumns)
    #CI_Lower_Band_df = pd.DataFrame(CI_Lower_Band, columns=localVarColumns)
    #CI_Upper_Band_df = pd.DataFrame(CI_Upper_Band, columns=localVarColumns)

    #Preds_Errors_df = X_Preds_df - X_test_df
    #CI_Lower_Errors_df = CI_Lower_Band_df - X_test_df
    #CI_Upper_Errors_df = CI_Upper_Band_df - X_test_df

    outList = []
    """
    for elem in [["ErrorsSpace", Preds_Errors_df.mean(), CI_Lower_Errors_df.mean(), CI_Upper_Errors_df.mean()], ["PredsSpace", X_Preds_df.mean(), CI_Lower_Band_df.mean(), CI_Upper_Band_df.mean()]]:

        subDF = pd.DataFrame(columns=["MEAN", "CI_LOWER", 'CI_UPPER'])

        subDF["MEAN"] = elem[1]
        subDF["CI_LOWER"] = elem[2]
        subDF["CI_UPPER"] = elem[3]

        subDF = subDF.round(4).astype(str)
        subDF["CI_LOWER"] = '('+subDF["CI_LOWER"]
        subDF["CI_UPPER"] = subDF["CI_UPPER"]+')'
        subDF["TEXT"] = subDF['MEAN'].str.cat(subDF['CI_LOWER'],sep="").str.cat(subDF['CI_UPPER'],sep=",")

        textDF = pd.DataFrame(subDF["TEXT"]).T
        textDF.index = [elem[0]]

        outList.append(textDF)
    """
    for col in X_test_df.columns:
        #rmse = np.round(np.sqrt(mse_psnr(X_Preds_df[col].values, X_test_df[col].values)[0]), 4) # MINE
        rmse = np.round(np.sqrt((1/X_test_df.shape[0]) * np.sum((X_Preds_df[col].values - X_test_df[col].values)**2)), 4) # SIETTOS
        outList.append(rmse)

    if 'returnData' in kwargs:
        return outList
    else:
        return ' & '.join([str(x) for x in outList])

def get_ML_Predictions(mode, MLmethod, predictorsData, y_shifted, forecastHorizon):

    if MLmethod == "GPR_Single":
        mainKernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * RBF() + 1 * WhiteKernel() # Official (29/8/2021)
        model_List = [GaussianProcessRegressor(kernel=mainKernel, alpha=0.01, n_restarts_optimizer=2, random_state=random_state) for var in range(predictorsData.shape[1])]
    elif MLmethod == "ANN_Single":
        model_List = []
        for var in range(predictorsData.shape[1]):
            ANN_model = Sequential()
            #ANN_model.add(Dense(2, input_dim=predictorsData.shape[1], activation='sigmoid'))
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
                    roll_reframedData = reframeData(predictorsData, 1, modelIn)
                    model_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1])
                    #print("model_List[modelIn].score = ", model_List[modelIn].score(roll_reframedData[0], roll_reframedData[1]))
                    print("model_List[", modelIn, "].kernel = ", model_List[modelIn].kernel_)
                    sub_row_Preds = model_List[modelIn].predict(roll_reframedData[2])
                else:
                    sub_row_Preds = model_List[modelIn].predict(total_row_subPred.reshape(roll_reframedData[2].shape))

                models_preds_list.append(sub_row_Preds[0][0])

            total_row_subPred = np.array(models_preds_list)
            print("step_i = ", step_i, ", MLmethod = ", MLmethod, ", total_row_subPred = ", total_row_subPred)
            Preds_List.append(total_row_subPred)
    elif mode == "HardCoded":

        mainKernel1 = 1 * RBF() + 1 * WhiteKernel() #"fixed", length_scale_bounds=(0,1)
        mainKernel2 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel3 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel4 = 1 * RBF() + 1 * WhiteKernel()
        mainKernel5 = 1 * RBF() + 1 * WhiteKernel()
        gprModel1 = GaussianProcessRegressor(kernel=mainKernel1, random_state=random_state) #n_restarts_optimizer=10, normalize_y=True
        gprModel2 = GaussianProcessRegressor(kernel=mainKernel2, random_state=random_state)
        gprModel3 = GaussianProcessRegressor(kernel=mainKernel3, random_state=random_state)
        gprModel4 = GaussianProcessRegressor(kernel=mainKernel4, random_state=random_state)
        gprModel5 = GaussianProcessRegressor(kernel=mainKernel5, random_state=random_state)

        #kernel1 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        #kernel2 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        #kernel3 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        #kernel4 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
        #kernel5 = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

        #scale = StandardScaler()
        #predictorsData = scale.fit_transform(predictorsData)

        d1 = predictorsData[:predictorsData.shape[0]-1]
        d2 = predictorsData[1:predictorsData.shape[0]]

        # SKLEARN
        gprModel1.fit(d1, d2[:, 0])
        gprModel2.fit(d1, d2[:, 1])
        gprModel3.fit(d1, d2[:, 2])
        gprModel4.fit(d1, d2[:, 3])
        gprModel5.fit(d1, d2[:, 4])

        # GPY
        #gprModel1 = GPy.models.GPRegression(d1, d2[:, 0].reshape(d2.shape[0], 1), kernel1)
        #gprModel2 = GPy.models.GPRegression(d1, d2[:, 1].reshape(d2.shape[0], 1), kernel2)
        #gprModel3 = GPy.models.GPRegression(d1, d2[:, 2].reshape(d2.shape[0], 1), kernel3)
        #gprModel4 = GPy.models.GPRegression(d1, d2[:, 3].reshape(d2.shape[0], 1), kernel4)
        #gprModel5 = GPy.models.GPRegression(d1, d2[:, 4].reshape(d2.shape[0], 1), kernel5)
        #gprModel1.optimize(messages=True)
        #gprModel2.optimize(messages=True)
        #gprModel3.optimize(messages=True)
        #gprModel4.optimize(messages=True)
        #gprModel5.optimize(messages=True)

        lastPreds = d1[-1].reshape(1,-1)
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

            lastPreds = np.array([gprModel1_Preds[0], gprModel2_Preds[0], gprModel3_Preds[0], gprModel4_Preds[0], gprModel5_Preds[0]])
            Preds_List.append(lastPreds)
            print("step_i = ", step_i, ", lastPreds = ", lastPreds)
            #print("lastPreds.shape = ", lastPreds.shape)
            lastPreds = lastPreds.reshape(1,-1)
            #print("lastPreds.shape = ", lastPreds.shape)
    #pd.DataFrame(Preds_List).plot()
    #plt.show()
    #time.sleep(3000)

    return Preds_List

################################## ROLLING RUN PROCESS ##############################
def RollingRunProcess(params):

    liftMethod = params['liftMethod']
    trainSetLength = params['trainSetLength']
    data = params['data']
    X_test = params['X_test']
    rolling_Embed_Memory = params['rolling_Embed_Memory']
    target_intrinsic_dim = params['target_intrinsic_dim']
    mode = params['mode']
    modeSplit = mode.split(',')
    rolling_Predict_Memory = params['rolling_Predict_Memory']
    embedMethod = params['embedMethod']
    processName = '_'.join([str(x) for x in list(params.values())[2:]])
    print(processName)
    #time.sleep(3000)

    model_GPR_Multi = GaussianProcessRegressor(kernel=1 * Matern(), random_state=0)

    roll_parsimoniousEigs_List = []
    PredsList, CI_Lower_Band_List, CI_Upper_Band_List = [], [], []
    for i in tqdm(range(trainSetLength, data.shape[0], 1)):  # , trainSetLength+10
        roll_X_train = data[i - rolling_Embed_Memory:i]
        roll_X_test = data[i]

        if liftMethod == 'FullModel':
            roll_target_mapping = roll_X_train
            roll_parsimoniousEigs = ""
            roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
            modelListSpectrum = roll_target_mapping.shape[1]
        else:
            roll_target_mapping_List = Embed(embedMethod, roll_X_train, target_intrinsic_dim)
            roll_target_mapping = roll_target_mapping_List[0]
            roll_parsimoniousEigs = roll_target_mapping_List[1]
            roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
            modelListSpectrum = roll_target_mapping.shape[1]

        if i == trainSetLength:
            model_GPR_List = [GaussianProcessRegressor(kernel=1 * RBF(), random_state=0) for var in range(modelListSpectrum)]

        ##############################################################################################################
        if modeSplit[1].strip() == 'VAR':
            roll_forecasting_model = VAR(roll_target_mapping[-rolling_Predict_Memory:])
            roll_model_fit = roll_forecasting_model.fit()
            roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=1, alpha=0.05)

            row_Preds = roll_target_mapping_Preds_All[0]
            #row_CI_Lower_Band = roll_target_mapping_Preds_All[1]
            #row_CI_Upper_Band = roll_target_mapping_Preds_All[2]
        elif modeSplit[1].strip() == 'GPR_Multi':

            roll_reframedData = reframeData(roll_target_mapping[-rolling_Predict_Memory:], 1, "all")
            model_GPR_Multi.fit(roll_reframedData[0], roll_reframedData[1])
            row_Preds, roll_target_mapping_Preds_Std = model_GPR_Multi.predict(roll_reframedData[2],return_std=True)
        elif modeSplit[1].strip() == 'GPR_Single':
            models_preds_list = []
            for modelIn in range(len(model_GPR_List)):
                roll_reframedData = reframeData(roll_target_mapping[-rolling_Predict_Memory:], 1, modelIn)
                model_GPR_List[modelIn].fit(roll_reframedData[0], roll_reframedData[1])
                sub_row_Preds, sub_roll_target_mapping_Preds_Std = model_GPR_List[modelIn].predict(roll_reframedData[2],return_std=True)
                models_preds_list.append(sub_row_Preds[0][0])

            row_Preds = np.array(models_preds_list).reshape(1, roll_target_mapping.shape[1])

            #print("X_train_gpr.shape = ", roll_reframedData[0].shape)
            #print("y_train_gpr.shape = ", roll_reframedData[1].shape)
            #print("test_point_gpr.shape = ", roll_reframedData[2])
        #print("row_Preds = ", row_Preds)
        #print("row_Preds.shape = ", row_Preds.shape)
        #print("roll_target_mapping.shape = ", roll_target_mapping.shape)
        #time.sleep(3000)

        if liftMethod == 'FullModel':
            PredsList.append(row_Preds[0])
            #CI_Lower_Band_List.append(row_CI_Lower_Band[0])
            #CI_Upper_Band_List.append(row_CI_Upper_Band[0])
        else:
            """
            "Build Combinations of the several CIs MSEs etc"
            varSet = [[] for varN in range(Preds.shape[1])]
            for varN in range(Preds.shape[1]):
                varSet[varN] = [["Mean_x" + str(varN), Preds[0][varN]], ["CI_lower_x" + str(varN), CI_Lower_Band[0][varN]],
                                ["CI_upper_x" + str(varN), CI_Upper_Band[0][varN]]]
            allCombos = [x for x in list(itertools.product(*varSet))]
            rowDataList = []
            for combo in allCombos:
                roll_target_mapping_Names = ','.join([x[0] for x in combo])
                roll_target_mapping_Preds = np.array([[x[1] for x in combo]])

                if liftMethod == "RBF":
                    single_lifted = RBFInterpolator(roll_target_mapping, roll_X_train, kernel=params["kernelIn"], degree=params["degreeIn"], neighbors=params["neighborsIn"], epsilon=params["epsilonIn"])(roll_target_mapping_Preds)[0]
                elif liftMethod == "GH":
                    single_lifted = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, roll_target_mapping_Preds, params["neighborsIn"])[0]
                elif liftMethod == "LP":
                    single_lifted = Lift("LP", roll_X_train, roll_X_test, roll_target_mapping, roll_target_mapping_Preds, params["neighborsIn"])[0][0]

                rowDataList.append([roll_target_mapping_Names, single_lifted])  # , single_LP_lifted
            rowData_df = pd.DataFrame(rowDataList, columns=["roll_target_mapping_Names", "single_lifted"])
            rowData_df['step'] = i
            Lifted_Data_List.append(rowData_df)
            """
            if liftMethod == "RBF":
                single_lifted_Preds = RBFInterpolator(roll_target_mapping, roll_X_train, kernel=params["kernelIn"], degree=params["degreeIn"],neighbors=params["neighborsIn"], epsilon=params["epsilonIn"])(row_Preds)[0]
                #single_CI_Lower_Band = RBFInterpolator(roll_target_mapping, roll_X_train, kernel=params["kernelIn"], degree=params["degreeIn"],neighbors=params["neighborsIn"], epsilon=params["epsilonIn"])(row_CI_Lower_Band)[0]
                #single_CI_Upper_Band = RBFInterpolator(roll_target_mapping, roll_X_train, kernel=params["kernelIn"], degree=params["degreeIn"],neighbors=params["neighborsIn"], epsilon=params["epsilonIn"])(row_CI_Upper_Band)[0]
            elif liftMethod == "GH":
                single_lifted_Preds = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_Preds, params["neighborsIn"])[0]
                #single_CI_Lower_Band = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_CI_Lower_Band,params["neighborsIn"])[0]
                #single_CI_Upper_Band = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_CI_Upper_Band,params["neighborsIn"])[0]

            PredsList.append(single_lifted_Preds)
            #CI_Lower_Band_List.append(single_CI_Lower_Band)
            #CI_Upper_Band_List.append(single_CI_Upper_Band)

    Preds = np.array(PredsList)
    #CI_Lower_Band = np.array(CI_Lower_Band_List)
    #CI_Upper_Band = np.array(CI_Upper_Band_List)

    print(PaperizePreds(Preds, X_test))
    pickle.dump(Preds, open(RollingRunnersPath + processName + ".p", "wb"))
    #pickle.dump(Lifted_X_Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + modelID + "_Preds.p", "wb"))

random_state = 0

modelParamsList = []

def Lift(method, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation, knn):
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
        #print("opt_epsilon = ", opt_epsilon)
        #print("opt_cutoff = ", opt_cutoff)
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
        mainKernel_Kriging_GP = 1 * RBF()
        gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP, normalize_y=True)
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        residual = gpr_model_fit.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gpr_model_fit.predict(eig_Simulation)
    elif method == 'SI': #Simple Linear ND Interpolator
        knn_interpolator = NearestNDInterpolator(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = knn_interpolator(eig_Simulation)
        residual = extrapolatedPsi_to_X - X_testSet

    try:
        mse_psnr_val = mse_psnr(extrapolatedPsi_to_X, X_testSet)
        mse = mse_psnr_val[0]
        #psnr = mse_psnr_val[1]
        #r2_score_val = r2_score(X_testSet, extrapolatedPsi_to_X)
        rmse = np.sqrt(mse)
        #nrmse = rmse / (np.amax(extrapolatedPsi_to_X)-np.amin(extrapolatedPsi_to_X))
        #mape = mean_absolute_percentage_error(X_testSet, extrapolatedPsi_to_X)
    except Exception as e:
        #print(e)
        mse_psnr_val = np.nan
        mse = np.nan
        #psnr = np.nan
        #r2_score_val = np.nan
        rmse = np.nan
        #nrmse = np.nan
        #mape = np.nan

    return [extrapolatedPsi_to_X, mse, rmse, residual]

def RunWithMatlabData(RunMode, TimeConfiguration, liftMethod, label):

    knnList = [20]

    if label == 'EEGdatanew':
        no_dims = 2
    elif label == 'EEGdataNonLinear':
        no_dims = 3

    if RunMode == 'Run':
        if TimeConfiguration == 'static':
            MSE_List = []
            for dmKnn in knnList:
                mat = scipy.io.loadmat('D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Static_DM_Matlab_Data_'+str(dmKnn)+'_1_'+str(no_dims)+'_'+label+'.mat')
                data = mat['Static_DM_Matlab_Data'][0]

                X_trainingSet = data[0]
                X_testSet = data[1]
                eig_trainingSet = data[2]
                elem = data[3]

                #for knn in knnList:
                knn = dmKnn
                for subElem in range(elem.shape[1]):
                    lift_out_eig_Simulation = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, elem[0][subElem], knn)
                    lift_out_eig_Simulation.append(dmKnn)
                    lift_out_eig_Simulation.append(knn)
                    lift_out_eig_Simulation.append(X_testSet)
                    MSE_List.append(lift_out_eig_Simulation)
                print("done dmKnn = ", dmKnn, " ... label = ", label)
            pickle.dump(MSE_List, open("MSE_List_" + str(TimeConfiguration) + "_" + liftMethod + "_" + label + ".p", "wb"))
            print("done all!")
        else:
            MSE_List = []
            for dmKnn in knnList:
                mat = scipy.io.loadmat(
                    'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Dynamic_DM_Matlab_Data_TS_'+str(dmKnn)+'_1_'+str(no_dims)+'_'+label+'.mat')
                data = mat['Dynamic_Matlab_Data_TS']

                #for knn in knnList:
                knn = dmKnn
                print("dmKnn = ", dmKnn, ", knn = ", knn)
                lift_out_Preds_list = []
                real_x_list = []
                for i in range(data.shape[0]):
                    print(i, ", ", data[i].shape)
                    ################################################
                    rowData = data[i]
                    y_intv = rowData[0]
                    obj_evecs = rowData[1]
                    Preds = rowData[2]
                    CI_Lower_Band = rowData[3]
                    CI_Upper_Band = rowData[4]
                    ################################################
                    varSet = [[] for varN in range(Preds.shape[1])]
                    varSetLabels = [[] for varN in range(Preds.shape[1])]
                    for varN in range(Preds.shape[1]):
                        varSet[varN] = [Preds[0][varN], CI_Lower_Band[0][varN], CI_Upper_Band[0][varN]]
                        if i == 0:
                            varSetLabels[varN] = ["Mean_Var"+str(varN), "CI_Lower_Band_Var"+str(varN), "CI_Upper_Band_Var"+str(varN)]
                    #allStatsCombos = [[x[0], x[1]] for x in itertools.product(*varSet)]
                    allStatsCombos = [list(x) for x in itertools.product(*varSet)]
                    if i == 0:
                        allStatsCombos_Labels = [[x[0]+"-"+x[1]] for x in itertools.product(*varSetLabels)]
                    ################################################
                    y_i = rowData[5]
                    real_x_list.append(y_i[0])

                    liftedCombos = [[] for varN in range(len(allStatsCombos))]
                    ## Process restricted Data of each combination (lift them back) for EACH stats combo
                    countCombo = 0
                    for subPred in allStatsCombos:
                        lift_out_Preds = Lift(liftMethod, y_intv, y_i, obj_evecs, np.array(subPred).reshape(1, -1), knn)[0]

                        if liftMethod not in ["GH"]:
                            lift_out_Preds = lift_out_Preds[0]
                        liftedCombos[countCombo].append(lift_out_Preds.tolist())
                        countCombo += 1

                    lift_out_Preds_list.append(liftedCombos)

                liftedCombos_TS = [[] for tsCount in range(len(allStatsCombos))]
                for tsCount in range(len(allStatsCombos)):
                    for t in range(len(lift_out_Preds_list)):
                        t_sub_lift_out_Preds_list = lift_out_Preds_list[t]
                        liftedCombos_TS[tsCount].append(t_sub_lift_out_Preds_list[tsCount][0])

                    extrapolatedPsi_to_X_Preds = np.array(liftedCombos_TS[tsCount])
                    X_testSet = np.array(real_x_list)

                    #fig, ax = plt.subplots(sharex=True, nrows=2, ncols=1)
                    #pd.DataFrame(X_testSet).plot(ax=ax[0], title='X Test Set')
                    #pd.DataFrame(extrapolatedPsi_to_X_Preds).plot(ax=ax[1], title='extrapolatedPsi_to_X_Preds')
                    #plt.show()
                    #time.sleep(3000)

                    mse_psnr_val = mse_psnr(extrapolatedPsi_to_X_Preds, X_testSet)
                    mse = mse_psnr_val[0]
                    psnr = mse_psnr_val[1]
                    r2_score_val = r2_score(X_testSet, extrapolatedPsi_to_X_Preds)
                    rmse = np.sqrt(mse)
                    nrmse = rmse / (np.amax(extrapolatedPsi_to_X_Preds) - np.amin(extrapolatedPsi_to_X_Preds))
                    mape = mean_absolute_percentage_error(X_testSet, extrapolatedPsi_to_X_Preds)

                    MSE_List.append([dmKnn, knn, allStatsCombos_Labels[tsCount], mse, psnr, r2_score_val, rmse, nrmse, mape, extrapolatedPsi_to_X_Preds, X_testSet])

            pickle.dump(MSE_List, open("MSE_List_" + str(TimeConfiguration) + "_" + liftMethod + '_' + label + ".p", "wb"))
            print("done all!")

    elif RunMode == 'Read':
        readData = pickle.load(open("MSE_List_" + str(TimeConfiguration) + "_" + liftMethod + "_" + label + ".p", "rb"))
        if TimeConfiguration == 'static':
            MSE_df = pd.DataFrame([[] for x in readData], columns=["mse", "rmse", "dmKnn", "knn"])
        else:
            MSE_df = pd.DataFrame([[] for x in readData], columns=["mse", "rmse", "dmKnn", "knn"])

        MSE_df_MIN = MSE_df.groupby(['dmKnn']).min()
        MSE_df_MIN = MSE_df_MIN.round(4)[["mse", "rmse"]]
        MSE_df_MIN['perfText'] = MSE_df_MIN.astype(str).agg(' & '.join, axis=1)

        MSE_df_MIN.to_excel("MSE_List_" + str(TimeConfiguration) + "_" + liftMethod + "_" + label + ".xlsx")
        print("TimeConfiguration = ", TimeConfiguration, ", label = ", label, ", liftMethod = ", liftMethod, " ... ", ", MSE_df_MIN['perfText'] = ", MSE_df_MIN['perfText'])

try:
    alreadyRunProcessesExcel = pd.read_excel("Reporter_dataDF_raw.xlsx")
    alreadyRunProcesses = alreadyRunProcessesExcel["file_name"].tolist()
except Exception as e:
    print(e)

def RunPythonDM(paramList):

    label = paramList[0]
    mode = paramList[1]
    simulationNumber = paramList[2]
    datasetsPath = paramList[3]
    RollingRunnersPath = paramList[4]
    embedMethod = paramList[5]

    matFileName = datasetsPath + label + "_" + str(simulationNumber) + '.mat'
    print("matFileName = ", matFileName)
    mat = scipy.io.loadmat(matFileName)
    #data = mat['y']
    data = mat['Ytrain']
    forecastHorizon = 500
    rolling_Embed_Memory = 500
    rolling_Predict_Memory = 100
    trainSetLength = data.shape[0]-forecastHorizon
    modeSplit = mode.split(',')

    if 'nonlin' in label:
        target_intrinsic_dim = 3
    elif 'FX' in label:
        target_intrinsic_dim = 3
    else:
        target_intrinsic_dim = 2

    X_train = data[:trainSetLength]
    X_test = data[trainSetLength:]
    X_test_shifted = sl.S(pd.DataFrame(X_test)).values
    X_test_shifted[0] = X_train[-1]

    if modeSplit[0] == 'FullModel_Static':

        if modeSplit[1] == "VAR":
            forecasting_model = VAR(X_train)
            model_fit = forecasting_model.fit()
            FullModel_VAR_Preds = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
            #FullModel_VAR_Preds = model_fit.forecast_interval(X_test_shifted, steps=forecastHorizon, alpha=0.05)
            Preds = FullModel_VAR_Preds[0]
        elif modeSplit[1] in ["GPR_Single", "ANN_Single"]:
            Preds_List = get_ML_Predictions("Main", modeSplit[1], X_train, X_test_shifted, forecastHorizon)
            #Preds_List = get_ML_Predictions("Use_shifted_Test_Set", modeSplit[1], X_train, X_test_shifted, forecastHorizon)
            #Preds_List = get_ML_Predictions("HardCoded", modeSplit[1], X_train, X_test_shifted, forecastHorizon)
            Preds = pd.DataFrame(Preds_List).values

        FullModel_Static_PaperText = PaperizePreds(Preds, X_test)
        print("FullModel_Static_PaperText = ", FullModel_Static_PaperText)
        pickle.dump(Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + embedMethod + ".p", "wb"))
    elif modeSplit[0] == "FullModel_Rolling":

        params = {'data': data,
                  'X_test': X_test,
                  'label': label,
                  'simulationNumber': simulationNumber,
                  'liftMethod': "FullModel",
                  'trainSetLength': trainSetLength,
                  'rolling_Embed_Memory': rolling_Embed_Memory,
                  'target_intrinsic_dim': target_intrinsic_dim, 'mode': "FullModel_Rolling,"+modeSplit[1],
                  'rolling_Predict_Memory': rolling_Predict_Memory,
                  'embedMethod': embedMethod,
                  'kernelIn': np.nan, 'degreeIn': np.nan, 'neighborsIn': np.nan, 'epsilonIn': np.nan
                  }

        RollingRunProcess(params)
    ###################################################################################################################
    elif modeSplit[0] == 'Static_run':

        ###################################### REDUCED TARGET SPACE #######################################
        "Embed"
        target_mapping_List = Embed(embedMethod, X_train, target_intrinsic_dim)
        target_mapping = target_mapping_List[0]
        #parsimoniousEigs = target_mapping_List[1]
        #target_mapping_EigVals = target_mapping_List[3]

        "Forecast and get CIs for the embedded space"
        if modeSplit[1] == "VAR":
            forecasting_model = VAR(target_mapping)
            model_fit = forecasting_model.fit()
            target_mapping_Preds_All = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
            mapped_Preds = target_mapping_Preds_All[0]
        elif modeSplit[1] in ["GPR_Single", "ANN_Single"]:

            Preds_List = get_ML_Predictions("Main", modeSplit[1], target_mapping, [], forecastHorizon)
            mapped_Preds = pd.DataFrame(Preds_List).values

        #if simulationNumber == 0:
        #    pd.DataFrame(target_mapping).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_Eigs_to_Matlab.xlsx")
        #    pd.DataFrame(target_mapping_EigVals).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_EigsVals_to_Matlab.xlsx")
        #    pd.DataFrame(mapped_Preds).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_Preds_to_Matlab.xlsx")

        #statsList = []
        for liftMethod in ["SI", "GH", "RBF", "KR"]: #
            if liftMethod == "RBF":
                for kernelIn in ['linear']: # , 'gaussian'
                    for degreeIn in [1]:
                        for neighborsIn in [50]:
                            for epsilonIn in [1]:
                                modelID = str(kernelIn) + '_' + str(degreeIn) + '_' + str(neighborsIn) + '_' + str(epsilonIn) + '_' + str(liftMethod)
                                Lifted_X_Preds = RBFInterpolator(target_mapping, X_train, kernel=kernelIn,degree=degreeIn, neighbors=neighborsIn,epsilon=epsilonIn)(mapped_Preds)
                                print("Static_Run, ", modeSplit[1], ", liftMethod = ", liftMethod, ", Lifted_X_Preds = ", Lifted_X_Preds)
                                pickle.dump(Lifted_X_Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + modelID + "_" + embedMethod + ".p","wb"))
            else: # knn Greedy Search is only valid for GH method = pls make another 'if' in case of other methods G.Sea
                for knn in [50]:
                    modelID = str(knn) + '_' + str(liftMethod)
                    Lifted_X_Preds = Lift(liftMethod, X_train, X_test, target_mapping, mapped_Preds, knn)[0]
                    print("Static_Run, ", modeSplit[1], ", liftMethod = ", liftMethod, ", Lifted_X_Preds = ", Lifted_X_Preds)
                    pickle.dump(Lifted_X_Preds, open(RollingRunnersPath + label + "_" + str(simulationNumber) + "_" + mode.replace(",","_") + "_" + modelID + "_" + embedMethod + ".p","wb"))
    ###################################################################################################################
    elif modeSplit[0] == 'Rolling_run':

        processList = []
        for liftMethod in ["RBF", "GH"]:
            for kernelIn in ['linear']:
                for degreeIn in [1]: #, 2, 3
                    for neighborsIn in [50]:
                        for epsilonIn in [1]: #0.1,
                            params = {'data': data,
                                'X_test': X_test,
                                'label': label,
                                'simulationNumber': simulationNumber,
                                'liftMethod': liftMethod,
                                'trainSetLength': trainSetLength,
                                'rolling_Embed_Memory': rolling_Embed_Memory,
                                'target_intrinsic_dim': target_intrinsic_dim, 'mode': mode,
                                'rolling_Predict_Memory': rolling_Predict_Memory,
                                'embedMethod': embedMethod,
                                'kernelIn':kernelIn, 'degreeIn':degreeIn, 'neighborsIn':neighborsIn, 'epsilonIn':epsilonIn
                            }
                            #processList.append(params)
                            "Check if process already processed (alreadyRunProcesses list above)!"
                            CheckedProcessName = '_'.join([str(x) for x in list(params.values())[2:]])+"_Preds.p"
                            if CheckedProcessName not in alreadyRunProcesses:
                                RollingRunProcess(params)
                            else:
                                print("Process ", CheckedProcessName, " has beed already processed !!!")
                                print("###############################################################")

        #p = mp.Pool(mp.cpu_count())
        #result = p.map(RollingRunProcess, tqdm(processList))
        #p.close()
        #p.join()
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
        full_eigenval_df.to_csv(label+"_"+mode+".csv")

def Reporter(mode, datasetsPath, RollingRunnersPath):

    forecastHorizon = 500

    if mode == "RunRandomWalks":

        allDataListRW = []
        for dataset_total_name in glob.glob(datasetsPath+'*.mat'):
            mat_dataset = scipy.io.loadmat(dataset_total_name)
            #dataRW = mat_dataset['y']
            dataRW = mat_dataset['Ytrain']
            trainSetLengthRW = dataRW.shape[0] - forecastHorizon
            X_testRW = dataRW[trainSetLengthRW:]

            PredsRW = sl.S(pd.DataFrame(X_testRW)).fillna(0).values
            rmseVarsRW = PaperizePreds(PredsRW, X_testRW, returnData='yes')
            allDataListRW.append([dataset_total_name.split("\\")[-1]+"_"+"-"+"_"+"-"+"_"+"-"+"_"+"-", rmseVarsRW])

        dataDFRW = pd.DataFrame(allDataListRW, columns=["file_name", "rmseVars"]).set_index("file_name", drop=True)
        dataDFRW.to_excel("Reporter_dataDF_RW_raw.xlsx")
    elif mode == "Run":

        allDataList = []
        for file_total_name in glob.glob(RollingRunnersPath+'\\*.p'):
            file_name = file_total_name.split("\\")[-1]
            file_name_split = file_name.split("_")
            label = file_name_split[0]
            simulationNumber = file_name_split[1]

            matFileName = datasetsPath + label + "_" + str(simulationNumber) + '.mat'
            mat = scipy.io.loadmat(matFileName)
            #data = mat['y']
            data = mat['Ytrain']
            trainSetLength = data.shape[0] - forecastHorizon
            X_test = data[trainSetLength:]

            Preds = pickle.load(open(file_total_name, "rb"))
            rmseVars = PaperizePreds(Preds, X_test, returnData='yes')

            allDataList.append([file_name, rmseVars])

        dataDF = pd.DataFrame(allDataList, columns=["file_name", "rmseVars"]).set_index("file_name", drop=True)
        dataDF.to_excel("Reporter_dataDF_raw.xlsx")
    elif mode == "Read":

        dataDF = pd.concat([pd.read_excel("Reporter_dataDF_RW_raw.xlsx"), pd.read_excel("Reporter_dataDF_raw.xlsx")])
        dataDF['Dataset'] = dataDF["file_name"].str.split("_").str[0]
        dataDF['SimulationNumber'] = dataDF["file_name"].str.split("_").str[1]
        dataDF['ID'] = dataDF["file_name"].str.split("_", n=2).str[2:].apply(lambda x: x[0])
        dataDF['rmseVars'] = dataDF['rmseVars'].str.replace("[", "").str.replace("]", "")
        dataDF["Dataset_ID"] = dataDF['Dataset'] + "_" + dataDF["ID"]

        ReportList = []
        for elem in tqdm(set(dataDF["Dataset_ID"].tolist())):

            subDF = dataDF[dataDF["Dataset_ID"] == elem]
            "Check First!"
            if subDF.shape[0] > 100:
                print(elem)
                print(subDF.shape)
                print("Sth's wrong here!!! duplicate (?)")
                time.sleep(3000)

            "Calculate RMSE(variables), Medians and CIs"
            rmseVarsDF = subDF['rmseVars'].apply(lambda x: x.split(",")).apply(pd.Series).reset_index(drop=True).astype(float)
            rmseVarsDF_median = rmseVarsDF.median().tolist()
            #conf = sms.DescrStatsW(rmseVarsDF.values).tconfint_mean()
            percentile5 = np.percentile(rmseVarsDF.values, 5, axis=0)
            percentile95 = np.percentile(rmseVarsDF.values, 95, axis=0)
            reportText = ' & '.join([str(np.round(rmseVarsDF_median[c],3)) + ' (' + str(np.round(percentile5[c],3)) + ',' + str(np.round(percentile95[c],3)) +')' for c in range(len(rmseVarsDF_median))])

            ReportList.append([elem, reportText, subDF.shape[0]])

        ReportDF = pd.DataFrame(ReportList, columns=["Dataset_ID", "reportText", "#simulations"]).set_index("Dataset_ID", drop=True)
        ReportDF.to_excel("Reporter.xlsx")

def Test():
    matFileName = 'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\EEGsynthetic2nonlin_0.mat'
    mat = scipy.io.loadmat(matFileName)
    data = mat['Ytrain']
    forecastHorizon = 500
    trainSetLength = data.shape[0] - forecastHorizon
    X_test = data[trainSetLength:]

    Preds1 = pickle.load(open("D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\EEGsynthetic2nonlin_0_FullModel_Static_VAR_DM.p", "rb"))
    Preds2 = pickle.load(open("D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\EEGsynthetic2nonlin_0_FullModel_Static_GPR_Single_DM.p", "rb"))

    print("VAR")
    print(pd.DataFrame(Preds1))#.tail(5))
    print(PaperizePreds(Preds1, X_test, returnData='yes'))
    print("GPR")
    print(pd.DataFrame(Preds2))#.tail(5))
    print(PaperizePreds(Preds2, X_test, returnData='yes'))

    pd.concat([pd.DataFrame(Preds1, columns=["Preds1_"+str(x) for x in range(Preds1.shape[1])]), pd.DataFrame(Preds2, columns=["Preds2_"+str(x) for x in range(Preds1.shape[1])])], axis=1).plot()
    plt.show()

if __name__ == '__main__':

    label = "EEGsynthetic2" # "FxDataAdjRets", "EEGsynthetic2", "EEGsynthetic2nonlin" (third and final approach)
    embedMethod = "DM" # PCA, LLE, DM

    datasetsPath = 'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\DataSets_Siettos_ThirdApproach\\'
    simulNum = 100 # 5, 100 CAUTION! How many datasets(simulations) to scan in the target folder !!!
    RollingRunnersPath = "D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\\"

    ##### STATIC ####
    #processToRun = "FullModel_Static,VAR" # done both
    #processToRun = "FullModel_Static,GPR_Single"
    processToRun = "Static_run,VAR"
    #processToRun = "Static_run,GPR_Single" # Siettos
    #processToRun = "Static_run,ANN_Single"

    ##### ROLLING ####
    #processToRun = "FullModel_Rolling,VAR" # 1 done(100), 2 done(100)
    #processToRun = "FullModel_Rolling,GPR_Single"
    #processToRun = "Rolling_run,VAR" # , 2 done(100)
    #processToRun = "Rolling_run,GPR_Single"

    "Run Single Process (for testing)"
    #RunPythonDM([label, "FullModel_Static,VAR", 0, datasetsPath, RollingRunnersPath, embedMethod])
    #RunPythonDM([label, "FullModel_Static,GPR_Single", 0, datasetsPath, RollingRunnersPath, embedMethod])

    #Test()

    runProcessesParallel = 1

    if runProcessesParallel == 1:

        simulList = []
        for simul in range(simulNum):
            try:
                simulList.append([label, processToRun, simul, datasetsPath, RollingRunnersPath, embedMethod])
            except Exception as e:
                print(simul)
                print(e)
        p = mp.Pool(mp.cpu_count())
        result = p.map(RunPythonDM, tqdm(simulList))
        p.close()
        p.join()

    elif runProcessesParallel == 0:
        " REPORT RESULTS "
        Reporter("RunRandomWalks", datasetsPath, RollingRunnersPath) # Done!
        Reporter("Run", datasetsPath, RollingRunnersPath)
        Reporter("Read", datasetsPath, RollingRunnersPath)

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
