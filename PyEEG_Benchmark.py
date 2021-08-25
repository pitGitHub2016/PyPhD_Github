from Slider import Slider as sl
import scipy.io, glob
import itertools, math
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.metrics import r2_score
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import cross_val_score, train_test_split
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection,
)
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector
from skopt import gp_minimize
from skopt.searchcv import BayesSearchCV
from skopt.space import Categorical, Integer, Real

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding

RollingRunnersPath = "D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners\\"

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mse_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return [mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))]

def reframeData(dataIn, reframeStep, varSelect):

    baseDF = pd.DataFrame(dataIn)

    df_List = []
    for i in range(reframeStep+1):
        if i == 0:
            subDF_i0 = baseDF.copy()
            subDF_i0.columns = ["base_" + str(x) for x in subDF_i0.columns]
            df_List.append(subDF_i0)
        else:
            subDF = baseDF.shift(i).fillna(0)
            subDF.columns = ["s_"+str(x) for x in subDF.columns]
            df_List.append(subDF)
    df = pd.concat(df_List, axis=1)
    if varSelect == "all":
        Y_DF = df.loc[:, [x for x in df.columns if "base_" in x]]
    else:
        Y_DF = df.loc[:, "base_"+str(varSelect)]
    X_DF = df.loc[:, [x for x in df.columns if "s_" in x]]
    lastY_test_point = df.loc[df.index[-1], [x for x in df.columns if "base_" in x]]

    print(df)
    print("X_DF = ", X_DF.tail(5))
    print("Y_DF = ", Y_DF.tail(5))
    print(lastY_test_point.values)
    time.sleep(3000)

    X_all_gpr, y_all_gpr, lastY_test_point_gpr = X_DF.values, Y_DF.values, lastY_test_point.values

    return [X_all_gpr, y_all_gpr, lastY_test_point_gpr]

def getTargetMapping(X_train_local, target_intrinsic_dim):
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

    return [target_mapping, parsimoniousEigs, X_pcm.kernel.epsilon, dmap_local.eigenvalues_[selection.evec_indices_]]

def TradePreds(X_Preds, X_test):
    trSig = pd.DataFrame(X_Preds)
    X_test_df = pd.DataFrame(X_test)

    pnl = sl.rs(sl.sign(trSig) * X_test_df)
    pnl_sh = np.sqrt(252) * sl.sharpe(pnl)

    return pnl_sh

def PaperizePreds(X_Preds, X_test): # CI_Lower_Band, CI_Upper_Band

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
        rmse = np.round(np.sqrt(mse_psnr(X_Preds_df[col].values, X_test_df[col].values)[0]), 4)
        outList.append(rmse)

    textOut = ' & '.join([str(x) for x in outList])
    
    #print(textOut)
    #outDF["RMSE"] = rmse
    return textOut

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
    processName = '_'.join([str(x) for x in list(params.values())[2:]])
    print(processName)
    #time.sleep(3000)

    mainKernel = 1 * RBF()
    model_GPR = GaussianProcessRegressor(kernel=mainKernel, random_state=0)

    roll_parsimoniousEigs_List = []
    PredsList, CI_Lower_Band_List, CI_Upper_Band_List = [], [], []
    for i in tqdm(range(trainSetLength, data.shape[0], 1)):  # , trainSetLength+10
        roll_X_train = data[i - rolling_Embed_Memory:i]
        roll_X_test = data[i]

        if liftMethod == 'FullModel':
            roll_target_mapping = roll_X_train
            roll_parsimoniousEigs = ""
            roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
        else:
            roll_target_mapping_List = getTargetMapping(roll_X_train, target_intrinsic_dim)
            roll_target_mapping = roll_target_mapping_List[0]
            roll_parsimoniousEigs = roll_target_mapping_List[1]
            roll_parsimoniousEigs_List.append(roll_parsimoniousEigs)
        ##############################################################################################################
        if modeSplit[1].strip() == 'VAR':
            roll_forecasting_model = VAR(roll_target_mapping[-rolling_Predict_Memory:])
            roll_model_fit = roll_forecasting_model.fit()
            roll_target_mapping_Preds_All = roll_model_fit.forecast_interval(roll_model_fit.y, steps=1, alpha=0.05)

            row_Preds = roll_target_mapping_Preds_All[0]
            #row_CI_Lower_Band = roll_target_mapping_Preds_All[1]
            #row_CI_Upper_Band = roll_target_mapping_Preds_All[2]

        elif modeSplit[1].strip() == 'GPR':
            roll_reframedData = reframeData(roll_target_mapping[-rolling_Predict_Memory:])
            X_train_gpr = roll_reframedData[0]
            y_train_gpr = roll_reframedData[1]
            # Create a new GPR each time!!! no memory is retained
            model_GPR.fit(X_train_gpr, y_train_gpr)
            roll_target_mapping_Preds, roll_target_mapping_Preds_Std = model_GPR.predict(y_train_gpr[-1].reshape(1, -1),return_std=True)

            row_Preds = roll_target_mapping_Preds

        #print("row_Preds.shape = ", row_Preds.shape)
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
                single_lifted_Preds = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_Preds,params["neighborsIn"])[0]
                #single_CI_Lower_Band = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_CI_Lower_Band,params["neighborsIn"])[0]
                #single_CI_Upper_Band = Lift("GH", roll_X_train, roll_X_test, roll_target_mapping, row_CI_Upper_Band,params["neighborsIn"])[0]

            PredsList.append(single_lifted_Preds)
            #CI_Lower_Band_List.append(single_CI_Lower_Band)
            #CI_Upper_Band_List.append(single_CI_Upper_Band)

    Preds = np.array(PredsList)
    #CI_Lower_Band = np.array(CI_Lower_Band_List)
    #CI_Upper_Band = np.array(CI_Upper_Band_List)

    #out = PaperizePreds(Preds, CI_Lower_Band, CI_Upper_Band, X_test)
    out = Preds
    pickle.dump(out, open(RollingRunnersPath+processName + "_Preds.p", "wb"))

random_state = 1

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
        gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP)
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        residual = gpr_model_fit.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gpr_model_fit.predict(eig_Simulation)

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

def RunPythonDM(label, mode):

    # label = "EEGdataNonLinear"
    mat = scipy.io.loadmat(
        'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\' + label + '.mat')
    data = mat['y']
    print("data.shape = ", data.shape)
    forecastHorizon = 500
    rolling_Embed_Memory = 500
    rolling_Predict_Memory = 100
    trainSetLength = data.shape[0]-forecastHorizon

    modeSplit = mode.split(',')
    print("modeSplit = ", modeSplit)

    if label == 'EEGdatanew':
        target_intrinsic_dim = 2
    else:
        target_intrinsic_dim = 3

    print("label = ", label, " : target_intrinsic_dim = ", target_intrinsic_dim)

    X_train = data[:trainSetLength]
    X_test = data[trainSetLength:]
    print("X_train.shape = ", X_train.shape)
    print("X_test.shape = ", X_test.shape)
    
    if modeSplit[0] == 'FullModel_Static':

        RandomWalkPaperText = PaperizePreds(sl.S(pd.DataFrame(X_test)).fillna(0).values, X_test)  # , CI_Lower_Band, CI_Upper_Band
        print("RandomWalkPaperText = ", RandomWalkPaperText)

        if modeSplit[1] == "VAR":
            forecasting_model = VAR(X_train)
            model_fit = forecasting_model.fit()
            # print(results.summary())
            FullModel_VAR_Preds = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)

            Preds = FullModel_VAR_Preds[0]
            #CI_Lower_Band = FullModel_VAR_Preds[1]
            #CI_Upper_Band = FullModel_VAR_Preds[2]
        elif modeSplit[1] == "GPR":
            #print("X_train_gpr.shape = ", X_train_gpr.shape)
            #print("y_train_gpr.shape = ", y_train_gpr.shape)
            #print("lastY_test_point.shape = ", lastY_test_point.shape)
            #time.sleep(3000)
            # Create a new GPR each time!!! no memory is retained

            #train_dataPackLists = [[] for var in range(X_train.shape[1])]
            #for varNum in range(X_train.shape[1]):
            #    reframedDataTrain = reframeData(X_train, 1, varNum)
            #    X_train_gpr = reframedDataTrain[0]
            #    y_train_gpr = reframedDataTrain[1].reshape(-1, 1)
            #    entry_train_points = reframedDataTrain[2].reshape(1, X_train.shape[1])
            #    train_dataPackLists[varNum].append([X_train_gpr, y_train_gpr, entry_train_points])

            reframedDataTrain = reframeData(X_train, 2, "all")

            model_gpr = GaussianProcessRegressor(kernel=1 * RBF(), random_state=0)
            print(reframedDataTrain[0].shape)
            print(reframedDataTrain[1].shape)
            time.sleep(3000)
            model_gpr.fit(reframedDataTrain[0], reframedDataTrain[1])

            Preds, sub_indPrediction_Std = model_gpr.predict(reframeData(X_test, 1, 0)[0], return_std=True)
            #pd.DataFrame(Preds).plot()
            #plt.show()
            print("Preds.shape = ", Preds.shape)
            #time.sleep(3000)

        pickle.dump(Preds, open(RollingRunnersPath + label + "_" + mode.replace(",","_") + "_Preds.p", "wb"))
        FullModel_Static_PaperText = PaperizePreds(Preds, X_test) #, CI_Lower_Band, CI_Upper_Band
        print("FullModel_Static_PaperText = ", FullModel_Static_PaperText)

        #mse_Preds = mse_psnr(Preds, X_test)[0]
        #mse_CI_Lower_Band = mse_psnr(CI_Lower_Band, X_test)[0]
        #mse_CI_Upper_Band = mse_psnr(CI_Upper_Band, X_test)[0]
        #rmse_Preds = np.sqrt(mse_Preds)
        #rmse_CI_Lower_Band = np.sqrt(mse_CI_Lower_Band)
        #rmse_CI_Upper_Band = np.sqrt(mse_CI_Upper_Band)
        #print("Preds.shape = ", Preds.shape, ", CI_Lower_Band.shap = ", CI_Lower_Band.shape, ", CI_Upper_Band.shape = ", CI_Upper_Band.shape)
        #print("rmse_Preds = ", rmse_Preds, ", rmse_CI_Lower_Band = ", rmse_CI_Lower_Band, ", rmse_CI_Upper_Band = ", rmse_CI_Upper_Band)
    elif modeSplit[0] == "FullModel_Rolling":

        params = {'data': data,
                  'X_test': X_test,
                  'label':label,
                  'liftMethod': "FullModel",
                  'trainSetLength': trainSetLength,
                  'rolling_Embed_Memory': rolling_Embed_Memory,
                  'target_intrinsic_dim': target_intrinsic_dim, 'mode': "FullModel,"+modeSplit[1],
                  'rolling_Predict_Memory': rolling_Predict_Memory,
                  'kernelIn': np.nan, 'degreeIn': np.nan, 'neighborsIn': np.nan, 'epsilonIn': np.nan
                  }

        RollingRunProcess(params)
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
    elif modeSplit[0] == 'Static_run':

        ###################################### REDUCED TARGET SPACE #######################################
        "Embed"
        target_mapping_List = getTargetMapping(X_train, target_intrinsic_dim)
        target_mapping = target_mapping_List[0]
        parsimoniousEigs = target_mapping_List[1]
        target_mapping_EigVals = target_mapping_List[3]

        "Forecast and get CIs for the embedded space"
        if modeSplit[1] == "VAR":
            forecasting_model = VAR(target_mapping)
            model_fit = forecasting_model.fit()
            # print(results.summary())
            target_mapping_Preds_All = model_fit.forecast_interval(model_fit.y, steps=forecastHorizon, alpha=0.05)
        elif modeSplit[1] == "GPR":
            reframedData = reframeData(target_mapping)
            X_train_gpr = reframedData[0]
            y_train_gpr = reframedData[1]
            # Create a new GPR each time!!! no memory is retained
            mainKernel = 1 * RBF()
            model_GPR = GaussianProcessRegressor(kernel=mainKernel, random_state=0)
            model_GPR.fit(X_train_gpr, y_train_gpr)
            roll_target_mapping_Preds, roll_target_mapping_Preds_Std = model_GPR.predict(y_train_gpr[-1].reshape(1, -1),return_std=True)

        mapped_Preds = target_mapping_Preds_All[0]
        mapped_CI_Lower_Band = target_mapping_Preds_All[1]
        mapped_CI_Upper_Band = target_mapping_Preds_All[2]

        pd.DataFrame(target_mapping).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_Eigs_to_Matlab.xlsx")
        pd.DataFrame(target_mapping_EigVals).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_EigsVals_to_Matlab.xlsx")
        pd.DataFrame(mapped_Preds).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_Preds_to_Matlab.xlsx")
        pd.DataFrame(mapped_CI_Lower_Band).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_CI_Lower_Band_to_Matlab.xlsx")
        pd.DataFrame(mapped_CI_Upper_Band).to_excel(label+"_"+modeSplit[1]+"_DM_mapped_CI_Upper_Band_to_Matlab.xlsx")

        statsList = []
        for liftMethod in ["RBF", "GH"]:
            if liftMethod == "RBF":
                for kernelIn in ['linear']: # , 'gaussian'
                    for degreeIn in [1]:
                        for neighborsIn in [50]:
                            for epsilonIn in [1]:

                                modelID = str(kernelIn) + '_' + str(degreeIn) + '_' + str(neighborsIn) + '_' + str(epsilonIn) + '_' + str(liftMethod)

                                Lifted_X_Preds = RBFInterpolator(target_mapping, X_train, kernel=kernelIn,degree=degreeIn, neighbors=neighborsIn,epsilon=epsilonIn)(mapped_Preds)
                                #Lifted_X_CI_Lower_Band = RBFInterpolator(target_mapping, X_train, kernel=kernelIn,degree=degreeIn, neighbors=neighborsIn,epsilon=epsilonIn)(mapped_CI_Lower_Band)
                                #Lifted_X_CI_Upper_Band = RBFInterpolator(target_mapping, X_train, kernel=kernelIn,degree=degreeIn, neighbors=neighborsIn,epsilon=epsilonIn)(mapped_CI_Upper_Band)

                                paperText = PaperizePreds(Lifted_X_Preds, X_test) # Lifted_X_CI_Lower_Band, Lifted_X_CI_Upper_Band

                                # TRADING
                                #trade_pnl_sh = TradePreds(mapped_Preds, X_test)

                                statsList.append([modelID, paperText])  # , pnl_sh
            else:
                for knn in [50]:
                    # TRADING
                    #trade_pnl_sh = TradePreds(Lifted_X_Preds[0], X_test)
                    modelID = str(knn) + '_' + str(liftMethod)

                    Lifted_X_Preds = Lift(liftMethod, X_train, X_test, target_mapping, mapped_Preds, knn)[0]
                    #Lifted_X_CI_Lower_Band = Lift(liftMethod, X_train, X_test, target_mapping, mapped_CI_Lower_Band, knn)[0]
                    #Lifted_X_CI_Upper_Band = Lift(liftMethod, X_train, X_test, target_mapping, mapped_CI_Upper_Band, knn)[0]

                    paperText = PaperizePreds(Lifted_X_Preds, X_test) # Lifted_X_CI_Lower_Band, Lifted_X_CI_Upper_Band,

                    statsList.append([modelID, paperText])  # , pnl_sh

        statsDF = pd.DataFrame(statsList, columns=["modelID", "paperText"]).set_index("modelID", drop=True)
        statsDF.to_excel(label + "_" + mode.replace(",", "_")+".xlsx")

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
    ######################################################
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
                                'liftMethod': liftMethod,
                                'trainSetLength': trainSetLength,
                                'rolling_Embed_Memory': rolling_Embed_Memory,
                                'target_intrinsic_dim': target_intrinsic_dim, 'mode': mode,
                                'rolling_Predict_Memory': rolling_Predict_Memory,
                                'kernelIn':kernelIn, 'degreeIn':degreeIn, 'neighborsIn':neighborsIn, 'epsilonIn':epsilonIn
                            }
                            processList.append(params)

        p = mp.Pool(mp.cpu_count())
        result = p.map(RollingRunProcess, tqdm(processList))
        p.close()
        p.join()
    elif mode == 'RollingBuilder':
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
        PaperTextList = []
        for file_name in glob.glob('D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\EEG Benchmark\RollingRunners/*.p'):
            print(file_name)
            Preds = pickle.load(open(file_name, "rb"))
            #print(out)
            paperText = PaperizePreds(Preds, X_test)
            PaperTextList.append([file_name.split("\\")[-1], paperText])
            
        PaperTextDF = pd.DataFrame(PaperTextList, columns=["file_name", "paperText"])
        PaperTextDF.to_excel("RollingBuilder.xlsx")
            
def Test():
    pass

if __name__ == '__main__':

    #PyRun()

    label = "EEGdatanew" # "EEGdatanew", "EEGdataNonLinear", "FxDataAdjRets"

    #RunPythonDM(label, "StaticFullEigvalPlot")
    #RunPythonDM(label, "FullModel_Static,VAR")
    RunPythonDM(label, "FullModel_Static,GPR")
    #RunPythonDM(label, "FullModel_Rolling,VAR")
    #RunPythonDM(label, "FullModel_Rolling,GPR")
    #RunPythonDM(label, "Static_run,VAR")
    #RunPythonDM(label, "Static_run,GPR")
    #RunPythonDM(label, "Static_run,GPR") # ---> Needs Discussion
    #RunPythonDM(label, "Rolling_run,VAR")
    #RunPythonDM(label, "Rolling_run,GPR")
    #RunPythonDM(label, 'RollingBuilder')

    #Test()

    #######################################################################################################################
    #RunWithMatlabData('Run', 'static', "GH", label)
    #RunWithMatlabData('Run', 'static', "LP", label)
    #RunWithMatlabData('Read', 'static', "GH", label)
    #RunWithMatlabData('Read', 'static', "LP", label)
    ########################################################
    #RunWithMatlabData('Run', 'dynamic', "GH", label)
    #RunWithMatlabData('Run', 'dynamic', "LP", label)
    #RunWithMatlabData('Read', 'dynamic', "GH", label)
    #RunWithMatlabData('Read', 'dynamic', "LP", label)
