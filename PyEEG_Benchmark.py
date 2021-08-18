from Slider import Slider as sl
import scipy.io
import itertools
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
from statsmodels.tsa.api import VAR
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection,
)

lpyr_interpolant_psi_to_X = LPI(auto_adaptive=True)
mainKernel_Kriging_GP = 1 * RBF()
gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP)

def Lift(method, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation):
    if method == 'GH':
        pcm = pfold.PCManifold(eig_trainingSet)
        pcm.optimize_parameters(random_state=1)
        opt_epsilon = pcm.kernel.epsilon
        opt_cutoff = pcm.cut_off
        opt_n_eigenpairs = eig_trainingSet.shape[1]
        gh_interpolant_psi_to_X = GHI(pfold.GaussianKernel(epsilon=opt_epsilon),
                                      n_eigenpairs=opt_n_eigenpairs, dist_kwargs=dict(cut_off=opt_cutoff), )
        gh_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = gh_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gh_interpolant_psi_to_X.predict(eig_Simulation)
    elif method == 'LP':
        lpyr_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = lpyr_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = lpyr_interpolant_psi_to_X.predict(eig_Simulation)
    elif method == 'KR':
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        residual = gpr_model_fit.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gpr_model_fit.predict(eig_Simulation)

    mse = (np.square(extrapolatedPsi_to_X - X_testSet)).mean(axis=None)

    return [extrapolatedPsi_to_X, mse, residual]

def RunWithMatlabData(RunMode, TimeConfiguration, liftMethod):
    if RunMode == 'Run':
        if TimeConfiguration == 'static':
            mat = scipy.io.loadmat('D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Static_DM_Matlab_Data.mat')
            data = mat['Static_DM_Matlab_Data'][0]
            #print(data)
            X_trainingSet = data[0]
            X_testSet = data[1]
            eig_trainingSet = data[2]
            eig_Simulation = data[3]
            MeanSim = data[4]
            eig_Simulation_Lower_Band = data[5]
            eig_Simulation_Upper_Band = data[6]
            CI_manifold = data[7]
            print("X_trainingSet.shape = ", X_trainingSet.shape)
            print("X_testSet.shape = ", X_testSet.shape)
            print("eig_trainingSet.shape = ", eig_trainingSet.shape)
            print("eig_Simulation.shape = ", eig_Simulation.shape)
            print("MeanSim.shape = ", MeanSim.shape)
            print("eig_Simulation_Lower_Band.shape = ", eig_Simulation_Lower_Band.shape)
            print("eig_Simulation_Upper_Band.shape = ", eig_Simulation_Upper_Band.shape)
            print("CI_manifold.shape = ", CI_manifold.shape)

            mse_eig_Simulation_standalone = (np.square(0 - X_testSet)).mean(axis=None)
            print("mse_eig_Simulation_standalone = ", mse_eig_Simulation_standalone)

            for liftMethod in ["GH", "LP", "KR"]:
                lift_out_eig_Simulation = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation)
                mse_eig_Simulation = np.round(lift_out_eig_Simulation[1],4)
                print(liftMethod, ", lift_out_eig_Simulation[0].shape = ", lift_out_eig_Simulation[0].shape, ", mse_eig_Simulation = ", mse_eig_Simulation)

                lift_out_MeanSim = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, MeanSim)
                mse_MeanSim = np.round(lift_out_MeanSim[1],4)
                print(liftMethod, ", ", lift_out_MeanSim[0], ", mse_MeanSim = ", mse_MeanSim)

                lift_out_eig_Simulation_Lower_Band = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation_Lower_Band)
                mse_eig_Simulation_Lower_Band = np.round(lift_out_eig_Simulation_Lower_Band[1],4)
                print(liftMethod, ", ", lift_out_eig_Simulation_Lower_Band[0], ", mse_eig_Simulation_Lower_Band = ", mse_eig_Simulation_Lower_Band)

                lift_out_eig_Simulation_Upper_Band = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation_Upper_Band)
                mse_eig_Simulation_Upper_Band = np.round(lift_out_eig_Simulation_Upper_Band[1],4)
                print(liftMethod, ", ", lift_out_eig_Simulation_Upper_Band[0], ", mse_eig_Simulation_Upper_Band = ", mse_eig_Simulation_Upper_Band)

                lift_out_CI_manifold_Lower_Band = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, CI_manifold[0,:].reshape(1,-1))
                mse_CI_manifold_Lower_Band = np.round(lift_out_CI_manifold_Lower_Band[1],4)
                print(liftMethod, ", ", lift_out_CI_manifold_Lower_Band[0], ", mse_CI_manifold_Lower_Band = ", mse_CI_manifold_Lower_Band)

                lift_out_CI_manifold_Upper_Band = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, CI_manifold[1,:].reshape(1,-1))
                mse_CI_manifold_Upper_Band = np.round(lift_out_CI_manifold_Upper_Band[1],4)
                print(liftMethod, ", ", lift_out_CI_manifold_Upper_Band[0], ", mse_CI_manifold_Upper_Band = ", mse_CI_manifold_Upper_Band)
        else:
            mat = scipy.io.loadmat(
                'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Dynamic_DM_Matlab_Data_TS.mat')
            data = mat['Dynamic_DM_Matlab_Data_TS']

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
                allStatsCombos = [[x[0], x[1]] for x in itertools.product(*varSet)]
                if i == 0:
                    allStatsCombos_Labels = [[x[0]+"-"+x[1]] for x in itertools.product(*varSetLabels)]
                ################################################
                y_i = rowData[5]
                real_x_list.append(y_i[0])

                liftedCombos = [[] for varN in range(len(allStatsCombos))]
                ## Process restricted Data of each combination (lift them back) for EACH stats combo
                countCombo = 0
                for subPred in allStatsCombos:
                    lift_out_Preds = Lift(liftMethod, y_intv, y_i, obj_evecs, np.array(subPred).reshape(1, -1))[0]

                    if liftMethod not in ["GH"]:
                        lift_out_Preds = lift_out_Preds[0]
                    liftedCombos[countCombo].append(lift_out_Preds.tolist())
                    countCombo += 1

                lift_out_Preds_list.append(liftedCombos)

            liftedCombos_TS = [[] for tsCount in range(len(allStatsCombos))]
            MSE_List = []
            for tsCount in range(len(allStatsCombos)):
                for t in range(len(lift_out_Preds_list)):
                    t_sub_lift_out_Preds_list = lift_out_Preds_list[t]
                    liftedCombos_TS[tsCount].append(t_sub_lift_out_Preds_list[tsCount][0])

                extrapolatedPsi_to_X_Preds = np.array(liftedCombos_TS[tsCount])
                X_testSet = np.array(real_x_list)

                mse = (np.square(extrapolatedPsi_to_X_Preds - X_testSet)).mean(axis=None)
                MSE_List.append([allStatsCombos_Labels[tsCount], mse])

            MSE_df = pd.DataFrame(MSE_List, columns=["StatsCombo", "MSE"])
            print(MSE_df)
            pickle.dump(MSE_df, open("MSE_df_" + str(TimeConfiguration) + "_" + liftMethod + ".p", "wb"))
    elif RunMode == 'Read':
        MSE_df = pickle.load(open("MSE_df_" + str(TimeConfiguration) + "_" + liftMethod + ".p", "rb"))

#PyRun()
#RunWithMatlabData('Run', 'static')
RunWithMatlabData('Run', 'dynamic', "GH")
