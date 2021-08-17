from Slider import Slider as sl
import scipy.io
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

def RunWithMatlabData(mode):
    if mode == 'static':
        mat = scipy.io.loadmat('D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Static_DM_Matlab_Data.mat')
        data = mat['Static_DM_Matlab_Data'][0]
        #print(data)
        X_trainingSet = data[0]
        X_testSet = data[1]
        eig_trainingSet = data[2]
        eig_Simulation = data[3]
        CI = data[4]
        print("X_trainingSet.shape = ", X_trainingSet.shape)
        print("X_testSet.shape = ", X_testSet.shape)
        print("eig_trainingSet.shape = ", eig_trainingSet.shape)
        print("eig_Simulation.shape = ", eig_Simulation.shape)
        print("CI.shape = ", CI.shape)

        for liftMethod in ["GH", "LP", "KR"]:
            lift_out = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, eig_Simulation)
            print(liftMethod, ", ", lift_out[0])
            print(liftMethod, ", lift_out[0].shape = ", lift_out[0].shape, ", mse = ", lift_out[1])

            lift_out_CI = Lift(liftMethod, X_trainingSet, X_testSet, eig_trainingSet, CI)
            print(liftMethod, ", ", lift_out_CI[0])
            print(liftMethod, ", lift_out_CI[0].shape = ", lift_out_CI[0].shape, ", mse = ", lift_out_CI[1])

    else:
        mat = scipy.io.loadmat(
            'D:\Dropbox\VM_Backup\RollingManifoldLearning\SmartGlobalAssetAllocation\MatlabCode_EqFree_DMAPs\FinEngineering_Application\Lifting_GG\\Dynamic_DM_Matlab_Data_TS.mat')
        data = mat['Dynamic_DM_Matlab_Data_TS']

        liftMethod = "GH"

        lift_out_Preds_list = []
        lift_out_CI_Lower_Band_list = []
        lift_out_CI_Upper_Band_list = []
        real_x_list = []
        for i in range(data.shape[0]):
            print(i, ", ", data[i].shape)

            rowData = data[i]
            y_intv = rowData[0]
            obj_evecs = rowData[1]
            Preds = rowData[2]
            CI_Lower_Band = rowData[3]
            CI_Upper_Band = rowData[4]
            y_i = rowData[5]
            lift_out_Preds = Lift(liftMethod, y_intv, y_i, obj_evecs, Preds)
            lift_out_CI_Lower_Band = Lift(liftMethod, y_intv, y_i, obj_evecs, CI_Lower_Band)
            lift_out_CI_Upper_Band = Lift(liftMethod, y_intv, y_i, obj_evecs, CI_Upper_Band)
            if liftMethod == "GH":
                lift_out_Preds_list.append(lift_out_Preds)
                lift_out_CI_Lower_Band_list.append(lift_out_CI_Lower_Band)
                lift_out_CI_Upper_Band_list.append(lift_out_CI_Upper_Band)
                real_x_list.append(y_i[0])

        extrapolatedPsi_to_X_Preds = np.array(lift_out_Preds_list)
        extrapolatedPsi_to_X_CI_Lower_Band = np.array(lift_out_CI_Lower_Band_list)
        extrapolatedPsi_to_X_CI_Upper_Band = np.array(lift_out_CI_Upper_Band_list)
        X_testSet = np.array(real_x_list)
        print("extrapolatedPsi_to_X_Preds.shape = ", extrapolatedPsi_to_X_Preds.shape)
        print("extrapolatedPsi_to_X_CI_Lower_Band.shape = ", extrapolatedPsi_to_X_CI_Lower_Band.shape)
        print("extrapolatedPsi_to_X_CI_Upper_Band.shape = ", extrapolatedPsi_to_X_CI_Upper_Band.shape)
        print("X_testSet.shape = ", X_testSet.shape)

        mse = (np.square(extrapolatedPsi_to_X_Preds - X_testSet)).mean(axis=None)
        print("MSE = ", mse)

#PyRun()
#RunWithMatlabData('static')
RunWithMatlabData('dynamic')
