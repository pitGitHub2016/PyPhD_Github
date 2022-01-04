from Slider import Slider as sl
from functools import reduce
from scipy.linalg import svd
import statsmodels.api as sm
import pylab
import numpy as np, json, time, pickle, glob, copy, shutil
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error
from geopy.distance import geodesic
from tqdm import tqdm
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, DotProduct, PairwiseKernel, RationalQuadratic, RBF, Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn import manifold
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datafold.dynfold import LocalRegressionSelection
import scipy, matplotlib
from scipy.interpolate import RBFInterpolator, NearestNDInterpolator
from numpy import linalg as LA

warnings.filterwarnings('ignore')

mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 200)

pathRawData = "D:\Dropbox\EpidemicModel\dati-regioni\dati-regioni\\"
pathWorkingData = ""
modelDataPath = "Modelling\\"
targetDataColumns = ["data", "ricoverati_con_sintomi", "terapia_intensiva", "totale_ospedalizzati", "isolamento_domiciliare",
                    "totale_positivi", "variazione_totale_positivi", "nuovi_positivi", "dimessi_guariti", "deceduti",
                    "casi_da_sospetto_diagnostico", "casi_da_screening", "totale_casi", "tamponi", "casi_testati"]
WeatherDataPath = "weather_data/data_aggregated\\"

#dataMode = ["Plain", [1,1,1,1]]
dataMode = ["Smoothed", [1, 5, 5, 1]]

totale_positiviDF_Raw = pd.read_excel(pathWorkingData + "totale_positivi.xlsx").set_index('data', drop=True).sort_index()
""""""
totale_positiviDF = pd.read_excel(pathWorkingData + "totale_positivi.xlsx").set_index('data', drop=True).sort_index()
nuovi_positiviDF = pd.read_excel(pathWorkingData + "nuovi_positivi.xlsx").set_index('data', drop=True).sort_index()
"TAKE DIFFS for recovered and dead"
dimessi_guaritiDF = pd.read_excel(pathWorkingData + "dimessi_guariti.xlsx").set_index('data', drop=True).sort_index()
decedutiDF = pd.read_excel(pathWorkingData + "deceduti.xlsx").set_index('data', drop=True).sort_index()

if dataMode[0] == "Smoothed":
    "Smooth up with Simple Moving Average"
    #totale_positiviDF = totale_positiviDF.rolling(window=dataMode[1][0]).mean().ffill().fillna(0)
    nuovi_positiviDF = nuovi_positiviDF.rolling(window=dataMode[1][1]).mean().ffill().fillna(0)
    dimessi_guaritiDF = dimessi_guaritiDF.rolling(window=dataMode[1][2]).mean().ffill().fillna(0)
    decedutiDF = decedutiDF.rolling(window=dataMode[1][3]).mean().ffill().fillna(0)

GeoNeighbors = pd.read_excel(pathWorkingData + "GeoNeighborsDF.xlsx")
DataRegionsMappingDF = pd.read_excel(WeatherDataPath+"city2region.xlsx")
region_keys_df = pd.read_excel(f'.\\region2keys.xlsx', index_col='Region')

def rmse(Preds, Real):
    out = np.round(np.sqrt((1 / Real.shape[0]) * np.sum((Preds - Real) ** 2)), 4)
    return out

def reframeData(mainDF, reframeStep, varSelect, **kwargs):
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

    if frameConstructor == "ascending":
        looperRange = range(reframeStep + 1)
    elif frameConstructor == "descending":
        looperRange = range(reframeStep, -1, -1)
    elif frameConstructor == "specific":
        looperRange = [int(x) for x in reframeStep.split(",")]

    baseDF = pd.DataFrame(mainDF.values, index=mainDF.index)

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

    if "ExternalData" in kwargs:
        ExternalData = kwargs["ExternalData"]
        ExternalMainDF = ExternalData[0]
        ExternalData_Shifts = ExternalData[1]
        ExternalDataList = []
        for j in ExternalData_Shifts:
            subExternalDF = ExternalMainDF.shift(j)
            subExternalDF.columns = ["delay_external" + str(j) + "_" + str(x) for x in subExternalDF.columns]
            ExternalDataList.append(subExternalDF)

        ExternalDF = pd.concat(ExternalDataList, axis=1)

        df_List.append(ExternalDF)

    df = pd.concat(df_List, axis=1).dropna()

    if varSelect == "all":
        Y_DF = df.loc[:, [x for x in df.columns if "base_" in x]]
    else:
        Y_DF = df.loc[:, "base_" + str(varSelect)]
    X_DF = df.loc[:, [x for x in df.columns if "delay_" in x]]

    baseList = [x for x in df.columns if "base_" in x]
    if len(baseList) == 1:
        lastY_test_point = X_DF.loc[df.index[-1]]
    else:
        lastY_test_point = df.loc[df.index[-1], baseList]

    X_all_gpr = X_DF.values
    if isinstance(Y_DF, pd.Series) == 1:
        Y_all_gpr = Y_DF.values.reshape(-1, 1)
    else:
        Y_all_gpr = Y_DF.values

    lastY_test_point_gpr = lastY_test_point.values.reshape(1, -1)

    return [X_all_gpr, Y_all_gpr, lastY_test_point_gpr, [X_DF.index, Y_DF.index]]

def generateWeatherDF(targetWeatherRegions):
    dfList = []
    for fileIn in tqdm(glob.glob(WeatherDataPath + "*.csv")):
        fileWeatherRegion = fileIn.split("\\")[-1].replace('.csv', '').split('_')[0]
        if fileWeatherRegion in targetWeatherRegions:
            #print(fileIn)
            subDF = pd.read_csv(fileIn)
            subDF['day'] = subDF['DATA'].str.split('/').str[0]
            subDF['month'] = subDF['DATA'].str.split('/').str[1]
            subDF['year'] = subDF['DATA'].str.split('/').str[2]
            subDF['data'] = pd.to_datetime(subDF['year'] + '-' + subDF['month'] + '-' + subDF['day'], yearfirst=True).astype(str)
            #subDF = subDF.drop(['LOCALITA', 'DATA', 'RAFFICA km/h', 'PRESSIONEMEDIA mb', 'PIOGGIA mm', 'FENOMENI', "day", "month", "year"], axis=1)
            subDF = subDF[["data", "TMEDIA °C", "UMIDITA %"]]
            subDF = subDF.set_index('data', drop=True)
            subDF.columns = [x+"_"+fileWeatherRegion for x in subDF.columns]
            dfList.append(subDF)
    dfOut = pd.concat(dfList, axis=1)
    return dfOut

def DatasetBuilder(mode):

    if mode == 'Raw':
        dfList = []
        for fileIn in tqdm(glob.glob(pathRawData + "*.csv")):
            subDF = pd.read_csv(fileIn)
            dfList.append(subDF)
        dfAll_Raw = pd.concat(dfList, axis=0).drop_duplicates()
        dfAll_Raw['data'] = dfAll_Raw['data'].astype(str).str.split('T').str[0]
        dfAll_Raw.to_excel(pathWorkingData+"DataAll.xlsx", index=False)

    elif mode == 'PerRegion':
        dfAll_Raw = pd.read_excel(pathWorkingData+"DataAll.xlsx")
        UniqueRegions = set(dfAll_Raw['denominazione_regione'].tolist())
        print("#UniqueRegions = ", len(UniqueRegions))
        regionsDataList = []
        for region in UniqueRegions:
            print(region)
            try:
                regionDataDF = dfAll_Raw[dfAll_Raw['denominazione_regione'] == region][targetDataColumns].set_index('data', drop=True)
                #print(regionDataDF.groupby(regionDataDF['data'])["data"].count().sort_values(ascending=False)) # Check duplicates!
                regionDataDF = regionDataDF[~regionDataDF.index.duplicated(keep='last')]
                regionDataDF.columns = [x+"_"+region for x in regionDataDF.columns]
                regionDataDF.to_excel(pathWorkingData+"PerRegionTimeSeries\\"+region+".xlsx")
                regionsDataList.append(regionDataDF)
            except Exception as e:
                print(e)
        dfAll = pd.concat(regionsDataList, axis=1)
        #print(dfAll)
        dfAll.to_excel(pathWorkingData + "DataRegionsTimeSeries.xlsx")

        #dfAll_toMatlab = dfAll[[x for x in dfAll.columns if ('totale_positivi' in x or 'totale_ospedalizzati' in x or 'nuovi_positivi' in x or 'dimessi_guariti' in x or 'deceduti' in x or 'totale_casi' in x or 'tamponi' in x or 'terapia_intensiva' in x or 'ricoverati_con_sintomi' in x) & ('variazione' not in x)]]
        dfAll_toMatlab = dfAll[[x for x in dfAll.columns if 'totale_positivi' in x or 'nuovi_positivi' in x or 'dimessi_guariti' in x or 'deceduti' in x]]
        dfAll[[x for x in dfAll.columns if 'dimessi_guariti' in x or 'deceduti' in x]] = dfAll[[x for x in dfAll.columns if 'dimessi_guariti' in x or 'deceduti' in x]].diff().fillna(0)
        #dfAll_toMatlab[[x for x in dfAll_toMatlab.columns if 'totale_casi' in x or 'tamponi' in x or 'dimessi_guariti' in x or 'deceduti' in x]] = dfAll_toMatlab[[x for x in dfAll_toMatlab.columns if 'totale_casi' in x or 'tamponi' in x or 'dimessi_guariti' in x or 'deceduti' in x]].diff().fillna(0)
        dfAll_toMatlab.to_excel(pathWorkingData + "dfAll_toMatlab.xlsx")
        print(dfAll_toMatlab.columns.get_loc('totale_positivi_Lombardia'))

        totale_positivi_Cols = [x for x in dfAll.columns if 'totale_positivi' in x and 'variazione' not in x]
        dfAll[totale_positivi_Cols].to_excel(pathWorkingData + "totale_positivi.xlsx")
        dfAll[totale_positivi_Cols].iloc[-100:,:].to_excel(pathWorkingData + "totale_positivi_latest.xlsx")
        nuovi_positivi_Cols = [x for x in dfAll.columns if 'nuovi_positivi' in x and 'variazione' not in x]
        dfAll[nuovi_positivi_Cols].to_excel(pathWorkingData + "nuovi_positivi.xlsx")
        dimessi_guariti_Cols = [x for x in dfAll.columns if 'dimessi_guariti' in x and 'variazione' not in x]
        dfAll[dimessi_guariti_Cols].to_excel(pathWorkingData + "dimessi_guariti.xlsx")
        deceduti_Cols = [x for x in dfAll.columns if 'deceduti' in x and 'variazione' not in x]
        dfAll[deceduti_Cols].to_excel(pathWorkingData + "deceduti.xlsx")

def GeoLocationNeighbors(mode, **kwargs):

    if 'kGeo' in kwargs:
        kGeo = kwargs['kGeo']
    else:
        kGeo = 3

    if mode == 'SetupCoordinates':
        dfAll_Raw = pd.read_excel(pathWorkingData + "DataAll.xlsx")
        UniqueRegions = set(dfAll_Raw['denominazione_regione'].tolist())
        regionsDataList = []
        for region in UniqueRegions:
            print(region)
            try:
                regionDataDF = dfAll_Raw[dfAll_Raw['denominazione_regione'] == region][["denominazione_regione", "lat", "long"]].iloc[-1]
                regionsDataList.append(regionDataDF)
            except Exception as e:
                print(e)

        dfAll = pd.concat(regionsDataList, axis=1).T.set_index("denominazione_regione", drop=True)
        dfAll.to_excel(pathWorkingData + "GeoLocations.xlsx")

    elif mode == 'MapNeighbors':
        dfAll = pd.read_excel(pathWorkingData + "GeoLocations.xlsx").set_index("denominazione_regione", drop=True)
        distPairsList = []
        for reg1 in dfAll.index:
            for reg2 in dfAll.index:
                reg1_Coordinates = tuple(map(tuple, dfAll[dfAll.index == reg1].values))[0]
                reg2_Coordinates = tuple(map(tuple, dfAll[dfAll.index == reg2].values))[0]
                geodesicDistance = geodesic(reg1_Coordinates, reg2_Coordinates).kilometers
                distPairsList.append([reg1, reg2, geodesicDistance])
        GeoDistancesMatDF = pd.DataFrame(distPairsList, columns=['Region1', 'Region2', 'Distance']).set_index('Region1', drop=True)
        GeoDistancesMatDF.to_excel(pathWorkingData + "GeoDistancesMatDF.xlsx")

    elif mode == 'KGeoNeighborsSelect':
        GeoDistancesMatDF = pd.read_excel(pathWorkingData + "GeoDistancesMatDF.xlsx").set_index("Region1", drop=True)
        "Select 'kGeo' closest geo-neighbors"
        GeoNeighborsDFList = []
        for reg1 in (set(list(GeoDistancesMatDF.index))):
            subDFGeo = GeoDistancesMatDF[GeoDistancesMatDF.index == reg1].sort_values(by='Distance', ascending=True).iloc[:kGeo+1]
            GeoNeighborsDFList.append([reg1, ','.join(subDFGeo['Region2'].tolist()), ','.join([str(np.round(x,2)) for x in subDFGeo['Distance'].tolist()])])
        GeoNeighborsDF = pd.DataFrame(GeoNeighborsDFList, columns=['Region', 'GeoNeighborsList', 'Distances(km)']).set_index('Region', drop=True)
        GeoNeighborsDF.to_excel(pathWorkingData + "GeoNeighborsDF.xlsx")

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
        X_pcm.optimize_parameters(random_state=0, k=dm_optParams_knn)
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

    if method == 'GH':
        pcm = pfold.PCManifold(eig_trainingSet)
        pcm.optimize_parameters(random_state=0, k=lift_optParams_knn)
        if GH_epsilon == "opt":
            GH_epsilon = pcm.kernel.epsilon
        if GH_cut_off == "opt":
            GH_cut_off = pcm.cut_off
        #opt_n_eigenpairs = eig_trainingSet.shape[0]-1
        opt_n_eigenpairs = eig_trainingSet.shape[1] # Official (Paper)
        gh_interpolant_psi_to_X = GHI(pfold.GaussianKernel(epsilon=GH_epsilon),
                                      n_eigenpairs=opt_n_eigenpairs, dist_kwargs=dict(cut_off=GH_cut_off))
        print("fit ... ")
        gh_interpolant_psi_to_X.fit(eig_trainingSet, X_trainingSet)
        residual = gh_interpolant_psi_to_X.score(eig_trainingSet, X_trainingSet)
        print("predict ... ")
        extrapolatedPsi_to_X = gh_interpolant_psi_to_X.predict(eig_Simulation)
        print("extrapolatedPsi_to_X.shape = ", extrapolatedPsi_to_X.shape)
        # print("opt_epsilon = ", opt_epsilon)
        # print("opt_cutoff = ", opt_cutoff)

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
        mainKernel_Kriging_GP = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * Matern() + 1 * WhiteKernel()  # Official (29/8/2021)
        gpr_model = GaussianProcessRegressor(kernel=mainKernel_Kriging_GP, normalize_y=True)
        gpr_model_fit = gpr_model.fit(eig_trainingSet, X_trainingSet)
        residual = gpr_model_fit.score(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = gpr_model_fit.predict(eig_Simulation)
    elif method == 'SI':  # Simple Linear ND Interpolator
        knn_interpolator = NearestNDInterpolator(eig_trainingSet, X_trainingSet)
        extrapolatedPsi_to_X = knn_interpolator(eig_Simulation)
        residual = extrapolatedPsi_to_X - X_testSet
    elif method == "RBF":
        print("lift_optParams_knn = ", lift_optParams_knn)
        extrapolatedPsi_to_X = RBFInterpolator(eig_trainingSet, X_trainingSet, kernel="linear", degree=1, neighbors=lift_optParams_knn, epsilon=1)(eig_Simulation)
        residual = extrapolatedPsi_to_X - X_testSet

    return [extrapolatedPsi_to_X, residual]

def Model(Settings):

    def getShifts(DFIn, lagsIn):
        outList = []
        for lag in lagsIn:
            subDF = pd.DataFrame(DFIn.copy().shift(lag))
            subDF.columns = [x+"_shift_"+str(lag) for x in subDF.columns]
            outList.append(subDF)
        out_shifted_DF = pd.concat(outList, axis=1).sort_index()
        return out_shifted_DF

    def RunIterativePredict(FeaturesDFin, currentStep):

        FeaturesDF = FeaturesDFin.copy()
        FeaturesDF_raw = FeaturesDF.copy()

        if Settings["Normalise"] == "Yes":
            for col in FeaturesDF.columns:
                FeaturesDF[col] = (FeaturesDF[col]-FeaturesDF_raw[col].mean()) / FeaturesDF_raw[col].std()

        if Settings["Regressor"] == "NN":
            nn_options = {  # options for neural network
                'hidden_layer_sizes': (2, 1),
                'solver': 'lbfgs',
                'activation': 'tanh',
                'max_iter': 1500,  # default 200
                'alpha': 0.001,  # default 0.0001
                'random_state': None  # default None
            }
            model = MLPRegressor(**nn_options)
        elif Settings["Regressor"] == "GPR":
            mainRolling_kernel = ConstantKernel() + Matern() + DotProduct() + WhiteKernel()# + PairwiseKernel() + RBF() + ExpSineSquared()
            #mainRolling_kernel = 1**2*ConstantKernel() + 1**2*Matern() + 1**2*DotProduct() + 1**2* ExpSineSquared() + 1**2*WhiteKernel() # + PairwiseKernel() + RBF() + ExpSineSquared()
            model = GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2)#, normalize_y=True
        elif Settings["Regressor"] == "LSTM":
            model = Sequential()
            model.add(LSTM(7, input_shape=(1, FeaturesDF.shape[1])))
            model.add(Dense(1))
            model.compile(loss='mean_squared_error', optimizer='adam')

        iterPredsList = []
        iterPreds_Std_List = []

        if "SingleStepPredict" in Settings["Reporter"]:
            fitInputX = FeaturesDF.shift(1).bfill().values
            fitTargetY = FeaturesDF[targetVarName].values.reshape(-1, 1)

            if Settings["Regressor"] == "LSTM":
                fitInputX = fitInputX.reshape(trainX, (1, FeaturesDF.shape[1]))
                for i in range(10):
                    model.fit(fitInputX, fitTargetY, epochs=1, batch_size=5, verbose=0, shuffle=False)
                    model.reset_states()
            else:
                model.fit(fitInputX, fitTargetY)

            if Settings['Regressor'] in ["NN", "LSTM"]:
                firstPred = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1))[0]
                firstPred_Std = 0
            elif Settings['Regressor'] == "GPR":
                firstPred, firstPred_Std = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1), return_std=True)
                firstPred = firstPred[0][0]
                firstPred_Std = firstPred_Std[0]
            iterPredsList.append(firstPred)
            iterPreds_Std_List.append(firstPred_Std)

        elif "Iterative" in Settings["Reporter"]:
            "Iterative Predictions"
            inputDataList_rep = []
            for j in range(Settings["predictAhead"] - 1):
                if j == 0:
                    fitInputX = FeaturesDF.shift(1).bfill().values
                    fitTargetY = FeaturesDF[targetVarName].values.reshape(-1, 1)

                    if Settings["Regressor"] == "LSTM":
                        fitInputX = np.reshape(fitInputX, (1, FeaturesDF.shape[1]))
                        for i in range(10):
                            model.fit(fitInputX, fitTargetY, epochs=1, batch_size=1, verbose=0, shuffle=False)
                            model.reset_states()
                    else:
                        model.fit(fitInputX, fitTargetY)

                    if Settings['Regressor'] in ["NN", "LSTM"]:
                        firstPred = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1))[0]
                        firstPred_Std = 0
                    elif Settings['Regressor'] == "GPR":
                        firstPred, firstPred_Std = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1), return_std=True)
                        #print(firstPred)
                        #print(firstPred_Std)
                        #time.sleep(3000)
                        firstPred = firstPred[0][0]
                        firstPred_Std = firstPred_Std[0]
                    iterPredsList.append(firstPred)
                    iterPreds_Std_List.append(firstPred_Std)

                expanding_infectedDF = infectedDF.copy().iloc[:currentStep+j+1]
                newDate = expanding_infectedDF.index[-1]
                knownWeather = allWeatherDF.loc[newDate].values
                knownMobility = mobility_df.loc[newDate]
                if Settings["Normalise"] == "Yes":
                    invertNormIterPreds = [x*FeaturesDF_raw[targetVarName].std()+FeaturesDF_raw[targetVarName].mean() for x in iterPredsList]
                else:
                    invertNormIterPreds = iterPredsList
                expanding_infectedDF.iloc[-len(iterPredsList):] = np.array(invertNormIterPreds)
                expanding_infectedDF_shifted = getShifts(expanding_infectedDF, Settings['lags'])

                if Settings["Scenario"] <= 1:
                    inputDataList = [invertNormIterPreds[-1]]
                    for elem in expanding_infectedDF_shifted.iloc[-1]:
                        inputDataList.append(elem)
                    for elem in knownWeather:
                        inputDataList.append(elem)
                    inputDataList.append(knownMobility)
                elif Settings["Scenario"] in [2, 3]:
                    inputDataList = [invertNormIterPreds[-1]]
                    for elem in expanding_infectedDF_shifted.iloc[-1]:
                        inputDataList.append(elem)
                elif Settings["Scenario"] == 4:
                    inputDataList = [invertNormIterPreds[-1]]
                    inputDataList.append(knownMobility)
                elif Settings["Scenario"] == 5:
                    inputDataList = [invertNormIterPreds[-1]]
                    for elem in knownWeather:
                        inputDataList.append(elem)

                inputDataList_rep.append([newDate, str(inputDataList)])

                if Settings["Normalise"] == "Yes":
                    for colCount in range(len(FeaturesDF_raw.columns)):
                        inputDataList[colCount] = (inputDataList[colCount]-FeaturesDF_raw.iloc[:,colCount].mean())/FeaturesDF_raw.iloc[:,colCount].std()
                if Settings["Regressor"] == "NN":
                    inputPointArray = np.array(inputDataList)
                    iterPred = model.predict(inputPointArray.reshape(1, -1))[0]
                    iterPred_std = 0
                else:
                    inputPointArray = np.array(inputDataList)
                    iterPred, iterPred_std = model.predict(inputPointArray.reshape(1, -1), return_std=True)
                    iterPred = iterPred[0][0]
                    iterPred_std = iterPred_std[0]

                iterPredsList.append(iterPred)
                iterPreds_Std_List.append(iterPred_std)

        iterPredsList.insert(0, FeaturesDF_raw.index[-1])
        iterPreds_Std_List.insert(0, FeaturesDF_raw.index[-1])
        if Settings["Normalise"] == "Yes":
            "standard normalisation"
            iterPredsList[1:] = [x * FeaturesDF_raw[targetVarName].std() + FeaturesDF_raw[targetVarName].mean() for x in iterPredsList[1:]]
            iterPreds_Std_List[1:] = [x * FeaturesDF_raw[targetVarName].std() for x in iterPreds_Std_List[1:]]

        if (Settings["Scenario"]==1)&(RegionName in ["Campania", "Lombardia"]):
            pd.concat([expanding_infectedDF, infectedDF.loc[expanding_infectedDF.index], allWeatherDF.loc[expanding_infectedDF.index], mobility_df.loc[expanding_infectedDF.index], pd.DataFrame(inputDataList_rep, columns=['data', 'inputs']).set_index('data', )], axis=1)\
                .to_excel(modelDataPath+str(Settings["Scenario"])+RegionName+"_expanding_infectedDF.xlsx")

        return [iterPredsList, iterPreds_Std_List]

    ###################################################################################################################

    ModelName = '_'.join([str(x) for x in list(Settings.values())[:-1]])
    RegionName = Settings["region"]

    targetVarName = "totale_positivi_"+RegionName
    infectedDF = totale_positiviDF.copy()[targetVarName]
    #newCasesDF = nuovi_positiviDF.copy()[targetVarName]

    "Weather Data"
    correspondingWeatherRegions = list(
        DataRegionsMappingDF[DataRegionsMappingDF["Covid Region"] == RegionName]['Weather Region'].values)
    allWeatherDF = generateWeatherDF(correspondingWeatherRegions).ffill().rolling(
        window=dataMode[1][1]).mean().interpolate()

    "Mobility Data"
    mobility_df = pd.read_csv('.\\IT_Region_Mobility_Report.csv')
    mobility_df["data"] = mobility_df["date"]
    mobility_df = mobility_df.set_index('data')
    region_mobility_code = region_keys_df.loc[RegionName, 'Mobility code']
    mobility_df = mobility_df[mobility_df['iso_3166_2_code'] == region_mobility_code]
    mobility_df = mobility_df["workplaces_percent_change_from_baseline"].rolling(window=dataMode[1][2]).mean().interpolate()

    #####################
    preds_list = []
    preds_std_list = []

    if "Rolling" in Settings["Reporter"]:
        for i in tqdm(range(Settings["LearningMemory"], Settings["LearningMemory"] + Settings["WindowsToRun"], Settings["RollStep"])):

            if "SingleRegion" in Settings["Reporter"]:
                #roll_train_infected_DF = infectedDF.iloc[0:i]
                roll_train_infected_DF = infectedDF.iloc[i-Settings["LearningMemory"]:i]
            elif "AllRegions" in Settings["Reporter"]:
                roll_train_infected_DF = totale_positiviDF.iloc[i-Settings["LearningMemory"]:i]
            roll_train_infected_DF_shifted = getShifts(roll_train_infected_DF, Settings['lags'])
            roll_train_weather_DF = allWeatherDF.copy().loc[roll_train_infected_DF.index]
            roll_train_mobility_DF = mobility_df.copy().loc[roll_train_infected_DF.index]

            if Settings["Scenario"] <= 1:
                roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_infected_DF_shifted, roll_train_weather_DF, roll_train_mobility_DF], axis=1).sort_index().bfill()
            elif Settings["Scenario"] in [2, 3]:
                roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_infected_DF_shifted], axis=1).sort_index().bfill()
            elif Settings["Scenario"] == 4:
                roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_mobility_DF], axis=1).sort_index().bfill()
            elif Settings["Scenario"] == 5:
                roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_weather_DF], axis=1).sort_index().bfill()

            if Settings["Regressor"] in ["GPR", "LSTM"]:
                out = RunIterativePredict(roll_train_FeaturesDF.copy(), i)
                preds_list.append(out[0])
                preds_std_list.append(out[1])
            else:
                preds_list_DF_runners = []
                for run in tqdm(range(500)):
                    out = RunIterativePredict(roll_train_FeaturesDF.copy())
                    out_pred_data = out[0]
                    preds_list_DF = pd.DataFrame(out_pred_data[1:])
                    preds_list_DF_runners.append(preds_list_DF)
                preds_list_DF_runners_All = pd.concat(preds_list_DF_runners, axis=1)
                preds_list_DF_runners_All_mean = preds_list_DF_runners_All.mean(axis=1)
                preds_list_DF_runners_All_std = preds_list_DF_runners_All.std(axis=1)

                sub_preds_list = preds_list_DF_runners_All_mean.tolist()
                sub_preds_list.insert(0, out_pred_data[0])
                sub_preds_std_list = preds_list_DF_runners_All_std.tolist()
                sub_preds_std_list.insert(0, out_pred_data[0])

                preds_list.append(sub_preds_list)
                preds_std_list.append(sub_preds_std_list)
    elif "MultiStepRegress" in Settings["Reporter"]:
        y_DF = getShifts(infectedDF, [x for x in reversed(range(Settings['predictAhead']))]).bfill()
        xShifts = [x+Settings['predictAhead'] for x in [0,1,3,5,7,10,25,50]]
        x_DF = getShifts(infectedDF, xShifts).bfill()
        trainX = x_DF.iloc[:round(x_DF.shape[0] * 0.7)]
        trainY = y_DF.iloc[:round(y_DF.shape[0] * 0.7)]
        print(trainX.shape, trainY.shape)
        testX = x_DF.loc[x_DF.index.difference(trainX.index)]
        testY = y_DF.loc[testX.index]
        print(testX.shape)
        mainRolling_kernel = ConstantKernel() + Matern() + ExpSineSquared() + WhiteKernel()
        model = GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0,
                                         n_restarts_optimizer=2)  # , normalize_y=True
        model.fit(trainX.values, trainY.values)
        Pred, Pred_Std = model.predict(testX.values, return_std=True)
        Pred_DF = pd.DataFrame(Pred, index=testY.index)
        testY.plot()
        Pred_DF.plot()
        plt.show()
        time.sleep(3000)

    ####################################### WRITE PICKLES #######################################
    mydirParent = Settings["mydir"]
    mydir = mydirParent + 'figures\\'
    if not os.path.exists(mydir):
        os.makedirs(mydir)
    pickle.dump(preds_list, open(mydirParent + ModelName + "_preds_list.p", "wb"))
    pickle.dump(preds_std_list, open(mydirParent + ModelName + "_preds_std_list.p", "wb"))

def Report(Settings):

    def iterPredListParser(predList):
        iterativePredictionsList = []
        for elem in predList:
            date = elem[0]
            dateIdxNum = realDF.index.get_loc(date)
            iterativePredictions = np.array(elem[1:])
            Data = pd.DataFrame(realDF.iloc[dateIdxNum + 1:dateIdxNum + iterativePredictions.shape[0] + 1])
            Data['temp'] = np.nan
            Data['temp'] = iterativePredictions[0:len(Data)]
            iterativePredictionsList.append(Data['temp'])

            out = pd.concat(iterativePredictionsList, axis=1)
        return [out, dateIdxNum]

    ModelName = '_'.join([str(x) for x in list(Settings.values())[:-1]])
    mydir = Settings["mydir"]

    if "Rolling" in Settings["Reporter"]:
        preds_list = pickle.load(open(mydir + ModelName+"_preds_list.p", "rb"))
        preds_std_list = pickle.load(open(mydir + ModelName+"_preds_std_list.p", "rb"))
        realDF = pd.DataFrame(totale_positiviDF_Raw["totale_positivi_"+Settings['region']])
        realDF_smoothed = totale_positiviDF["totale_positivi_"+Settings['region']]
        realDF_smoothed.name = realDF_smoothed.name + " (avg " + str(dataMode[1][0]) + ")"

        dataOut = iterPredListParser(preds_list)
        subPredsDF = pd.DataFrame(dataOut[0])
        subPredsDF.columns = [Settings["Regressor"] + " Prediction "+str(x) for x in range(len(subPredsDF.columns))]
        dataOut_std = iterPredListParser(preds_std_list)
        subPredsDF_std = dataOut_std[0]
        subPredsDF_std.columns = [Settings["Regressor"] + " Prediction "+str(x) for x in range(len(subPredsDF_std.columns))]

        officialPredsDF = subPredsDF.copy()
        stdScale = 1.96
        for colCount in range(len(subPredsDF.columns)):
            officialPredsDF["Upper"+str(colCount)] = subPredsDF.iloc[:, colCount] + stdScale * subPredsDF_std.iloc[:, colCount]
            officialPredsDF["Lower"+str(colCount)] = subPredsDF.iloc[:, colCount] - stdScale * subPredsDF_std.iloc[:, colCount]

        concatDF = pd.concat([realDF, realDF_smoothed, officialPredsDF], axis=1).sort_index().loc[:subPredsDF.index[-1], :]
        realCasesDF = concatDF["totale_positivi_"+Settings['region']]

        PredictionsDF = concatDF[[x for x in concatDF.columns if "Prediction" in x]]

        residsList = []
        for predCol in PredictionsDF.columns:
            sub_residsDF = pd.concat([realCasesDF, PredictionsDF[predCol]], axis=1).dropna()
            residsList.append(pd.Series(((sub_residsDF.iloc[:,1] - sub_residsDF.iloc[:,0])/sub_residsDF.iloc[:,0]).tolist()[0]))#
        residsDF = pd.concat(residsList)
        residsValues = residsDF.values
        Z_residsDF = (residsDF - residsDF.mean()) / residsDF.std()
        #print(Z_residsDF)
        #time.sleep(3000)
        #rmseElem = rmse(residsDF.iloc[:,1], residsDF.iloc[:,0])
        #dwElem = durbin_watson(residsValues)
        jbElem = jarque_bera(residsValues)
        #acLB = acorr_ljungbox(residsValues)

        #fig0, ax0 = plt.subplots(figsize=(35,20))
        #scipy.stats.probplot(Z_residsDF.values, dist="norm", plot=matplotlib.pyplot)
        fig0 = sm.qqplot(Z_residsDF.values, line='45')
        plt.savefig(mydir + "figures\\" + Settings['region'] + '_ResidualsQQplot.png')

        firstPredDF = concatDF[[Settings["Regressor"] + " Prediction 0", 'Upper0', 'Lower0']].reset_index().dropna()

        fig, ax = plt.subplots(figsize=(35,20))
        realCasesDF.plot(ax=ax, color="black", linewidth=3)
        plt.legend(prop={"size": 12})
        concatDF[realDF_smoothed.name].plot(ax=ax, color="green", linewidth=3)
        plt.legend(prop={"size": 12})
        PredictionsDF.plot(ax=ax, color="blue", linewidth=2, legend=None)
        concatDF[[x for x in concatDF.columns if "Upper" in x]].plot(ax=ax, color="red", linewidth=1, linestyle="dotted", legend=None)
        concatDF[[x for x in concatDF.columns if "Lower" in x]].plot(ax=ax, color="red", linewidth=1, linestyle="dotted", legend=None)
        plt.fill_between(range(firstPredDF.index[0], firstPredDF.index[-1]+1), concatDF['Upper0'].loc[firstPredDF["data"]], concatDF['Lower0'].loc[firstPredDF["data"]], alpha=0.4)
        plt.axvline(x=firstPredDF.index[0], ymin=-1, ymax=1, color='b', ls='--', lw=1.5, label='axvline - % of full height')
        for label in ax.get_xticklabels():
            label.set_fontsize(20)
            label.set_ha("right")
            label.set_rotation(20)
        plt.grid()
        plt.savefig(mydir + "figures\\" + Settings['region'] + '_Rolling_Plotter.png')

        #calcMetricsDF = concatDF[["totale_positivi_"+Settings['region'], Settings["Regressor"] + " Prediction"]].dropna()
        #pickle.dump(calcMetricsDF, open(mydir + ModelName + "_calcMetricsDF.p", "wb"))

def RunSystem():
    for Regressor in ["GPR"]: #"NN", "GPR", "LSTM"
        for region in region_keys_df.reset_index()['Region'].values:
            if region == "Lombardia":
                for scenario in [2]: #1, 2, 3, 4, 5
                    #try:
                    print(scenario, region)
                    mydir = modelDataPath + "Scenario_" + str(scenario) + Regressor + '\\'

                    Settings = {"ModelSpace": "OriginalSpace",
                                "region": region,
                                "lags": [1, 3, 5, 7],
                                "Scenario": scenario,
                                "WindowsToRun": 350,  # "entireDataset"
                                "LearningMemory": 60, # 60, 90 or 120 days
                                "Regressor": Regressor,  # GPR, NN
                                "RollStep": 1,
                                "predictAhead": 5, # 2*7 = 14 (2 weeks)
                                "Reporter": "Rolling_Iterative_SingleRegion", #Rolling_Iterative_SingleRegion, Rolling_Iterative_AllRegions
                                "Normalise": "Yes", #Yes, No
                                "mydir": mydir}

                    #Model(Settings)
                    Report(Settings)
                    #time.sleep(30000)
                    #except Exception as e:
                    #   print(e)

DatasetBuilder('Raw')
DatasetBuilder('PerRegion')
#GeoLocationNeighbors('SetupCoordinates')
#GeoLocationNeighbors('MapNeighbors')
#GeoLocationNeighbors('KGeoNeighborsSelect', kGeo=4)
#RunSystem()

