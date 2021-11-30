from Slider import Slider as sl
from functools import reduce
from scipy.linalg import svd
import numpy as np, json, time, pickle, glob, copy, shutil
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
from datafold.dynfold import LocalRegressionSelection
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

pathRawData = "dati-regioni\\"
pathWorkingData = "WorkingDataset\\"
modelDataPath = "WorkingDataset\Modelling\\"
targetDataColumns = ["data", "ricoverati_con_sintomi", "terapia_intensiva", "totale_ospedalizzati", "isolamento_domiciliare",
                    "totale_positivi", "variazione_totale_positivi", "nuovi_positivi", "dimessi_guariti", "deceduti",
                    "casi_da_sospetto_diagnostico", "casi_da_screening", "totale_casi", "tamponi", "casi_testati"]
WeatherDataPath = "WorkingDataset/weather_data/data_aggregated\\"

#dataMode = ["Plain", [1,1,1,1]]
dataMode = ["Smoothed", [5,21,7,1]]

totale_positiviDF_Raw = pd.read_excel(pathWorkingData + "totale_positivi.xlsx").set_index('data', drop=True).sort_index()
""""""
totale_positiviDF = pd.read_excel(pathWorkingData + "totale_positivi.xlsx").set_index('data', drop=True).sort_index()
nuovi_positiviDF = pd.read_excel(pathWorkingData + "nuovi_positivi.xlsx").set_index('data', drop=True).sort_index()
"TAKE DIFFS for recovered and dead"
dimessi_guaritiDF = pd.read_excel(pathWorkingData + "dimessi_guariti.xlsx").set_index('data', drop=True).sort_index().diff()
decedutiDF = pd.read_excel(pathWorkingData + "deceduti.xlsx").set_index('data', drop=True).sort_index().diff()

if dataMode[0] == "Smoothed":
    "Smooth up with Simple Moving Average"
    totale_positiviDF = totale_positiviDF.rolling(window=dataMode[1][0]).mean().ffill().fillna(0)
    nuovi_positiviDF = nuovi_positiviDF.rolling(window=dataMode[1][1]).mean().ffill().fillna(0)
    dimessi_guaritiDF = dimessi_guaritiDF.rolling(window=dataMode[1][2]).mean().ffill().fillna(0)
    decedutiDF = decedutiDF.rolling(window=dataMode[1][3]).mean().ffill().fillna(0)

GeoNeighbors = pd.read_excel(pathWorkingData + "GeoNeighborsDF.xlsx")
DataRegionsMappingDF = pd.read_excel(WeatherDataPath+"city2region.xlsx")
region_keys_df = pd.read_excel(f'.\\WorkingDataset\\region2keys.xlsx', index_col='Region')

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
        print(dfAll)
        dfAll.to_excel(pathWorkingData + "DataRegionsTimeSeries.xlsx")

        totale_positivi_Cols = [x for x in dfAll.columns if 'totale_positivi' in x and 'variazione' not in x]
        dfAll[totale_positivi_Cols].to_excel(pathWorkingData + "totale_positivi.xlsx")
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

    def getShifts(DFIn):
        outList = []
        for lag in Settings['lags']:
            outList.append(DFIn.copy().shift(lag))
        out_shifted_DF = pd.concat(outList, axis=1).sort_index()
        out_shifted_DF.columns = [RegionName + "_" + str(x) for x in Settings['lags']]
        return out_shifted_DF

    def RunPredict(FeaturesDFin):

        FeaturesDF = FeaturesDFin.copy()
        FeaturesDF_raw = FeaturesDF.copy()
        # print(FeaturesDF_raw)

        if Settings["Normalise"] == "Yes":
            for col in FeaturesDF.columns:
                FeaturesDF[col] = (FeaturesDF[col]-FeaturesDF_raw[col].mean()) / FeaturesDF_raw[col].std()

        if Settings["Regressor"] == "NN":
            # todo --> 1 layer, 10 nodes (rule of thump on the unknown, dataLength / 4)
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
            #mainRolling_kernel = Matern() # ConstantKernel() + ExpSineSquared() + RationalQuadratic() + WhiteKernel()
            #mainRolling_kernel = ConstantKernel() + Matern() + ExpSineSquared() + RationalQuadratic() + WhiteKernel()
            mainRolling_kernel = ConstantKernel() + Matern() + ExpSineSquared() + WhiteKernel()
            model = GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2, normalize_y=True)

        "Iterative Predictions"
        iterPredsList = []
        inputDataList_rep = []
        iterPreds_Std_List = []
        for j in range(Settings["predictAhead"] - 1):
            if j == 0:
                fitInputX = FeaturesDF.shift(1).bfill().values
                fitTargetY = FeaturesDF[targetVarName].values.reshape(-1, 1)
                model.fit(fitInputX, fitTargetY)
                if Settings['Regressor'] == "NN":
                    firstPred = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1))[0]
                    firstPred_Std = 0

                elif Settings['Regressor'] == "GPR":
                    firstPred, firstPred_Std = model.predict(FeaturesDF.iloc[-1].values.reshape(1, -1), return_std=True)
                    firstPred = firstPred[0][0]
                    firstPred_Std = firstPred_Std[0]

                iterPredsList.append(firstPred)
                iterPreds_Std_List.append(firstPred_Std)

            expanding_infectedDF = infectedDF.copy().iloc[:i+j+1]
            newDate = expanding_infectedDF.index[-1]
            knownWeather = allWeatherDF.loc[newDate].values
            knownMobility = mobility_df.loc[newDate]
            if Settings["Normalise"] == "Yes":
                invertNormIterPreds = [x*FeaturesDF_raw[targetVarName].std()+FeaturesDF_raw[targetVarName].mean() for x in iterPredsList]
            else:
                invertNormIterPreds = iterPredsList
            expanding_infectedDF.iloc[-len(iterPredsList):] = np.array(invertNormIterPreds)
            expanding_infectedDF_shifted = getShifts(expanding_infectedDF)

            if Settings["Scenario"] == 1:
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
            iterPredsList[1:] = [x * FeaturesDF_raw[targetVarName].std() + FeaturesDF_raw[targetVarName].mean() for x in iterPredsList[1:]]
            iterPreds_Std_List[1:] = [x * FeaturesDF_raw[targetVarName].std() + FeaturesDF_raw[targetVarName].mean() for x in iterPreds_Std_List[1:]]

        if (Settings["Scenario"]==1)&(RegionName in ["Campania", "Lombardia"]):
            pd.concat([expanding_infectedDF, infectedDF.loc[expanding_infectedDF.index], allWeatherDF.loc[expanding_infectedDF.index], mobility_df.loc[expanding_infectedDF.index], pd.DataFrame(inputDataList_rep, columns=['data', 'inputs']).set_index('data', )], axis=1)\
                .to_excel(modelDataPath+str(Settings["Scenario"])+RegionName+"_expanding_infectedDF.xlsx")

        return [iterPredsList, iterPreds_Std_List]

    ###################################################################################################################

    ModelName = '_'.join([str(x) for x in list(Settings.values())[:-1]])
    RegionName = Settings["region"]

    targetVarName = "totale_positivi_"+RegionName
    infectedDF = totale_positiviDF.copy()[targetVarName]

    "Weather Data"
    correspondingWeatherRegions = list(
        DataRegionsMappingDF[DataRegionsMappingDF["Covid Region"] == RegionName]['Weather Region'].values)
    allWeatherDF = generateWeatherDF(correspondingWeatherRegions).ffill().rolling(
        window=dataMode[1][1]).mean().interpolate()#.ffill().fillna(0)

    "Mobility Data"
    mobility_df = pd.read_csv('.\\WorkingDataset\\IT_Region_Mobility_Report.csv')
    mobility_df["data"] = mobility_df["date"]
    mobility_df = mobility_df.set_index('data')
    region_mobility_code = region_keys_df.loc[RegionName, 'Mobility code']
    mobility_df = mobility_df[mobility_df['iso_3166_2_code'] == region_mobility_code]
    mobility_df = mobility_df["workplaces_percent_change_from_baseline"].rolling(window=dataMode[1][2]).mean().interpolate()#.ffill().fillna(0)

    if Settings["WindowsToRun"] == "entireDataset":
        iterationsToRun = infectedDF.shape[0]
    else:
        iterationsToRun = Settings["LearningMemory"] + Settings["WindowsToRun"]

    #####################
    preds_list = []
    preds_std_list = []
    for i in range(Settings["LearningMemory"], iterationsToRun, Settings["RollStep"]):

        #roll_train_infected_DF = infectedDF.iloc[0:i]
        roll_train_infected_DF = infectedDF.iloc[i-Settings["LearningMemory"]:i]
        roll_train_infected_DF_shifted = getShifts(roll_train_infected_DF)
        roll_train_weather_DF = allWeatherDF.copy().loc[roll_train_infected_DF.index]
        roll_train_mobility_DF = mobility_df.copy().loc[roll_train_infected_DF.index]

        if Settings["Scenario"] == 1:
            roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_infected_DF_shifted, roll_train_weather_DF, roll_train_mobility_DF], axis=1).sort_index().bfill()
        elif Settings["Scenario"] in [2, 3]:
            roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_infected_DF_shifted], axis=1).sort_index().bfill()
        elif Settings["Scenario"] == 4:
            roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_mobility_DF], axis=1).sort_index().bfill()
        elif Settings["Scenario"] == 5:
            roll_train_FeaturesDF = pd.concat([roll_train_infected_DF, roll_train_weather_DF], axis=1).sort_index().bfill()

        if Settings["Regressor"] in ["GPR"]:
            out = RunPredict(roll_train_FeaturesDF.copy())
            preds_list.append(out[0])
            preds_std_list.append(out[1])
        else:
            preds_list_DF_runners = []
            for run in tqdm(range(500)):
                out = RunPredict(roll_train_FeaturesDF)
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

    if Settings["Reporter"] == "Iterative_Rolling":
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
        stdScale = 1 #1.96
        for colCount in range(len(subPredsDF.columns)):
            officialPredsDF["Upper"+str(colCount)] = subPredsDF.iloc[:, colCount] + stdScale * subPredsDF_std.iloc[:, colCount]
            officialPredsDF["Lower"+str(colCount)] = subPredsDF.iloc[:, colCount] - stdScale * subPredsDF_std.iloc[:, colCount]

        concatDF = pd.concat([realDF, realDF_smoothed, officialPredsDF], axis=1).sort_index().loc[:subPredsDF.index[-1],:]

        firstPredDF = concatDF[[Settings["Regressor"] + " Prediction 0", 'Upper0', 'Lower0']].reset_index().dropna()

        color_dict = {"totale_positivi_"+Settings['region']: 'black'}
        for colNum in range(len(concatDF.columns)):
            color_dict[Settings["Regressor"] + " Prediction "+str(colNum)] = '#0000FF'
            color_dict['Upper'+str(colNum)] = '#FF0000'
            color_dict['Lower'+str(colNum)] = '#FF0000'

        concatDF.plot(color=[color_dict.get(x, 'green') for x in concatDF.columns])
        plt.legend(prop={"size": 10})
        plt.fill_between(range(firstPredDF.index[0], firstPredDF.index[-1]+1), concatDF['Upper0'].loc[firstPredDF["data"]], concatDF['Lower0'].loc[firstPredDF["data"]], alpha=0.4)

        plt.grid()
        plt.axvline(x=firstPredDF.index[0], ymin=-1, ymax=1, color='b', ls='--', lw=1.5, label='axvline - % of full height')
        plt.savefig(mydir + "figures\\" + Settings['region'] + '_Rolling_Plotter.png')

        #calcMetricsDF = concatDF[["totale_positivi_"+Settings['region'], Settings["Regressor"] + " Prediction"]].dropna()
        #pickle.dump(calcMetricsDF, open(mydir + ModelName + "_calcMetricsDF.p", "wb"))

def RunSystem():
    for Regressor in ["GPR"]: #"NN", "GPR"
        for region in region_keys_df.reset_index()['Region'].values:
            if True:#region == "Campania":
                for scenario in [1]: #1, 2, 3, 4, 5
                    #try:
                    print(scenario, region)
                    mydir = modelDataPath + "Scenario_" + str(scenario) + Regressor + '\\'

                    Settings = {"ModelSpace": "OriginalSpace",
                                "region": region,
                                "lags": [2, 4, 6],
                                "Scenario": scenario,
                                "WindowsToRun": 350,  # "entireDataset"
                                "Regressor": Regressor,  # GPR, NN
                                "LearningMemory": 60, # 60, 90 or 120 days
                                "RollStep": 1,
                                "predictAhead": 7, # 2*7 = 14 (2 weeks)
                                "Reporter": "Iterative_Rolling",
                                "Normalise": "Yes", #Yes, No
                                "mydir": mydir}

                    if Settings["Scenario"] == 3:
                        Settings["lags"] = [2, 4, 6, 8]

                    Model(Settings)
                    Report(Settings)
                    #time.sleep(30000)
                    #except Exception as e:
                    #   print(e)

def MetricsManager():

    reportData = []
    for (dirpath, dirnames, filenames) in os.walk(modelDataPath):
        for filename in filenames:
            if "calcMetricsDF" in filename:
                scenario = dirpath.split("\\")[-1]
                region = filename.split("\\")[-1].split("_")[1]
                subDF = pickle.load(open(os.path.join(dirpath, filename), "rb"))
                y_pred = subDF[[x for x in subDF.columns if "Prediction" in x]].values
                y = subDF[[x for x in subDF.columns if "totale_positivi" in x]].values
                L2_norm_region = np.linalg.norm(y_pred - y, ord=2)
                mse_region = rmse(y_pred, y)
                reportData.append([scenario, region, L2_norm_region, mse_region])
                print(scenario, region)
                print(subDF)
    MetricsManagerDF = pd.DataFrame(reportData, columns=["scenario", "region", "L2", "RMSE"])
    print(MetricsManagerDF)
    MetricsManagerDF.to_excel(modelDataPath+"MetricsManager.xlsx", index=False)

#DatasetBuilder('Raw')
#DatasetBuilder('PerRegion')
#GeoLocationNeighbors('SetupCoordinates')
#GeoLocationNeighbors('MapNeighbors')
#GeoLocationNeighbors('KGeoNeighborsSelect', kGeo=4)
RunSystem()
#MetricsManager()

"""
    NOTES :
    ############
    Settings = {"ModelSpace": "OriginalSpace",
                "RegressFramework": "I(t)=f(I(t-lags),Weather(t-lags),Mobility(t-lags))", #I(t)=f(I(t-lags)), I(t)=f(dI(t-lags)), I(t)=f(DI(t-lags),DR(t-lags),DD(t-lags)), I(t)=f(I(t-lags),Weather(t-lags),Mobility(t-lags))
                "reframeDataConstructor": ["specific", "0,2,4,6"],
                "iterationsToRun" : 1, #"entireDataset"
                "LearningMemory": 366,
                "LearningWindowMode": "Rolling", #Rolling, Expanding
                "RollStep": 1,
                "predictAhead": 30,
                "Reporter": "Iterative_Rolling"}
    ############             
    Model("FullModel", predMode="iterative", scaler="Standard")
    #Model("FullModel", predMode="conditioned") # PP
    
    #####################
    TrainSet_DF = totale_positiviDF.loc[:traintoDate, :]
    
            if predSchemeSplit[1] == "DM":
                X_TrainSet_DF = TrainSet_DF.copy()
                target_mapping_List = Embed("DMComputeParsimonious", TrainSet_DF.values, target_intrinsic_dim,
                                            dm_epsilon="opt", cut_off=np.inf, dm_optParams_knn=dm_optParams_knnIn)
                TrainSet_DF = pd.DataFrame(target_mapping_List[0], index=TrainSet_DF.index)
                TrainSet_DF.columns = ['totale_positivi_DM' + str(x) for x in range(len(TrainSet_DF.columns))]
    
            TestSet_DF = totale_positiviDF.loc[testFromDate:testToDate, :]
            forecastHorizon = len(TestSet_DF)
    
            model_GPR_List = [GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2, normalize_y=True) for var in range(len(TrainSet_DF.columns))]
    
            yWithPredictionsList = [[] for var in range(len(TrainSet_DF.columns))]
    
            Preds_List = []
            for step_i in tqdm(range(forecastHorizon)):
    
                models_preds_list = []
                for modelIn in range(len(model_GPR_List)):
    
                    region_df = TrainSet_DF.iloc[:, modelIn]
                    region_name = region_df.name.replace('totale_positivi_', '')
    
                    if predScheme[0] == "OS":
                        region_GeoNeighbors = ['totale_positivi_' + str(x) for x in GeoNeighbors[GeoNeighbors["Region"] == region_name]['GeoNeighborsList'].values[0].split(',')]
                        region_GeoNeighbors_Pos = [TrainSet_DF.columns.get_loc(x) for x in region_GeoNeighbors]
    
                    if ModelSpace == 'OnlyInfectedSelf':
                        SubSet_DF = region_df.copy()
                    elif ModelSpace == 'InfectedSelfWithMobilityData':
                        SubSet_DF = region_df.copy()
                        mobilityRegionName = MobilityDataNames[MobilityDataNames["Region"] == region_name]["MobilityRegion"].values
                        mobilityRegionData = MobilityData[(MobilityData['sub_region_1'] == mobilityRegionName[0])]
                        mobilityRegionData = mobilityRegionData[mobilityRegionData['sub_region_2'].isna()]
                        mobilityRegionData = mobilityRegionData[['date', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline','transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
                        mobilityRegionData = mobilityRegionData.rename(columns={'date': 'data'})
                        mobilityRegionData = mobilityRegionData.set_index('data', drop=True)
                        #mobilityRegionData = mobilityRegionData / mobilityRegionData.stack().std()
                        mobilityRegionData_TrainSet_DF = pd.DataFrame(mobilityRegionData.loc[SubSet_DF.index].values)
                        mobilityRegionData_TestSet_DF = pd.DataFrame(mobilityRegionData.loc[TestSet_DF.index].values)
                    elif ModelSpace == 'OnlyInfectedGeoNeighbors':
                        SubSet_DF = TrainSet_DF.loc[:, region_GeoNeighbors]
                    elif ModelSpace == 'WithWeatherData':
                        correspondingWeatherRegions = list(DataRegionsMappingDF[DataRegionsMappingDF["Covid Region"] == region_name]['Weather Region'].values)
                        subTrainSet_DF = TrainSet_DF.loc[:, region_GeoNeighbors]
                        allWeatherDF = generateWeatherDF(correspondingWeatherRegions).ffill()
                        trainWeatherDF = allWeatherDF.loc[subTrainSet_DF.index]
                        SubSet_DF = pd.concat([subTrainSet_DF, trainWeatherDF], axis=1)
                        testWeatherDataDF = allWeatherDF.loc[TestSet_DF.index]
    
                    if step_i == 0:
    
                        if ModelSpace == 'InfectedSelfWithMobilityData':
                            reframedData = reframeData(SubSet_DF, reframeData_frameConstructor[1], 0, frameConstructor=reframeData_frameConstructor[0],
                                                       ExternalData=[mobilityRegionData_TrainSet_DF, reframeData_External_Shift])
                        else:
                            reframedData = reframeData(SubSet_DF, reframeData_frameConstructor[1], 0, frameConstructor=reframeData_frameConstructor[0])
    
                        if scaler == "Standard":
                            scX = StandardScaler()
                            scY = StandardScaler()
                            #scX = MinMaxScaler()
                            #scY = MinMaxScaler()
                            reframedDataX = scX.fit_transform(reframedData[0])
                            reframedDataY = scY.fit_transform(reframedData[1])
                        else:
                            reframedDataX = reframedData[0]
                            reframedDataY = reframedData[1]
    
                        "Fit the GPR"
                        model_GPR_List[modelIn].fit(reframedDataX, reframedDataY.reshape(-1, 1))
    
                        #print("reframedDataX.shape = ", reframedDataX.shape)
                        #print("reframedDataY.shape = ", reframedDataY.shape)
                        if ModelSpace == 'OnlyInfectedSelf':
                            lags = [int(x) for x in reframeData_frameConstructor[1].split(",")]
                            startingYasRegressInput = np.array([reframedDataY[-x][0] for x in lags][1:]).reshape(1, -1)
                        elif ModelSpace == 'InfectedSelfWithMobilityData':
                            lags = [int(x) for x in reframeData_frameConstructor[1].split(",")]
                            InfectedX_list = [reframedDataY[-x][0] for x in lags][1:]
                            for x1 in reframeData_External_Shift:
                                for x2 in mobilityRegionData_TrainSet_DF.values[-x1]:
                                    InfectedX_list.append(x2)
                            startingYasRegressInput = np.array(InfectedX_list).reshape(1, -1)
                            #print("startingYasRegressInput.shape = ", startingYasRegressInput.shape)
                        else:
                            startingYasRegressInput = reframedData[2]
    
                        sub_row_Preds, sub_row_Preds_std = model_GPR_List[modelIn].predict(startingYasRegressInput, return_std=True)
    
                        yWithPredictionsList[modelIn].append(pd.concat([SubSet_DF, pd.Series(sub_row_Preds[0])], axis=0))
                    else:
                        if ModelSpace in ['OnlyInfectedSelf', 'InfectedSelfWithMobilityData']:
                            previousPredictionOfRegion = total_row_subPred[modelIn]
                            yWithPredictionsList[modelIn].append(pd.Series(previousPredictionOfRegion))
                            yWithPredictionsDF = pd.concat(yWithPredictionsList[modelIn])
                            iterativePredictionInput = reframeData(yWithPredictionsDF, reframeData_frameConstructor[1], 0, frameConstructor=reframeData_frameConstructor[0])[2]
                            if ModelSpace == 'InfectedSelfWithMobilityData':
                                iterativePredictionInput = np.concatenate([iterativePredictionInput[0], mobilityRegionData_TestSet_DF.iloc[step_i].values]).reshape(1, -1)
                        elif ModelSpace == 'OnlyInfectedGeoNeighbors':
                            iterativePredictionInput = total_row_subPred[region_GeoNeighbors_Pos].reshape(reframedData[2].shape)
                        elif ModelSpace == 'WithWeatherData':
                            previousPredictionsCovidData = total_row_subPred[region_GeoNeighbors_Pos]
                            testSubWeatherData = pd.Series(previousPredictionsCovidData).append(testWeatherDataDF.iloc[step_i-1]).values
                            iterativePredictionInput = testSubWeatherData.reshape(reframedData[2].shape)
    
                        sub_row_Preds, sub_row_Preds_std = model_GPR_List[modelIn].predict(iterativePredictionInput, return_std=True)
    
                    if scaler == "Standard":
                        subPredOut = scY.inverse_transform(sub_row_Preds)[0][0]
                    else:
                        subPredOut = sub_row_Preds[0][0]
    
                    models_preds_list.append(subPredOut)
    
                total_row_subPred = np.array(models_preds_list)
                #print("step_i = ", step_i, ", total_row_subPred = ", total_row_subPred)
                Preds_List.append(total_row_subPred)
    
            IterativePredsDF = pd.DataFrame(Preds_List, columns=TrainSet_DF.columns, index=TestSet_DF.index)
            pd.concat([TrainSet_DF,IterativePredsDF]).to_excel(modelDataPath + mode + "_" + ModelSpace + "_target_intrinsic_dim_" + str(target_intrinsic_dim) + "_dmKnn_" + str(dm_optParams_knnIn) + "_kernelChoice_" + str(kernelChoice) + "_EmbeddingSpacePredsDF.xlsx")
            if predSchemeSplit[2] == "GH":
                GH_epsilonIn = "opt"
                GH_cut_offIn = "opt"
                #GH_cut_offIn = np.inf
                lift_optParams_knnIn = dm_optParams_knnIn
                LiftedPreds = Lift("GH", X_TrainSet_DF.values, TestSet_DF.values, TrainSet_DF.values, IterativePredsDF.values,
                                           lift_optParams_knn=lift_optParams_knnIn, GH_epsilon=GH_epsilonIn,
                                           GH_cut_off=GH_cut_offIn)[0]
                IterativePredsDF = pd.DataFrame(LiftedPreds, index=TestSet_DF.index, columns=TestSet_DF.columns)
                IterativePredsDF = pd.concat([X_TrainSet_DF, IterativePredsDF], axis=0)
            else:
                IterativePredsDF = pd.concat([TrainSet_DF, IterativePredsDF], axis=0)
    
            IterativePredsDF.to_excel(modelDataPath + mode + "_" + ModelSpace + "_target_intrinsic_dim_" + str(target_intrinsic_dim) + "_dmKnn_" + str(dm_optParams_knnIn) + "_kernelChoice_" + str(kernelChoice) + "_dataMde_" + dataMode + "_PredsDF.xlsx")
            pickle.dump(forecastHorizon, open("forecastHorizon.p", "wb"))
    
    //////////////////////////////////////////////////////////////////////////////////////////////////
        PredsDataList = []
        for modelIn in range(len(preds_list)):
            subPredsList = preds_list[modelIn]
            regionPredsDF = pd.concat(subPredsList)
            PredsDataList.append(regionPredsDF)
    
        ConditionedPredsData = pd.concat(PredsDataList, axis=1)
        ConditionedPredsData.columns = ModelDF.columns
        ConditionedPredsData = pd.concat([ModelDF.loc[:traintoDate, :].replace(modelTermLabel+'_', ''), ConditionedPredsData], axis=0)
        ConditionedPredsData.to_excel(modelDataPath + mode + "_" + ModelSpace + "_" + modelTerm + "_PredsDF.xlsx")
    //////////////////////////////////////////////////////////////////////////////////////////////////
    
        else:
            try:
                embedSpacePredsDF = pd.read_excel(modelDataPath + ModelName + "_EmbeddingSpacePredsDF.xlsx").set_index('data', drop=True)
                embedSpacePredsDF.plot(legend=None)
                plt.axvline(x=len(embedSpacePredsDF) - forecastHorizon, ymin=-1, ymax=1, color='b', ls='--', lw=1.5,label='axvline - % of full height')
                plt.savefig(mydirEmbedSpacePredictions + ModelName + '_embedSpacePredsDF.png')
            except Exception as e:
                print(e)
    
            PredsList = []
            for modelTerm in ["DI", "DR", "DD"]:
                ModelDF = pd.read_excel(modelDataPath + ModelName + "_" + modelTerm + "_PredsDF.xlsx").set_index('data', drop=True)
                if modelTerm == "DI":
                    modelTermLabel = 'nuovi_positivi'
                elif modelTerm == "DR":
                    modelTermLabel = 'dimessi_guariti'
                elif modelTerm == "DD":
                    modelTermLabel = 'deceduti'
                ModelDF.columns = [x.replace(modelTermLabel+'_', '') for x in ModelDF.columns]
                PredsList.append(ModelDF)
    
            RealDF = totale_positiviDF.copy()
            PredsDF = PredsList[0] - PredsList[1] - PredsList[2] # DI(t) - DR(t) - DD(t)
    
            RealDF = RealDF.loc[PredsDF.index, :]
            RealDF.columns = PredsDF.columns
            shifted_RealDF = RealDF.shift()
            PredsDF = PredsDF + shifted_RealDF #I(t-1) + (DI(t) - DR(t) - DD(t))
    
            "Individual RMSEs"
            rmseList = []
            for col in PredsDF.columns:
                if dataMode == 'Diff':
                    pd.concat([PredsDF[col].rename("GPR"), RealDF[col]], axis=1).cumsum().plot()
                else:
                    pd.concat([PredsDF[col].rename("GPR"), RealDF[col]], axis=1).plot()
                plt.axvline(x=len(RealDF[col])-forecastHorizon, ymin=0, ymax=RealDF[col].max(), color='b', ls='--', lw=1.5, label='axvline - % of full height')
                plt.savefig(mydir + ModelName + '_' + col + '.png')
                sub_rmse = rmse(PredsDF[col].values, RealDF[col].values)
                rmseList.append(sub_rmse)
    
            #"Total RMSE (Matrix)"
            #rmseTotal = rmse(PredsDF.values, RealDF.values)
            #print("rmseTotal = ", rmseTotal)
            #rmseDF = pd.DataFrame(rmseList, index=PredsDF.columns, columns=['RMSE'])
            #print("rmseDF = ")
            #print(rmseDF)
            #rmseDF.to_excel(modelDataPath + mode + "_" + ModelSpace + "_rmseDF.xlsx")
####################################################################################################

def ModelRepo(Settings):

    ModelDF = totale_positiviDF.copy()
    ModelDF_Label = "totale_positivi_"
    ModelName = '_'.join([str(x) for x in list(Settings.values())])
    print("ModelDF.shape = ", ModelDF.shape)

    mainRolling_kernel = ConstantKernel() + Matern() + ExpSineSquared() + WhiteKernel()
    #ConstantKernel() + Matern() + RBF() + ExpSineSquared() + RationalQuadratic() + DotProduct() + PairwiseKernel() + WhiteKernel()

    RegressDF_List = [totale_positiviDF.copy()]
    RegressNum = 1

    if Settings["RegressFramework"] == "I(t)=f(dI(t-lags))":
        RegressDF_List = [totale_positiviDF.copy().diff()]
        RegressNum = 2
    elif Settings["RegressFramework"] == "I(t)=f(DI(t-lags),DR(t-lags),DD(t-lags))":
        RegressDF_List = [nuovi_positiviDF.copy(), dimessi_guaritiDF.copy(), decedutiDF.copy()]
        RegressNum = 3

    nn_options = {
        'hidden_layer_sizes': (3, 3),
        'solver': 'lbfgs',
        'activation': 'tanh',
        'max_iter': 1500,  # default 200
        'alpha': 0.01,  # default 0.0001
        'random_state': None  # default None
    }

    #model_GPR_List = [[GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=5, normalize_y=True) for var1 in range(len(RegressDF_List))] for var in range(len(ModelDF))]
    model_GPR_List = [[MLPRegressor(**nn_options) for var1 in range(len(RegressDF_List))] for var in range(len(ModelDF))]

    print("len(model_GPR_List) = ", len(model_GPR_List), ", len(model_GPR_List[0]) = ", len(model_GPR_List[0]))

    lags = [int(x) for x in Settings["reframeDataConstructor"][1].split(",")]

    if Settings["iterationsToRun"] == "entireDataset":
        iterationsToRun = ModelDF.shape[0]
    else:
        iterationsToRun = Settings["LearningMemory"] + Settings["iterationsToRun"]

    preds_list = [[] for var in range(len(ModelDF.columns))]
    for i in tqdm(range(Settings["LearningMemory"], iterationsToRun, Settings["RollStep"])):#
        for modelIn in range(len(ModelDF.columns)):
            if Settings["LearningWindowMode"] == "Expanding":
                I_t = ModelDF.iloc[0:i, modelIn]
            else:
                I_t = ModelDF.iloc[i - Settings["LearningMemory"]:i, modelIn]

            region_name = I_t.name.replace(ModelDF_Label, "")
            if region_name == "Lombardia":
                #print("region_name = ", region_name)

                if Settings["RegressFramework"] == "I(t)=f(I(t-lags),Weather(t-lags),Mobility(t-lags))":
                    "Weather Data"
                    correspondingWeatherRegions = list(DataRegionsMappingDF[DataRegionsMappingDF["Covid Region"] == region_name]['Weather Region'].values)
                    allWeatherDF = generateWeatherDF(correspondingWeatherRegions).ffill().rolling(window=dataMode[1][1]).mean().ffill().fillna(0)

                    "Mobility Data"
                    mobility_df = pd.read_csv('.\\WorkingDataset\\IT_Region_Mobility_Report.csv')
                    #mobility_df['date'] = pd.to_datetime(mobility_df['date'])
                    mobility_df["data"] = mobility_df["date"]
                    mobility_df = mobility_df.set_index('data')
                    region_keys_df = pd.read_excel(f'.\\WorkingDataset\\region2keys.xlsx', index_col='Region')
                    region_mobility_code = region_keys_df.loc[region_name, 'Mobility code']
                    mobility_df = mobility_df[mobility_df['iso_3166_2_code'] == region_mobility_code]
                    mobility_df['workplaces_percent_change_from_baseline'] = mobility_df[
                        'workplaces_percent_change_from_baseline'].rolling(dataMode[1][2]).mean()
                    mobility_df = mobility_df["workplaces_percent_change_from_baseline"]

                    RegressDF_List = [ModelDF[ModelDF_Label + region_name], pd.concat([allWeatherDF, mobility_df], axis=1).sort_index()]
                    RegressNum = 4

                if RegressNum in [3]:
                    reframedData = reframeData(I_t.copy(), Settings["reframeDataConstructor"][1], 0, frameConstructor=Settings["reframeDataConstructor"][0])
                    reframedDataX = pd.DataFrame(reframedData[0], index=reframedData[3][0])

                    subPredList = []
                    for TargetVar in range(len(RegressDF_List)):
                        TargetVarDF = RegressDF_List[TargetVar].loc[
                            reframedData[3][1], [x for x in RegressDF_List[TargetVar].columns if region_name in x]]
                        model_GPR_List[modelIn][TargetVar].fit(reframedDataX.values, TargetVarDF.values.reshape(-1, 1))
                        RegressInput = np.array([TargetVarDF.values[-x][0] for x in lags][1:]).reshape(1, -1)
                        sub_Preds, sub_Preds_Std = model_GPR_List[modelIn][TargetVar].predict(RegressInput, return_std=True)
                        subPredList.append(sub_Preds[0][0])

                    I_t_List = list(I_t.values)
                    I_t_Pred = I_t.values[-1] + subPredList[0] - subPredList[1] - subPredList[2]
                    I_t_List.append(I_t_Pred)
                    ###########################################################################################################

                    iterativeSubPredictions = [I_t_Pred]
                    step_I_t_List = I_t_List
                    for iterativeStep in range(Settings["predictAhead"] - 1):
                        #RegressInput = np.array([pd.Series(step_I_t_List).diff().tolist()[-x] for x in lags][1:]).reshape(1, -1)
                        RegressInput = np.array([step_I_t_List[-x] for x in lags][1:]).reshape(1, -1)

                        subIterativePredList = [model_GPR_List[modelIn][iter_TargetVar].predict(RegressInput)[0][0] for iter_TargetVar in range(len(RegressDF_List))]
                        iter_I_t_Pred = I_t.values[-1] + subIterativePredList[0] - subIterativePredList[1] - subIterativePredList[2]
                        step_I_t_List.append(iter_I_t_Pred)

                        iterativeSubPredictions.append(iter_I_t_Pred)
                    iterativeSubPredictions.insert(0, I_t.index[-1])
                    preds_list[modelIn].append(iterativeSubPredictions)
                elif RegressNum in [4]:
                    regressorsToReframeDF = RegressDF_List[0].loc[I_t.index]
                    ExternalDataDF = RegressDF_List[1].loc[I_t.index]

                    reframedData = reframeData(regressorsToReframeDF, Settings["reframeDataConstructor"][1], 0, frameConstructor=Settings["reframeDataConstructor"][0],
                                               ExternalData=[ExternalDataDF, [1]])
                    reframedDataX = pd.DataFrame(reframedData[0], index=reframedData[3][0])
                    reframedDataY = pd.DataFrame(reframedData[1], index=reframedData[3][1])

                    model_GPR_List[modelIn][0].fit(reframedDataX.values, reframedDataY.values.reshape(-1, 1))

                    RegressInputList = []
                    for x in lags[1:]:
                        RegressInputList.append(regressorsToReframeDF.iloc[-x])
                    for x in ExternalDataDF.iloc[-1].values:
                        RegressInputList.append(x)

                    RegressInput = pd.Series(RegressInputList).values.reshape(1, -1)
                    #I_t_Pred, I_t_Pred_Std = model_GPR_List[modelIn][0].predict(RegressInput, return_std=True)
                    I_t_Pred = model_GPR_List[modelIn][0].predict(RegressInput)
                    #I_t_Pred = I_t_Pred[0][0]
                    #I_t_Pred_Std = I_t_Pred_Std[0]
                    print("I_t_Pred = ", I_t_Pred)
                    I_t_List = list(I_t.values)
                    I_t_List.append(I_t_Pred)
                    lastDatePosition = RegressDF_List[0].index.get_loc(regressorsToReframeDF.index[-1])
                    print("lastDatePosition = ", lastDatePosition)
                    ###########################################################################################################

                    iterativeSubPredictions = [I_t_Pred]
                    for iterativeStep in range(Settings["predictAhead"] - 1):
                        subDF = RegressDF_List[0].iloc[0:lastDatePosition + iterativeStep + 1]
                        subDF.iloc[-1] = iterativeSubPredictions[-1]
                        RegressInputList = []
                        for x in lags[1:]:
                            RegressInputList.append(subDF.iloc[-x])
                        for x in RegressDF_List[1].loc[subDF.index[-1]].values:
                            RegressInputList.append(x)
                        RegressInput = pd.Series(RegressInputList).values.reshape(1, -1)

                        #subIterativePred, subIterativePred_Std = model_GPR_List[modelIn][0].predict(RegressInput, return_std=True)[0][0]
                        subIterativePred = model_GPR_List[modelIn][0].predict(RegressInput)
                        iterativeSubPredictions.append(subIterativePred)
                    iterativeSubPredictions.insert(0, I_t.index[-1])

                    preds_list[modelIn].append(iterativeSubPredictions)

    pickle.dump(preds_list, open(modelDataPath+ModelName+"_preds_list.p", "wb"))
    
"""