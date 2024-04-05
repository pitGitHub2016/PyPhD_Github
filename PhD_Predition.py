from scipy.linalg import svd
import numpy as np, json, time, pickle, glob, copy
from tqdm import tqdm
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.gaussian_process.kernels import ConstantKernel, ExpSineSquared, Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import manifold
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import (
    GeometricHarmonicsInterpolator as GHI, LaplacianPyramidsInterpolator as LPI, TSCRadialBasis,
    LocalRegressionSelection, TSCTakensEmbedding
)
from datafold.dynfold import LocalRegressionSelection

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

dataMode = None
#dataMode = 'Diff'

CovidDF = pd.read_excel(pathWorkingData + "totale_positivi.xlsx").set_index('data', drop=True).sort_index()
if dataMode == 'Diff':
    CovidDF = CovidDF.diff().dropna()

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

def Model(mode, **kwargs):

    if 'predMode' in kwargs:
        predMode = kwargs['predMode']
    else:
        predMode = "conditioned"

    if 'scaler' in kwargs:
        scaler = kwargs['scaler']
    else:
        scaler = None

    GeoNeighbors = pd.read_excel(pathWorkingData + "GeoNeighborsDF.xlsx")
    DataRegionsMappingDF = pd.read_excel(pathWorkingData + "DataRegionsMapping.xlsx")

    ModelDF = CovidDF.copy()

    predHorizon = 50
    traintoDate = '2021-02-23'
    testFromDate = '2021-02-24'
    testToDate = '2021-05-01'

    mainRolling_kernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * Matern() + 1 * WhiteKernel()
    model_GPR_List = [GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2, normalize_y=True) for var in range(len(CovidDF.columns))] #

    if predMode == "iterative":

        TrainSet_DF = ModelDF.loc[:traintoDate, :]
        TestSet_DF = ModelDF.loc[testFromDate:testToDate, :]
        forecastHorizon = len(TestSet_DF)

        Preds_List = []
        for step_i in tqdm(range(forecastHorizon)):

            models_preds_list = []
            for modelIn in range(len(model_GPR_List)):
                region_df = TrainSet_DF.iloc[:, modelIn]
                region_name = region_df.name.replace('totale_positivi_', '')
                print(region_name)
                region_GeoNeighbors = ['totale_positivi_' + str(x) for x in GeoNeighbors[GeoNeighbors["Region"] == region_name]['GeoNeighborsList'].values[0].split(',')]
                print(region_GeoNeighbors)
                region_GeoNeighbors_Pos = [TrainSet_DF.columns.get_loc(x) for x in region_GeoNeighbors]
                print(region_GeoNeighbors_Pos)

                SubSet_DF = TrainSet_DF.loc[:, region_GeoNeighbors]

                if step_i == 0:
                    reframedData = reframeData(SubSet_DF.values, 1, 1)
                    if scaler == "Standard":
                        scX = StandardScaler()
                        reframedDataX = scX.fit_transform(reframedData[0])
                        scY = StandardScaler()
                        reframedDataY = scY.fit_transform(reframedData[1])
                    else:
                        reframedDataX = reframedData[0]
                        reframedDataY = reframedData[1]

                    model_GPR_List[modelIn].fit(reframedDataX, reframedDataY.reshape(-1, 1))
                    try:
                        print("model_List[", modelIn, "].kernel = ", model_GPR_List[modelIn].kernel_)
                    except:
                        pass
                    sub_row_Preds, sub_row_Preds_std = model_GPR_List[modelIn].predict(reframedData[2], return_std=True)
                else:
                    sub_row_Preds, sub_row_Preds_std = model_GPR_List[modelIn].predict(total_row_subPred[region_GeoNeighbors_Pos].reshape(reframedData[2].shape), return_std=True)

                if scaler == "Standard":
                    subPredOut = scY.inverse_transform(sub_row_Preds)[0][0]
                else:
                    subPredOut = sub_row_Preds[0][0]

                models_preds_list.append(subPredOut)

            total_row_subPred = np.array(models_preds_list)
            print("step_i = ", step_i, ", total_row_subPred = ", total_row_subPred)
            Preds_List.append(total_row_subPred)

        IterativePredsDF = pd.DataFrame(Preds_List, columns=TrainSet_DF.columns, index=TestSet_DF.index)
        IterativePredsDF = pd.concat([TrainSet_DF, IterativePredsDF], axis=0)
        print(IterativePredsDF)
        IterativePredsDF.to_excel(modelDataPath + mode + "_PredsDF.xlsx")

    elif predMode == "conditioned":

        preds_list = []
        for modelIn in range(len(model_GPR_List)):
            region_df = ModelDF.iloc[:, modelIn]
            region_name = region_df.name.replace('totale_positivi_', '')
            print(region_name)
            region_GeoNeighbors = ['totale_positivi_'+str(x) for x in GeoNeighbors[GeoNeighbors["Region"] == region_name]['GeoNeighborsList'].values[0].split(',')]
            print(region_GeoNeighbors)

            SubSet_DF = ModelDF.loc[:, region_GeoNeighbors]
            SubSet_DF_shift = SubSet_DF.shift()
            SubSet_DF_shift.columns = [x+'_shift' for x in SubSet_DF_shift.columns]

            "Reframe Data for Regression"
            reframedDataDF = pd.concat([SubSet_DF.iloc[:, 0], SubSet_DF_shift], axis=1).dropna()
            "Split in Train and Set"
            reframedDataDF_Train = reframedDataDF.loc[:traintoDate, :] #ModelDF.shape[0]-predHorizon
            reframedDataDF_Test = reframedDataDF.loc[testFromDate:testToDate, :] #ModelDF.shape[0]-predHorizon+1

            "Fit only on TrainSet"
            model_GPR_List[modelIn].fit(reframedDataDF_Train.iloc[:, 1:].values, reframedDataDF_Train.iloc[:, 0].values.reshape(-1, 1))
            "Predict using the Input Test Set"
            sub_Preds, sub_Preds_Std = model_GPR_List[modelIn].predict(reframedDataDF_Test.iloc[:, 1:].values, return_std=True)
            sub_PredsDF = pd.DataFrame(sub_Preds, index=reframedDataDF_Test.index)
            preds_list.append(sub_PredsDF)

        ConditionedPredsData = pd.concat(preds_list, axis=1)
        ConditionedPredsData.columns = ModelDF.columns
        ConditionedPredsData = pd.concat([ModelDF.loc[:traintoDate, :], ConditionedPredsData], axis=0)
        ConditionedPredsData.to_excel(modelDataPath + mode + "_PredsDF.xlsx")

def Reporter(mode):
    def rmse(Preds, Real):
        out = np.round(np.sqrt((1 / Real.shape[0]) * np.sum((Preds - Real) ** 2)), 4)
        return out

    PredsDF = pd.read_excel(modelDataPath + mode + "_PredsDF.xlsx").set_index('data', drop=True)
    RealDF = CovidDF.copy()
    RealDF = RealDF.loc[PredsDF.index, :]
    RealDF.columns = PredsDF.columns

    "Total RMSE (Matrix)"
    rmseTotal = rmse(PredsDF.values, RealDF.values)
    print("rmseTotal = ", rmseTotal)

    "Individual RMSEs"
    rmseList = []
    for col in PredsDF.columns:
        if dataMode == 'Diff':
            pd.concat([PredsDF[col].rename("GPR"), RealDF[col]], axis=1).cumsum().plot()
        else:
            pd.concat([PredsDF[col].rename("GPR"), RealDF[col]], axis=1).plot()
        plt.savefig(modelDataPath + '\\figures\\' + mode + '_' + col + '.png')
        sub_rmse = rmse(PredsDF[col].values, RealDF[col].values)
        rmseList.append(sub_rmse)

    rmseDF = pd.DataFrame(rmseList, index=PredsDF.columns, columns=['RMSE'])
    print("rmseDF = ")
    print(rmseDF)
    rmseDF.to_excel(modelDataPath + mode + "_rmseDF.xlsx")

#DatasetBuilder('Raw')
#DatasetBuilder('PerRegion')
#GeoLocationNeighbors('SetupCoordinates')
#GeoLocationNeighbors('MapNeighbors')
#GeoLocationNeighbors('KGeoNeighborsSelect', kGeo=4)
Model("FullModel", predMode="iterative") # SIETTOS
#Model("FullModel", predMode="conditioned") # PP
Reporter("FullModel")

"""
NOTES : 
Model("FullModel", predMode="iterative", scaler="Standard")
"""