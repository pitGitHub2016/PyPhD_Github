import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from numpy.linalg import LinAlgError
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, RationalQuadratic, ExpSineSquared, Matern, \
    ConstantKernel
import warnings
warnings.filterwarnings("ignore")
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

def DataHandler(mode):
    if mode == "FirstSetup":
        df = pd.read_excel("Energy generation and MCP per hour.xlsx", sheet_name='data', skiprows=1)
        df.to_excel("dataTS_Main.xlsx")
    elif mode == "SetupHourly":
        df = pd.read_excel("dataTS.xlsx")
        df['DATE'] = df['DATE'].astype(str).str.split(" ").str[0]
        df = df.set_index("DATE",drop=True)
        dfList = []
        for hourBase in range(1,25):
            subDF = df[df['HOUR'] == hourBase]
            subDF.columns = [x+"_"+str(hourBase) for x in subDF.columns]
            dfList.append(subDF)
        dfAll = pd.concat(dfList, axis=1).sort_index()
        dfAll.to_excel("dataHourlyTS.xlsx")
    ############################################################################################################
    elif mode == "Extend":
        df = pd.read_excel("dataHourlyTS.xlsx").set_index("DATE",drop=True).ffill().bfill()
        df.index = pd.to_datetime(df.index)
        CalendarDates = pd.DataFrame(None,index=pd.date_range(start=df.index[0],end="2050-01-01"), columns=['Calendar'])
        dfAll = pd.concat([df, CalendarDates],axis=1)
        dfAll.index = [str(x).split(" ")[0] for x in dfAll.index]
        dfAll.index.names = ['DATE']
        dfAll.to_excel("dataHourlyTS_Extended.xlsx")
    elif mode == "SetupRESHourly":
        df = pd.read_excel("dataHourlyTS_Extended.xlsx").set_index("DATE",drop=True)
        dfRES = pd.read_excel("dataStatsRES.xlsx")
        dfRES.columns = [str(x)+"_RES" for x in dfRES.columns]
        dfRES["HOUR_num"] = dfRES["HOUR_RES"].astype(str).str.split(":").str[0].astype(float)
        dfRES.loc[dfRES["HOUR_num"]==0,"HOUR_num"] = 24
        for hourBase in tqdm(range(1, 25)):
            ############################################
            df["RESparticipation_"+str(hourBase)] = None
            ############################################
            for idx, row in df.iterrows():
                try:
                    idx_Split = str(idx).split("-")
                    yearIdx = idx_Split[0]
                    monthIdx = float(idx_Split[1])
                    dayIdx = float(idx_Split[2])
                    df.loc[idx,"RESparticipation_"+str(hourBase)] = dfRES[(dfRES['DAY_RES']==dayIdx)&(dfRES['MONTH_RES']==monthIdx)&(dfRES['HOUR_num']==hourBase)][yearIdx+"_RES"].values[0]
                except Exception as e:
                    print(e)
        #######################################################################################
        df.to_excel("dataHourlyTS_withRES.xlsx")
        #######################################################################################
        df = df.drop([x for x in df.columns if ('Calendar' in x)|('HOUR' in x)|('Crete' in x)|('Oil' in x)|('PUMP' in x)],axis=1)
        df.to_excel("dataHourlyTS_withRES_Edited.xlsx")
    elif mode == "IncludeESEK":
        df = pd.read_excel("dataHourlyTS_withRES_Edited.xlsx").set_index("DATE", drop=True)
        df.index = pd.to_datetime(df.index)
        df_ESEK = pd.read_excel("StatsRES_Totals.xlsx",sheet_name='TS')
        df_ESEK = df_ESEK.set_index("Date",drop=True)
        df_ESEK.index = pd.to_datetime(df_ESEK.index)

        dfAll = pd.concat([df, df_ESEK],axis=1).sort_index()
        dfAll.to_excel("dataHourlyTS_withRES_Edited_preInterpolation.xlsx")

        WhatToInterpolate = [x for x in df_ESEK.columns]
        dfAll[WhatToInterpolate] = dfAll[WhatToInterpolate].interpolate(method='linear', limit_direction='both').ffill().bfill()
        dfAll.to_excel("dataHourlyTS_withRES_Edited_withESEK.xlsx")

def Model(modelSpace, ModelFormat, modelName, **kwargs):
    ID = modelSpace + "_" + ModelFormat + "_" + modelName
    modelNameSplit = modelName.split("_")

    if 'Scaler' in kwargs:
        Scaler = kwargs['Scaler']
    else:
        Scaler = 'Standard'

    lookbackPeriod_Base = 25
    NumPredAhead = 5

    df = pd.read_excel("dataHourlyTS_withRES_Edited_withESEK.xlsx")
    df = df.set_index(df.columns[0],drop=True)
    df.index.names = ['DATE']
    df.index = pd.to_datetime(df.index)
    df = df.loc[:'2030-01-01',:]

    SMP_MCP_Cols = [x for x in df.columns if 'SMP/MCP' in x]
    SMP_MCP_DF = df[SMP_MCP_Cols].dropna()

    df = df.ffill().bfill()

    if modelSpace == "SMP":
        modelDF_Base = df[SMP_MCP_Cols]
    elif modelSpace == "ScenarioA":
        modelDF_Base = df[SMP_MCP_Cols+[x for x in df.columns if ('ESEK_'in x)]]
        #|('SUPPLY TOTAL_' in x)|("DEMAND TOTAL_" in x)
    elif modelSpace == "ALL":
        modelDF_Base = df.copy()

    if ModelFormat == 'Raw':
        modelDF = modelDF_Base.copy()
        PredsNaN_Fill = None
    elif ModelFormat == 'Diff':
        modelDF = modelDF_Base.diff().fillna(0)
        PredsNaN_Fill = 0

    modelDF.to_excel('MetaData/modelDF_' + modelSpace + "_" + modelName + '.xlsx')

    ######################################################################################################
    PredsDF = modelDF.copy()#pd.DataFrame(None,index=modelDF.index,columns=modelDF.columns)
    for i in tqdm(range(lookbackPeriod_Base, modelDF.index.get_loc(modelDF.index[-1]), NumPredAhead)):
        lookbackPeriod = lookbackPeriod_Base

        valData = modelDF.iloc[i:i + NumPredAhead, :]
        PredsIndex = valData.index

        ReplaceCols = SMP_MCP_Cols
        baseData = modelDF.copy()
        if i >= modelDF.index.get_loc(SMP_MCP_DF.index[-1]):
            baseData = PredsDF.copy()

        fitFlag = False
        GeneralErrorFlag = False

        while (fitFlag == False)&(lookbackPeriod >= 1):#150
            #try:
                trainData = baseData.iloc[i - lookbackPeriod:i, :].astype(float)
                latestIndex = trainData.index[-1]

                if modelNameSplit[0] == "VAR":
                    model = VAR(trainData.values)
                    model_fit = model.fit()
                    sub_preds = model_fit.forecast(model_fit.endog, steps=len(PredsIndex))

                    sub_predDF = pd.DataFrame(sub_preds, index=PredsIndex, columns=trainData.columns)
                elif modelNameSplit[0] in ["GPR", "RNN", "RF"]:
                    if modelNameSplit[1] == "A":

                        Y = SMP_MCP_Cols
                        X = [x for x in list(modelDF.columns) if x not in SMP_MCP_Cols]
                        NewX = valData[X]
                        ################################################################################################
                        if Scaler == 'Standard':
                            scX = StandardScaler()
                            scNX = StandardScaler()
                            scY = StandardScaler()
                        elif Scaler == 'MinMax':
                            scX = MinMaxScaler(feature_range=(0, 1))
                            scNX = MinMaxScaler(feature_range=(0, 1))
                            scY = MinMaxScaler(feature_range=(0, 1))
                        reframedDataX = scX.fit_transform(trainData[X])
                        reframedDataNewX = scNX.fit_transform(NewX)
                        reframedDataY = scY.fit_transform(trainData[Y])
                        ################################################################################################
                        if modelNameSplit[0] == "GPR":
                            mainRolling_kernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * Matern() + 1 * WhiteKernel()
                            mainKernel = GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2)
                            model_fit = mainKernel.fit(reframedDataX, reframedDataY)
                            [sub_preds, preds_Std] = model_fit.predict(reframedDataNewX, return_std=True)
                        elif modelNameSplit[0] == "RNN":
                            "Reshape"
                            reframedDataX = reframedDataX.reshape((reframedDataX.shape[0], 1, reframedDataX.shape[1]))
                            reframedDataNewX = reframedDataNewX.reshape((reframedDataNewX.shape[0], 1, reframedDataNewX.shape[1]))
                            reframedDataY = reframedDataY.reshape((reframedDataY.shape[0], 1, reframedDataY.shape[1]))

                            model = Sequential()
                            model.add(LSTM(reframedDataX.shape[0], input_shape=(reframedDataX.shape[1], reframedDataX.shape[2])))
                            #model.add(LSTM(reframedDataX.shape[1], input_shape=(reframedDataX.shape[1], FeatSpaceDims)))
                            model.add(Dense(reframedDataY.shape[2]))
                            model.compile(loss='mean_squared_error', optimizer='adam')
                            model.fit(reframedDataX, reframedDataY, epochs=100, batch_size=1, verbose=2)
                            sub_preds = model.predict(reframedDataNewX)
                        elif modelNameSplit[0] == "RF":
                            "Reshape"
                            regr = RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=500, random_state=0)
                            regr.fit(reframedDataX, reframedDataY)
                            sub_preds = regr.predict(reframedDataNewX)
                        ################################################################################################
                        inv_sub_preds = scY.inverse_transform(sub_preds)
                        ################################################################################################
                        sub_predDF = pd.DataFrame(inv_sub_preds, index=PredsIndex, columns=Y)

                    elif modelNameSplit[1] == "B":

                        WhatToPredict = 'SMP_MCP'

                        if WhatToPredict == 'SMP_MCP':
                            WhatToPredict_Y = SMP_MCP_Cols
                        else:
                            WhatToPredict_Y = list(trainData.columns)

                        if modelNameSplit[0] == "GPR":
                            mainRolling_kernel = 1 * ConstantKernel() + 1 * ExpSineSquared() + 1 * Matern() + 1 * WhiteKernel()
                            model_List = [GaussianProcessRegressor(kernel=mainRolling_kernel, random_state=0, n_restarts_optimizer=2,normalize_y=True) for var in range(len(SMP_MCP_Cols))]
                        if modelNameSplit[0] == "RNN":
                            pass
                        elif modelNameSplit[0] == "RF":
                            model_List = [RandomForestRegressor(max_depth=10, min_samples_split=2, n_estimators=500, random_state=0)
                                          for var in range(len(WhatToPredict_Y))]

                        Preds_List = []
                        for step_i in range(NumPredAhead):
                            models_preds_list = []
                            for modelIn in range(len(model_List)):
                                ########################################################################################
                                if step_i == 0:
                                    #reframedData = reframeData(trainData.values, 1, modelIn)
                                    reframedData = reframeData(trainData.values, 1, modelIn)
                                    #print(pd.DataFrame(reframedData[0]))
                                    #print('///////////////////////////////////')
                                    #print(pd.DataFrame(reframedData[1]))
                                    #print('///////////////////////////////////')
                                    #print(pd.DataFrame(reframedData[2], columns=trainData.columns))
                                    #time.sleep(300000)
                                    if Scaler == "Standard":
                                        scX = StandardScaler()
                                        reframedDataX = scX.fit_transform(reframedData[0])
                                        scY = StandardScaler()
                                        reframedDataY = scY.fit_transform(reframedData[1])
                                    elif Scaler == 'MinMax':
                                        scX = MinMaxScaler(feature_range=(0, 1))
                                        reframedDataX = scX.fit_transform(reframedData[0])
                                        scY = MinMaxScaler(feature_range=(0, 1))
                                        reframedDataY = scY.fit_transform(reframedData[1])
                                    else:
                                        reframedDataX = reframedData[0]
                                        reframedDataY = reframedData[1]
                                    ########################################################################################
                                    if modelNameSplit[0] == "GPR":
                                        model_List[modelIn].fit(reframedDataX, reframedDataY.reshape(-1, 1))
                                        sub_row_Preds, sub_row_Preds_std = model_List[modelIn].predict(reframedData[2])#, return_std=True)
                                    elif modelNameSplit[0] == "RF":
                                        model_List[modelIn].fit(reframedDataX, reframedDataY)
                                        sub_row_Preds = model_List[modelIn].predict(reframedData[2])
                                else:
                                    if modelNameSplit[0] == "GPR":
                                        sub_row_Preds, sub_row_Preds_std = model_List[modelIn].predict(
                                            total_row_subPred.reshape(reframedData[2].shape),return_std=True)
                                    elif modelNameSplit[0] == "RF":
                                        sub_row_Preds = model_List[modelIn].predict(total_row_subPred.reshape(reframedData[2].shape))
                                ########################################################################################
                                if Scaler == "Standard":
                                    subPredOut = scY.inverse_transform(sub_row_Preds)[0]
                                else:
                                    subPredOut = sub_row_Preds[0][0]
                                ########################################################################################
                                models_preds_list.append(subPredOut)
                            ########################################################################################
                            total_row_subPred = np.array(models_preds_list)
                            ########################################################################################
                            if WhatToPredict == 'SMP_MCP':
                                LastInputPoint = pd.DataFrame(reframedData[2], columns=trainData.columns).iloc[0,:]
                                LastInputPoint.loc[[x for x in LastInputPoint.index if x in SMP_MCP_Cols]] = total_row_subPred
                                total_row_subPred = LastInputPoint.values
                            ########################################################################################
                            #print("step_i = ", step_i, ", total_row_subPred = ", total_row_subPred)
                            Preds_List.append(total_row_subPred)

                        sub_predDF = pd.DataFrame(Preds_List, index=PredsIndex, columns=trainData.columns)

                fitFlag = True
            #except LinAlgError as e:
            #    print(e, ", Reduced lookbackPeriod by 25 obs .. retrying ... ")
            #    lookbackPeriod -= 25
            #except Exception as e:
            #    print(e)
            #    fitFlag = True
            #    GeneralErrorFlag = True

        if (fitFlag == False)|(GeneralErrorFlag==True):
            sub_predDF = pd.DataFrame(PredsNaN_Fill, index=PredsIndex, columns=modelDF.columns)

        PredsDF.loc[sub_predDF.index, ReplaceCols] = sub_predDF.copy()

        if ModelFormat == 'Raw':
            PredsDF = PredsDF.ffill()

    ######################################################################################################

    if ModelFormat == 'Diff':
        PredsDF.to_excel('Results/predDF_' + ID + '_PrePurePredStage.xlsx')
        PredsDF = (PredsDF.fillna(0) + modelDF_Base.ffill().shift()).ffill()

    PredsDF.to_excel('Results/predDF_' + ID + '.xlsx')

    errDF = PredsDF - modelDF_Base
    errDF.to_excel('Results/errDF_' + ID + '.xlsx')

    fig,ax = plt.subplots(nrows=2,ncols=1)
    PredsDF[SMP_MCP_Cols].plot(ax=ax[0])
    errDF[SMP_MCP_Cols].plot(ax=ax[1])
    plt.show()

def FlattenToCalendarDates():
    ID = 'ALL_Raw_VAR'
    df = pd.read_excel('Results/predDF_' + ID + '.xlsx').set_index("DATE",drop=True)

    CalendarDatesFlattening = []
    for hourBase in tqdm(range(1, 25)):
        subDF = df[[x for x in df.columns if float(x.split("_")[1])==hourBase]]
        subDF.columns = [x.split("_")[0] for x in subDF.columns]
        subDF.index = pd.to_datetime(subDF.index)
        subDF['Time'] = hourBase
        subDF['Time'] = pd.to_timedelta(subDF['Time'], unit='h')
        subDF.index = subDF.index + subDF['Time']
        subDF = subDF.drop('Time', axis=1)
        CalendarDatesFlattening.append(subDF)

    CalendarDatesFlatteningDF = pd.concat(CalendarDatesFlattening,axis=0).sort_index()
    CalendarDatesFlatteningDF.to_excel('Results\CalendarDatesFlattening/' + ID + '.xlsx')

    CalendarDatesFlatteningDF['SMP/MCP'].plot()
    plt.show()

#DataHandler("FirstSetup")
#DataHandler("SetupHourly")
#DataHandler("Extend")
#DataHandler("SetupRESHourly")
#DataHandler("IncludeESEK")
#########################################################
#Model("SMP", "Raw", "VAR")
#Model("SMP", "Diff", "VAR")
#########################################################
#Model("ScenarioA", "Raw", "VAR")
#Model("ScenarioA", "Diff", "VAR")
#########################################################
#Model("ScenarioA", "Raw", "GPR_A")
#Model("ScenarioA", "Diff", "GPR_A")
#########################################################
#Model("ScenarioA", "Raw", "RNN_A")
#Model("ScenarioA", "Diff", "RNN_A")
#########################################################
#Model("ScenarioA", "Raw", "RF_A")
#Model("ScenarioA", "Diff", "RF_A")
#########################################################
Model("ScenarioA", "Raw", "RF_B")
#Model("ScenarioA", "Diff", "RF_A")
#########################################################
#Model("ALL", "Raw", "VAR")
#Model("ALL", "Diff", "VAR")
#Model("ALL", "Raw", "GPR")
#########################################################
#FlattenToCalendarDates()
  