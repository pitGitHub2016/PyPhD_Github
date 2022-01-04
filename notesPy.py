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