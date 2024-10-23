import pandas as pd, numpy as np, sqlite3, matplotlib.pyplot as plt, os, sys
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
sys.path.insert(0,'F:/Dealing/Panagiotis Papaioannou/pyerb/')
import time, pickle, inspect, glob
from tqdm import tqdm
from pyerb import pyerb as pe
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class TD_DT_Validation:

    def __init__(self, Main_DT_Space):
        self.Main_DT_Space = Main_DT_Space #"Rets","RetsLive","RetsRollWin250"
        self.AlternativeStorageLocation = "C:/SinceWeHaveLimitedSpace/"
        self.mainExcel = pd.read_excel("AssetsDashboard.xlsx", sheet_name="ActiveStrategies")
        self.ActiveStrategiesFactorsControl = pd.read_excel("AssetsDashboard.xlsx", sheet_name="ActiveStrategiesFactorsControl").set_index("Asset", drop=True)
        #"""
        self.EigenPortfoliosFactorsControl = pd.read_excel("AssetsDashboard.xlsx", sheet_name="EigenPortfoliosFactorsControl").set_index("Asset", drop=True)
        self.workConn = sqlite3.connect("F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/DataDeck.db")
        self.TimeDependent_DT_Validation_Conn = sqlite3.connect(self.AlternativeStorageLocation+"TimeDependent_DT_Validation.db")
        self.AssetsDashboardExcel = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/AssetsDashboard.xlsx"
        self.AllDF = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)
        self.ActiveStrategies = pd.read_excel(self.AssetsDashboardExcel, engine='openpyxl', sheet_name='ActiveStrategies')
        self.ActiveAssets = []
        for col in self.ActiveStrategies:
            self.ActiveAssets += self.ActiveStrategies[col].dropna().tolist()
        self.ActiveAssets = list(set(self.ActiveAssets))
        self.FuturesTable = pd.read_sql('SELECT * FROM FuturesTable', self.workConn).set_index('index', drop=True)
        self.IndicatorsData = pd.read_excel("IndicatorsDashboard.xlsx", engine='openpyxl')
        self.IndicatorsDF_Raw = pd.read_sql('SELECT * FROM IndicatorsDeck_Raw',self.workConn).set_index('date', drop=True)
        self.IndicatorsDF = pd.read_sql('SELECT * FROM IndicatorsDeck',self.workConn).set_index('date', drop=True)
        self.IndicatorsDF = pe.IndicatorsHandler(self.IndicatorsDF, self.IndicatorsData, SelectorHandler=[[0, "exclude"], [2, "diff"]])
        AlternativeDataFieldsIndicators = pd.read_sql('SELECT * FROM AlternativeDataFieldsIndicators', self.workConn).set_index('date', drop=True)
        self.IndicatorsDF = pd.concat([self.IndicatorsDF, AlternativeDataFieldsIndicators], axis=1).sort_index()
        self.IndicatorsDF_Raw = pd.concat([self.IndicatorsDF_Raw, AlternativeDataFieldsIndicators], axis=1).sort_index()
        self.Store_Metrics = True
        self.EligibleForBestFactors = self.IndicatorsData[["Indicator", "EligibleForBestFactorsList_TimeDepValidation"]].set_index("Indicator", drop=True)
        self.EligibleForBestFactors = self.EligibleForBestFactors[self.EligibleForBestFactors["EligibleForBestFactorsList_TimeDepValidation"] != 0]
        self.DT_MacroFactors_Signals_Conn = sqlite3.connect(self.AlternativeStorageLocation + '/DT_Filters_Alphas\DT_MacroFactors_Signals.db')
        #self.Main_DT_Space = "RetsRollWin250" #"Rets","RetsLive","RetsRollWin250"
        #"""
    def DT_Filters_Alphas(self, **kwargs):

        if "InactivityThr" in kwargs:
            InactivityThr = kwargs['InactivityThr']
        else:
            InactivityThr = 0.9

        ########################################################################################################
        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)
        "Select only the assets that are in the ActiveStrategiesFactorsControl sheet"
        localRets = pe.dlog(self.df).fillna(0)[[x for x in self.df.columns if x in self.ActiveStrategiesFactorsControl.index]]
        ManifoldLearnerMethodsSpace = ["PCA","DMAPs","LLE"]
        ########################################################################################################
        for TargetDEMA_File in glob.glob(self.AlternativeStorageLocation + "/DEMAs/*"):
            DEMA_ID = TargetDEMA_File.split("\\")[-1]
            print(DEMA_ID)
            if "_DefaultSingleLookBack_" in DEMA_ID:#DEMA_ID.split("_")[0] not in ManifoldLearnerMethodsSpace:
                #################################################################################################
                if DEMA_ID.split("_")[0] in ManifoldLearnerMethodsSpace:
                    self.Main_DT_Space = "ManifoldLearner"
                    Pickle = open(self.AlternativeStorageLocation + "/LookbacksPacks/"+DEMA_ID.replace("_DEMAs", ""), 'rb')
                    outPickle = pickle.load(Pickle)
                    Pickle.close()
                    rets = outPickle[0]
                else:
                    rets = localRets
                    DT_Space_ID = DEMA_ID.replace("_LookBacksPack_DEMAs","")
                #################################################################################################
                DEMAs = pe.readPickleDF(TargetDEMA_File).fillna(0)
                for MacroMode in [""]:#"_diff"
                    print("MacroMode = ", MacroMode)
                    DT_Folder = self.AlternativeStorageLocation+"/DecisionTrees"+MacroMode+"/"+self.Main_DT_Space+"/"
                    pnlList = []
                    exPostPnlList = []
                    for TargetFile in tqdm(glob.glob(DT_Folder+'*')):
                        print(TargetFile)
                        if ("BeforeUpdate" not in TargetFile):
                        #if ("BeforeUpdate" not in TargetFile)&("DecisionTrees_RV_Rets_ExpWindow_25_USYC5Y20 Index" in TargetFile):
                            try:
                                MacroFactor = TargetFile.split("\\")[-1].split("_")[-1]
                                RollMode = TargetFile.split("_")[-3]+"_"+TargetFile.split("_")[-2]

                                DT_RV = pe.readPickleDF(TargetFile)
                                DT_RV = DT_RV[[x for x in DT_RV.columns if x.split(",")[1] in rets.columns]].ffill()

                                for filterMode in ["Raw"]:#"1STD"
                                    if filterMode == "Raw":
                                        filter = pe.sign(DT_RV.sub(self.IndicatorsDF[MacroFactor], axis=0).sort_index())
                                    elif filterMode == "1STD":
                                        filter = pe.sign((DT_RV+pe.expander(DT_RV, np.std, n=25)).sub(self.IndicatorsDF[MacroFactor],axis=0).sort_index())
                                filter.columns = [x.split(",")[1] for x in filter.columns]
                                for mode in ["Lower", "Upper"]:
                                    SigFilter = filter.copy()
                                    if mode == "Upper":
                                        SigFilter[filter > 0] = 0
                                    elif mode == "Lower":
                                        SigFilter[filter < 0] = 0
                                    ##############################################################
                                    sig = (pe.sign(DEMAs) * SigFilter).fillna(0)
                                    pe.savePickleDF(sig,self.AlternativeStorageLocation + "DT_Filters_Alphas/DT_MacroFactors_Signals/"+MacroFactor+"_"+RollMode+ "_" + mode+".p")
                                    ##############################################################
                                    pnl = (pe.S(sig, nperiods=2) * rets).sort_index()
                                    pnl = pnl[self.ActiveAssets]
                                    pnl.columns = [MacroFactor + "," + x + ",TreeThreshold_" + filterMode + "_" + RollMode + "_" + mode for x in pnl.columns]
                                    pnlList.append(pnl)
                                    ##############################################################
                                    exPostPnl = (sig * rets).sort_index()
                                    exPostPnl = exPostPnl[self.ActiveAssets]
                                    exPostPnl.columns = [MacroFactor + "," + x + ",TreeThreshold_" + filterMode + "_" + RollMode + "_" + mode for x in exPostPnl.columns]
                                    exPostPnlList.append(exPostPnl)
                            except Exception as e:
                                print(e)
                    #######################################################################################################
                    pnlDF = pd.concat(pnlList, axis=1).sort_index()
                    exPostPnlDF = pd.concat(exPostPnlList, axis=1).sort_index()
                    #############
                    pe.savePickleDF(pnlDF, self.AlternativeStorageLocation+'/DT_Filters_Alphas/DT_MacroFactors_PnLs/'+MacroMode+"_"+DT_Space_ID+"_pnlDF.p")
                    pe.savePickleDF(exPostPnlDF,self.AlternativeStorageLocation + '/DT_Filters_Alphas/DT_MacroFactors_PnLs/' + MacroMode + "_" + DT_Space_ID + "_exPostPnlDF.p")
                    #######################################################################################################
                    InactivityDF = pd.DataFrame(pe.Inactivity(pnlDF), columns=["Inactivity"]).reset_index()
                    InactivityDF.columns = ["ID", "Inactivity"]
                    ############# ALL HISTORY SNAPSHOT EX-POST SHARPE REPORTER ############################################################################
                    print("Creating _Alphas_DT_Filters_sh.xlsx ... ")
                    shSer = np.sqrt(252) * pe.sharpe(pnlDF)
                    sh = pd.DataFrame(shSer, columns=["Sharpe"]).reset_index()
                    sh.columns = ["ID", "Sharpe"]
                    sh["MacroFactor"] = sh["ID"].str.split(",").str[0]
                    sh["Asset"] = sh["ID"].str.split(",").str[1].str.split("TreeThreshold").str[0]
                    #######################################################################################################
                    ReportDF = pd.concat([sh.set_index("ID",drop=True), InactivityDF.set_index("ID",drop=True)],axis=1).reset_index()
                    ReportDF.to_excel(self.AlternativeStorageLocation+"DT_Filters_Alphas/"+MacroMode+"_"+DT_Space_ID+"_Alphas_DT_Filters_sh_"+self.Main_DT_Space+".xlsx", index=False)
                    ###########################BEST FACTORS##################################
                    shDF = pd.read_excel(self.AlternativeStorageLocation + "DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_" + self.Main_DT_Space + ".xlsx").set_index("ID", drop=True)
                    ##############################################################################################################################
                    shDF = shDF[shDF["Inactivity"] <= InactivityThr]
                    shDF_InactivityReport = shDF.copy().reset_index()
                    shDF_InactivityReport.to_excel(self.AlternativeStorageLocation + "DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_Inactivity_" + str(InactivityThr).replace(".", "") + "_" + self.Main_DT_Space + ".xlsx", index=False)
                    ##############################################################################################################################
                    shDF = shDF.reset_index()
                    shDF = shDF[shDF["MacroFactor"].isin(list(self.EligibleForBestFactors.index))]
                    ##############################################################################################################################
                    Top_Bottom_Features_List = []
                    for Asset in list(set(shDF["Asset"].tolist())):
                        subDF = shDF.loc[shDF['Asset'] == Asset, :].dropna(subset=["Sharpe"]).sort_values(by='Sharpe',ascending=False).set_index('Asset', drop=True)
                        maxPerformingFactor = subDF.iloc[0, :]
                        minPerformingFactor = subDF.iloc[-1, :]
                        Top_Bottom_Features_List.append(
                            [Asset, maxPerformingFactor["MacroFactor"] + "_" + maxPerformingFactor["ID"].split("_")[-1],
                             minPerformingFactor["MacroFactor"] + "_" + minPerformingFactor["ID"].split("_")[-1]])
                    BestFactorsDF = pd.DataFrame(Top_Bottom_Features_List,columns=["Asset", "SingleDecisionTreesControllers_Positive",
                                                          "SingleDecisionTreesControllers_Negative"]).set_index("Asset",drop=True)
                    BestFactorsDF = pd.concat([BestFactorsDF, self.ActiveStrategiesFactorsControl["VolControllers"]],axis=1).loc[self.ActiveStrategiesFactorsControl.index, :]
                    BestFactorsDF.to_excel(self.AlternativeStorageLocation + "TimeDependent_DT_Validation/BestFactorsDF_" + self.Main_DT_Space + ".xlsx")
                    #######################################################################################################

    def CreateExpandingSharpes(self,top_bottom_columns_Mode, **kwargs):

        ########################################################################################################
        self.top_bottom_columns_Mode = top_bottom_columns_Mode

        if "KillersFinderSettings" in kwargs:
            KillersFinderSettings = kwargs['KillersFinderSettings']
        else:
            KillersFinderSettings = {'outputFormat':'excel'}

        self.df = pd.read_sql('SELECT * FROM DataDeck', self.workConn).set_index('date', drop=True)
        "Select only the assets that are in the ActiveStrategiesFactorsControl sheet"
        localRets = pe.dlog(self.df).fillna(0)[[x for x in self.df.columns if x in self.ActiveStrategiesFactorsControl.index]]
        ManifoldLearnerMethodsSpace = ["PCA", "DMAPs", "LLE"]
        #################################################################################################
        for TargetDEMA_File in glob.glob(self.AlternativeStorageLocation + "/DEMAs/*"):
            DEMA_ID = TargetDEMA_File.split("\\")[-1]
            print(DEMA_ID)
            if "_DefaultSingleLookBack_" in DEMA_ID:#DEMA_ID.split("_")[0] not in ManifoldLearnerMethodsSpace:
                #################################################################################################
                if DEMA_ID.split("_")[0] in ManifoldLearnerMethodsSpace:
                    self.Main_DT_Space = "ManifoldLearner"
                    Pickle = open(self.AlternativeStorageLocation + "/LookbacksPacks/"+DEMA_ID.replace("_DEMAs", ""), 'rb')
                    outPickle = pickle.load(Pickle)
                    Pickle.close()
                    rets = outPickle[0]
                else:
                    rets = localRets
                    DT_Space_ID = DEMA_ID.replace("_LookBacksPack_DEMAs","")
                #################################################################################################
                DEMAs = pe.readPickleDF(TargetDEMA_File).fillna(0)
                for MacroMode in [""]:#"_diff"
                    print("Check & Filter with the Availability of Indicators")
                    #PnL_Copy_DF = pnlDF.copy()
                    exPostPnlDF = pe.readPickleDF(self.AlternativeStorageLocation + '/DT_Filters_Alphas/DT_MacroFactors_PnLs/' + MacroMode + "_" + DT_Space_ID + "_exPostPnlDF.p")
                    #######################################################################################################
                    PnL_Copy_DF = exPostPnlDF.copy()
                    PnL_Copy_DF = PnL_Copy_DF[[c for c in PnL_Copy_DF.columns if c.split(",")[0] in list(self.EligibleForBestFactors.index)]]
                    ExpSharpe_DF = np.sqrt(252) * pe.expanderMetric(PnL_Copy_DF, "Sharpe", 25)
                    pe.savePickleDF(ExpSharpe_DF, self.AlternativeStorageLocation + '/DT_Filters_Alphas/DT_MacroFactors_PnLs/' + MacroMode + "_" + DT_Space_ID + "_ExpSharpe_DF.p")
                    for c in tqdm(ExpSharpe_DF.columns):
                        try:
                            #print(c)
                            #print(self.IndicatorsDF_Raw[c.split(",")[0]].isna().astype(float).sum())
                            #print(len(ExpSharpe_DF[c]))
                            #ExpSharpe_DF[c].plot()
                            #plt.show()
                            ExpSharpe_DF.loc[self.IndicatorsDF_Raw.loc[self.IndicatorsDF_Raw[c.split(",")[0]].isna()==True].index,c] = None
                        except Exception as e:
                            print(e)
                            print(c)
                            print(self.IndicatorsDF_Raw[c.split(",")[0]].isna()==True)
                            ExpSharpe_DF[c].plot()
                            plt.show()
                            time.sleep(3000)
                    #######################################################################################################
                    if self.Store_Metrics:
                        N = 100
                        if N >= ExpSharpe_DF.shape[1]:
                            N = round(ExpSharpe_DF.shape[1]/2)
                        print("Saving PnL_Copy_DF_Chunks and ExpSharpe_DF_Chunks ... ")
                        ######################################################################################
                        ExpSharpe_DF_Chunks = pe.chunkMaker(ExpSharpe_DF, 1, N) #ReturnIntervals="yes"
                        for elem in tqdm(ExpSharpe_DF_Chunks):
                            out1 = pe.savePickleDF(elem, self.AlternativeStorageLocation+'/DT_ExpandingMetrics/'+MacroMode+"_"+DT_Space_ID+"_"+str(ExpSharpe_DF.columns.get_loc(elem.columns[0]))+"_"+str(ExpSharpe_DF.columns.get_loc(elem.columns[-1]))+"_ExpSharpe")
                        ######################################################################################
                        PnL_Copy_DF_Chunks = pe.chunkMaker(PnL_Copy_DF, 1, N)
                        for elem in tqdm(PnL_Copy_DF_Chunks):
                            out0 = pe.savePickleDF(elem, self.AlternativeStorageLocation+'/DT_Filters_PnLs/'+MacroMode+"_"+DT_Space_ID+"_"+str(PnL_Copy_DF.columns.get_loc(elem.columns[0]))+"_"+str(PnL_Copy_DF.columns.get_loc(elem.columns[-1]))+"_PnL")
                    ######################## FINDING DT FEATURES ##################################
                    print("FINDING DT KILLERS FEATURES")
                    for asset in tqdm(rets.columns):
                        if KillersFinderSettings['outputFormat'] == 'excel':
                            suffix = 'xlsx'
                        elif KillersFinderSettings['outputFormat'] == 'pickle':
                            suffix = 'p'
                        StoringExcelFile = self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/' + DT_Space_ID + "_" + asset + '.' + suffix
                        ExploreSpace = ExpSharpe_DF[[c for c in ExpSharpe_DF.columns if c.split(",")[1]==asset]]
                        ####################################################
                        if self.top_bottom_columns_Mode == "Setup":
                            result_df = pe.top_bottom_columns(ExploreSpace, Normaliser="Raw", Scalers=[np.sqrt(252), np.sqrt(252)])
                        elif self.top_bottom_columns_Mode == "Update":
                            new_result_df = pe.top_bottom_columns(ExploreSpace.iloc[-50:,:], Normaliser="Raw", Scalers=[np.sqrt(252), np.sqrt(252)])
                            if KillersFinderSettings['outputFormat'] == 'excel':
                                result_df = pe.updateExcelDF(StoringExcelFile, new_result_df)
                            elif KillersFinderSettings['outputFormat'] == 'pickle':
                                pe.updatePickleDF(new_result_df, StoringExcelFile)
                        ####################################################
                        result_df = pe.stringReplace(result_df, [","+asset+",TreeThreshold_Raw_RollWindow_250", "'", "[", "]"], "")
                        ####################################################
                        if KillersFinderSettings['outputFormat'] == 'excel':
                            result_df.to_excel(StoringExcelFile)
                        elif KillersFinderSettings['outputFormat'] == 'pickle':
                            pe.savePickleDF(result_df,StoringExcelFile)

    ###########################################################################################################################

    def Expanding_Top_Bottom_Features(self):
        #####################################################################################################
        print("Identifying UniquePacks ... ")
        UniquePacksList = []
        for TargetFile in glob.glob(self.AlternativeStorageLocation+'/DT_ExpandingMetrics/*'):
            TargetFileID = TargetFile.split("\\")[-1]
            UniquePacksList.append('_'.join(TargetFileID.split("_")[:-3]))
        UniquePacks = list(set(UniquePacksList))
        #####################################################################################################
        print("Identifying DT_Filters_Top_Bottom_Features ... ")
        for pack in UniquePacks:
            packData = []
            for TargetFile in tqdm(glob.glob(self.AlternativeStorageLocation + '/DT_ExpandingMetrics/*')):
                TargetFileID = TargetFile.split("\\")[-1]
                if '_'.join(TargetFileID.split("_")[:-3]) == pack:
                    out = pe.readPickleDF(TargetFile)
                    packData.append(out)
            packDF = pd.concat(packData, axis=1).sort_index()
            ##################################################################################################
            result_df = pe.top_bottom_columns(packDF, Normaliser="Raw", Scalers=[np.sqrt(252),np.sqrt(252)])
            result_df.to_excel(self.AlternativeStorageLocation+'/DT_Filters_Top_Bottom_Features/'+pack+'.xlsx')

    def Expanding_Top_Botton_Features_Analytics(self):
        #############################################################################################################################
        DT_Filters_sh = pd.read_excel(self.AlternativeStorageLocation + "/DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_RetsLive.xlsx")
        DT_Filters_sh = DT_Filters_sh.dropna(subset=["Sharpe"])
        MacroFactors = list(set(DT_Filters_sh["ID"].str.split(",").str[0]+"_"+DT_Filters_sh["ID"].str.split("_").str[-1].tolist()))
        TargetAssetsList = list(set(DT_Filters_sh["Asset"].tolist()))
        print("len(MacroFactors) = ", len(MacroFactors))
        print("len(TargetAssetsList) = ", len(TargetAssetsList))
        #############################################################################################################################
        TimeDependent_DT_Validators_List = []
        TimeDependent_DT_Validators_Analytics_List = []
        for TargetAsset in tqdm(TargetAssetsList):
            for TargetFile in glob.glob(self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/*xlsx'):
                if '_DefaultSingleLookBack_' + TargetAsset in TargetFile:
                    for TargetFactor in tqdm(MacroFactors):
                        ###############################################################################################################################
                        Top_Bottom_Features_Cols = ["Top_5_Columns", "Bottom_5_Columns"]
                        TBF = pd.read_excel(TargetFile)
                        TBF = TBF.rename(columns={TBF.columns[0]:'date'}).set_index("date", drop=True)[Top_Bottom_Features_Cols]
                        for TargetSpace in Top_Bottom_Features_Cols:
                            ###############################################################################################################################
                            TargetFactor_Existence = (TBF[TargetSpace].str.contains(TargetFactor)).astype(int).reset_index().reset_index()
                            TargetFactor_Existence["index"] += 1
                            TargetFactor_Existence[TargetSpace+"_Freq"] = pe.cs(TargetFactor_Existence[TargetSpace]) / TargetFactor_Existence["index"]
                            TargetFactor_Existence = TargetFactor_Existence[["date", TargetSpace+"_Freq"]].set_index("date", drop=True)
                            ###########################################################################################################################
                            TargetFactor_Existence.columns = [TargetAsset+","+TargetFactor+","+TargetSpace]
                            ###########################################################################################################################
                            TimeDependent_DT_Validators_List.append(TargetFactor_Existence)
                            TimeDependent_DT_Validators_Analytics_List.append([TargetAsset, TargetFactor, TargetSpace, TargetFactor_Existence.iloc[-1].values[0]])
        ###################################################################################################################################################
        TimeDependent_DT_Validators_DF = pd.concat(TimeDependent_DT_Validators_List, axis=1)
        pe.savePickleDF(TimeDependent_DT_Validators_DF, self.AlternativeStorageLocation+"TimeDependent_DT_Validation/TimeDependent_DT_Validation_DF")
        TimeDependent_DT_Validators_Analytics_DF = pd.DataFrame(TimeDependent_DT_Validators_Analytics_List,columns=["TargetAsset", "TargetFactor", "TargetSpace", "CumulativeParticipation"])
        pe.savePickleDF(TimeDependent_DT_Validators_Analytics_DF, self.AlternativeStorageLocation+"TimeDependent_DT_Validation/TimeDependent_DT_Validators_Analytics_DF")
        TimeDependent_DT_Validators_Analytics_DF.to_excel(self.AlternativeStorageLocation+"TimeDependent_DT_Validation/TimeDependent_DT_Validators_Analytics_DF.xlsx", index=False)

    def Expanding_ScanOptimals(self, mode):
        if mode == "excelOptimals_Analytics":
            TimeDependent_DT_Validators_Analytics_DF = pd.read_excel(self.AlternativeStorageLocation+"TimeDependent_DT_Validation/TimeDependent_DT_Validators_Analytics_DF.xlsx")
            TimeDependent_DT_Validators_Analytics_DF = TimeDependent_DT_Validators_Analytics_DF.set_index("TargetAsset", drop=True)
            #####################################################################################################################################################
            ActiveAssets = pd.read_excel(self.AssetsDashboardExcel, sheet_name="ActiveStrategiesFactorsControl")["Asset"].tolist()
            #####################################################################################################################################################
            OptimalsList = []
            for asset in ActiveAssets:
                try:
                    subDF = TimeDependent_DT_Validators_Analytics_DF.loc[asset].sort_values(by="CumulativeParticipation", ascending=False)
                    subDF_Lower = subDF[subDF["TargetSpace"] == "Bottom_5_Columns"]
                    subDF_Upper = subDF[subDF["TargetSpace"] == "Top_5_Columns"]
                    OptimalsList.append(subDF_Lower.iloc[0,:])
                    OptimalsList.append(subDF_Upper.iloc[0,:])
                except Exception as e:
                    print(e)
            #####################################################################################################################################################
            OptimalsDF = pd.concat(OptimalsList,axis=1).T
            print(OptimalsDF)
            OptimalsDF.to_excel(self.AlternativeStorageLocation + "TimeDependent_DT_Validation/TimeDependent_DT_Validators_Analytics_OptimalsDF.xlsx")

        elif mode == "dynamics":
            TimeDependent_DT_Validators_DF = pe.readPickleDF(self.AlternativeStorageLocation+"TimeDependent_DT_Validation/TimeDependent_DT_Validation_DF")
            #########################################################################
            LastLine = TimeDependent_DT_Validators_DF.iloc[-1, :]
            #########################################################################
            for TargetAsset in self.ActiveStrategiesFactorsControl.index:
                try:
                    topSel = 10
                    Query_DT = LastLine.loc[[x for x in LastLine.index if TargetAsset in x]].sort_values(ascending=False)
                    Best_DT_Killers_Bottom = TimeDependent_DT_Validators_DF[[x for x in Query_DT.index[:topSel] if "Bottom" in x]]
                    Best_DT_Killers_Top = TimeDependent_DT_Validators_DF[[x for x in Query_DT.index[:topSel] if "Top" in x]]
                    if TargetAsset == "EC1 Curncy":
                        print(Query_DT)
                        print(Best_DT_Killers_Bottom.columns)
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                        print(Best_DT_Killers_Top.columns)
                        #time.sleep(3000)
                        #########################################################################
                        fig, ax = plt.subplots(nrows=2, ncols=1, )
                        fig.set_size_inches(18.5, 10.5)
                        Best_DT_Killers_Bottom.plot(ax=ax[0])
                        Best_DT_Killers_Top.plot(ax=ax[1])
                        # plt.show()
                        plt.savefig(self.AlternativeStorageLocation + "TimeDependent_DT_Validation/Images/" + TargetAsset,
                            dpi=100)
                    else:
                        pass
                except Exception as e:
                    print(e)

    def Expanding_Top_Bottom_Features_Trader(self):

        TargetAssetsList = self.ActiveAssets
        for Top_Bottom_Features_Cols in [['Top_5_Columns', 'Top_5_Values'], ['Bottom_5_Columns', 'Bottom_5_Values']]:
            for ThrDirection in ["Upper", "Lower"]:
                for TargetAsset in tqdm(TargetAssetsList):
                    if True:#TargetAsset == "NQ1 Index":#("1 Comdty" in TargetAsset)|("2 Comdty" in TargetAsset):
                        ####################################################################################################################################
                        for TargetFile in glob.glob(self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/*'):
                            if '_DefaultSingleLookBack_'+TargetAsset in TargetFile:
                                ############################################################################################################################
                                TBF_main = pd.read_excel(TargetFile)
                                TBF_main = TBF_main.rename(columns={TBF_main.columns[0]: 'date'}).set_index("date", drop=True)
                                TBF = TBF_main[Top_Bottom_Features_Cols[0]]
                                TBF_Sh = TBF_main[Top_Bottom_Features_Cols[1]]
                                ############################################################################################################################
                                TBF = TBF.str.split(',', expand=True)
                                TBF_Sh = TBF_Sh.str.split(',', expand=True).fillna(0)
                                ############################################################################################################################
                                TBF_direction = TBF.copy()
                                UniqueFactorsList = []
                                for c in TBF.columns:
                                    TBF[c] = TBF[c].str.replace("[","").str.replace("]","")
                                    TBF_Sh[c] = TBF_Sh[c].str.replace("[","").str.replace("]","")
                                    TBF_direction[c] = TBF_direction[c].str.replace("[","").str.replace("]","")
                                for c in TBF_direction.columns:
                                    TBF[c] = TBF[c].str.split("_").str[0]
                                    TBF[c] = TBF[c].fillna('').apply(lambda x: x[1:] if x.startswith(' ') else x)#.str[:-1]
                                    UniqueFactorsList += list(set(TBF[c].dropna().drop_duplicates().tolist()))
                                    TBF_direction[c] = TBF_direction[c].str.split("_").str[1]
                                ############################################################################################################################
                                TBF_direction.to_sql(TargetAsset + "_TBF_direction_" + Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                                ############################################################################################################################
                                UniqueFactorsList = [x for x in list(set(UniqueFactorsList)) if len(x)>1]
                                subFactorsSigDF = pd.DataFrame(None, index=TBF.index, columns=TBF.columns)
                                for f in UniqueFactorsList:
                                    mFactorSigDF = pe.readPickleDF(self.AlternativeStorageLocation + '/DT_Filters_Alphas\DT_MacroFactors_Signals/'+f+'_RollWindow_250_'+ThrDirection+'.p')
                                    for c in subFactorsSigDF.columns:
                                        subFactorsSigDF.loc[TBF[c]==f,c] = mFactorSigDF[TargetAsset]
                                subFactorsSigDF = subFactorsSigDF.fillna(0)
                                ############################################################################################################################
                                "SQL SAVE"
                                TBF.to_sql(TargetAsset + "_TBF_" + Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                                ###############
                                subFactorsSigDF.to_sql(TargetAsset+"_subFactorsSigDF_"+Top_Bottom_Features_Cols[0], self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                                ###############
                                TBF_Sh = TBF_Sh.apply(pd.to_numeric, args=('coerce',)).fillna(0)
                                TBF_Sh.to_sql(TargetAsset + "_TBF_Sh_" + Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                                TBF_Sh_RowStoch = pe.rowStoch(TBF_Sh)
                                TBF_Sh_RowStoch.to_sql(TargetAsset + "_TBF_Sh_RowStoch_" + Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                                ############################################################################################################################
                                for SignalAggregationMethod in ["SharpeWeighted_RowStoch", "Sum"]:
                                    if SignalAggregationMethod == "SharpeWeighted_RowStoch":
                                        SignalAggrDF = (subFactorsSigDF*TBF_Sh_RowStoch).sum(axis=1)
                                    else:
                                        SignalAggrDF = pe.sign(subFactorsSigDF.sum(axis=1))
                                    SignalAggrDF.to_sql(TargetAsset + "_SignalAggregator_"+SignalAggregationMethod+"_"+ Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')

    def Expanding_Top_Bottom_Features_SignalAggregatorTOTAL_BackTest(self):
        #############################################################################################################################
        LookBacksPickle = open(self.AlternativeStorageLocation + "/LookbacksPacks/_DefaultSingleLookBack_LookBacksPack",'rb')
        LookBacksPack = pickle.load(LookBacksPickle)
        LookBacksPickle.close()
        rets = LookBacksPack[0]
        #############################################################################################################################
        TargetAssetsList = self.ActiveAssets
        for Top_Bottom_Features_Cols in [['Top_5_Columns', 'Top_5_Values'], ['Bottom_5_Columns', 'Bottom_5_Values']]:
            "Get All Aggregated Signals into one Dataframe"
            for SignalAggregationMethod in ["SharpeWeighted_RowStoch", "Sum"]:
                SignalAggregator_List = []
                for TargetAsset in tqdm(TargetAssetsList):
                    subSig = pd.read_sql("SELECT * FROM '"+TargetAsset + "_SignalAggregator_"+SignalAggregationMethod+"_"+ Top_Bottom_Features_Cols[0]+"'", self.DT_MacroFactors_Signals_Conn).set_index('date',drop=True)
                    subSig.columns = [TargetAsset]
                    SignalAggregator_List.append(subSig)
                SignalAggregatorDF = pd.concat(SignalAggregator_List,axis=1).sort_index()
                SignalAggregatorDF.to_sql("SignalAggregatorTOTAL_" + SignalAggregationMethod + "_" + Top_Bottom_Features_Cols[0],self.DT_MacroFactors_Signals_Conn, if_exists='replace')
                ###################################################################################################
                pnl = pe.S(SignalAggregatorDF,nperiods=2) * rets
                pnl["TOTAL"] = pe.rs(pnl)
                sh = np.sqrt(252) * pe.sharpe(pnl)
                print(sh)
                ###################################################################################################
                pe.cs(pnl).plot()
                plt.show()

###################################################################################################

def MainRunner():

    print("Running Time Depencent Decision Trees (DT) Validator ... ")
    obj = TD_DT_Validation("RetsLive")
    print("Extracting DT Filters' Alphas ... ")
    #obj.DT_Filters_Alphas("Setup")
    obj.DT_Filters_Alphas("Update", ExpandingSharpe="NO",InactivityThr = 0.9)
    print("Ready!")

def AdHocRunner():
    for DT_Space in ["RetsLive"]: ##diffRets_ExpWindow, "RetsExpWindow25", "RetsLive"
        obj = TD_DT_Validation(DT_Space)
        ###############################################################################################
        #obj.DT_Filters_Alphas(InactivityThr = 0.9)
        #obj.CreateExpandingSharpes("Setup", KillersFinderSettings = {'outputFormat':'excel'})
        ###############################################################################################
        #obj.Expanding_Top_Bottom_Features()
        #obj.Expanding_Top_Botton_Features_Analytics()
        #obj.Expanding_ScanOptimals("excelOptimals_Analytics")
        #obj.Expanding_ScanOptimals("dynamics")
        ###############################################################################################
        #obj.Expanding_Top_Bottom_Features_Trader()
        obj.Expanding_Top_Bottom_Features_SignalAggregatorTOTAL_BackTest()

###################################################################################################

#MainRunner()
AdHocRunner()