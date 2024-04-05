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
        #self.Main_DT_Space = "RetsRollWin250" #"Rets","RetsLive","RetsRollWin250"
        #"""
    def DT_Filters_Alphas(self, top_bottom_columns_Mode, **kwargs):
        if "Alphas_Sh_ExcelModeOnly" in kwargs:
            Alphas_Sh_ExcelModeOnly = kwargs['Alphas_Sh_ExcelModeOnly']
        else:
            Alphas_Sh_ExcelModeOnly = "NO"
        ########################################################################################################
        self.top_bottom_columns_Mode = top_bottom_columns_Mode
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
                    for TargetFile in tqdm(glob.glob(DT_Folder+'*')):
                        print(TargetFile)
                        if ("BeforeUpdate" not in TargetFile):
                        #if ("BeforeUpdate" not in TargetFile)&("DecisionTrees_RV_Rets_ExpWindow_25_USYC5Y20 Index" in TargetFile):
                            try:
                                MacroFactor = TargetFile.split("\\")[-1].split("_")[-1]
                                print(MacroFactor)
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
                                    pnl = (pe.S(sig, nperiods=2) * rets).sort_index()
                                    pnl = pnl[self.ActiveAssets]
                                    """
                                    #tempAssets = ["EC1 Curncy"]
                                    tempAssets = ["EC1 Curncy", "BP1 Curncy"]
                                    #tempAssets = ["JY1 Curncy", "BP1 Curncy"]
                                    print(sig[tempAssets].tail())
                                    print(SigFilter[tempAssets].tail())
                                    print("Sharpe")
                                    print(np.sqrt(252) * pe.sharpe(pnl[tempAssets]))
                                    fig, ax = plt.subplots(nrows=4, ncols=1)
                                    pe.cs(pnl[tempAssets]).plot(ax=ax[0])
                                    SigFilter[tempAssets].plot(ax=ax[1], title="SigFilter")
                                    sig[tempAssets].plot(ax=ax[2], title="sig")
                                    pd.concat([DT_RV[[x for x in DT_RV.columns if (tempAssets[0] in x)|(tempAssets[1] in x)]],self.IndicatorsDF[MacroFactor]], axis=1).plot(ax=ax[3])
                                    plt.show()
                                    #"""
                                    pnl.columns = [MacroFactor + "," + x + ",TreeThreshold_" + filterMode + "_" + RollMode + "_" + mode for x in pnl.columns]
                                    pnlList.append(pnl)
                            except Exception as e:
                                print(e)
                    #######################################################################################################
                    pnlDF = pd.concat(pnlList, axis=1).sort_index()
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
                    ReportDF.to_excel(self.AlternativeStorageLocation+"/DT_Filters_Alphas/"+MacroMode+"_"+DT_Space_ID+"_Alphas_DT_Filters_sh_"+self.Main_DT_Space+".xlsx", index=False)
                    ######################################### EXPANDING SHARPES ###########################################
                    if Alphas_Sh_ExcelModeOnly == "NO":
                        print("Calculating Expanding Metrics ... ")
                        PnL_Copy_DF = pnlDF.copy()
                        ExpSharpe_DF = pe.expanderMetric(PnL_Copy_DF, "Sharpe", 25)
                        print("Check & Filter with the Availability of Indicators")
                        for c in tqdm(ExpSharpe_DF.columns):
                            ExpSharpe_DF.loc[self.IndicatorsDF_Raw.loc[self.IndicatorsDF_Raw[c.split(",")[0]].isna()==True].index,c] = None
                        #######################################################################################################
                        if self.Store_Metrics:
                            N = 1000
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
                            StoringExcelFile = self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/' + DT_Space_ID + "_" + asset + '.xlsx'
                            ExploreSpace = ExpSharpe_DF[[c for c in ExpSharpe_DF.columns if c.split(",")[1]==asset]]
                            ####################################################
                            if self.top_bottom_columns_Mode == "Setup":
                                result_df = pe.top_bottom_columns(ExploreSpace, Normaliser="Raw", Scalers=[np.sqrt(252), np.sqrt(252)])
                            elif self.top_bottom_columns_Mode == "Update":
                                new_result_df = pe.top_bottom_columns(ExploreSpace.iloc[-50:,:], Normaliser="Raw", Scalers=[np.sqrt(252), np.sqrt(252)])
                                result_df = pe.updateExcelDF(StoringExcelFile, new_result_df)
                            ####################################################
                            result_df = pe.stringReplace(result_df, [","+asset+",TreeThreshold_Raw_RollWindow_250", "'", "[", "]"], "")
                            ####################################################
                            result_df.to_excel(StoringExcelFile)

    def Top_Bottom_Features(self, InactivityThr):
        shDF = pd.read_excel(self.AlternativeStorageLocation+"/DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_"+self.Main_DT_Space+".xlsx").set_index("ID",drop=True)
        ##############################################################################################################################
        shDF = shDF[shDF["Inactivity"] <= InactivityThr]
        shDF_InactivityReport = shDF.copy().reset_index()
        shDF_InactivityReport.to_excel(self.AlternativeStorageLocation + "/DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_Inactivity_"+str(InactivityThr).replace(".","")+"_"+self.Main_DT_Space+".xlsx",index=False)
        ##############################################################################################################################
        shDF = shDF.reset_index()
        self.EligibleForBestFactors = self.IndicatorsData[["Indicator", "EligibleForBestFactorsList"]].set_index("Indicator",drop=True)
        self.EligibleForBestFactors = self.EligibleForBestFactors[self.EligibleForBestFactors["EligibleForBestFactorsList"]!=0]
        shDF = shDF[shDF["MacroFactor"].isin(list(self.EligibleForBestFactors.index))]
        ##############################################################################################################################
        Top_Bottom_Features_List = []
        for Asset in list(set(shDF["Asset"].tolist())):
            if True:#Asset == "EC1 Curncy":
                #try:
                subDF = shDF.loc[shDF['Asset']==Asset,:].dropna(subset=["Sharpe"]).sort_values(by='Sharpe',ascending=False).set_index('Asset',drop=True)
                maxPerformingFactor = subDF.iloc[0,:]
                minPerformingFactor = subDF.iloc[-1,:]
                Top_Bottom_Features_List.append([Asset,maxPerformingFactor["MacroFactor"]+"_"+maxPerformingFactor["ID"].split("_")[-1],minPerformingFactor["MacroFactor"]+"_"+minPerformingFactor["ID"].split("_")[-1]])
            #if Asset == "EC1 Curncy":
            #    print([Asset,maxPerformingFactor["MacroFactor"]+"_"+maxPerformingFactor["ID"].split("_")[-1],minPerformingFactor["MacroFactor"]+"_"+minPerformingFactor["ID"].split("_")[-1]])
            #    time.sleep(30000)
            #except Exception as e:
            #    print(e)
        BestFactorsDF = pd.DataFrame(Top_Bottom_Features_List,columns=["Asset","SingleDecisionTreesControllers_Positive","SingleDecisionTreesControllers_Negative"]).set_index("Asset",drop=True)
        BestFactorsDF = pd.concat([BestFactorsDF, self.ActiveStrategiesFactorsControl["VolControllers"]],axis=1).loc[self.ActiveStrategiesFactorsControl.index,:]
        BestFactorsDF.to_excel(self.AlternativeStorageLocation+"/TimeDependent_DT_Validation/BestFactorsDF_"+self.Main_DT_Space+".xlsx")

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
            print(pack)
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
        MainID = "_DefaultSingleLookBack"
        #############################################################################################################################
        DT_Filters_sh = pd.read_excel(self.AlternativeStorageLocation + "/DT_Filters_Alphas/_"+MainID+"_Alphas_DT_Filters_sh.xlsx")
        DT_Filters_sh = DT_Filters_sh.dropna(subset=["Sharpe"])
        MacroFactors = list(set(DT_Filters_sh["ID"].str.split(",").str[0]+"_"+DT_Filters_sh["ID"].str.split("_").str[-1].tolist()))
        TargetAssetsList = list(set(DT_Filters_sh["Asset"].tolist()))
        print("len(MacroFactors) = ", len(MacroFactors))
        print("len(TargetAssetsList) = ", len(TargetAssetsList))
        #############################################################################################################################
        TimeDependent_DT_Validators_List = []
        TimeDependent_DT_Validators_Analytics_List = []
        for TargetAsset in tqdm(TargetAssetsList):
            for TargetFile in glob.glob(self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/*'):
                if '_DefaultSingleLookBack_' + TargetAsset in TargetFile:
                    for TargetFactor in tqdm(MacroFactors):
                        ###############################################################################################################################
                        Top_Bottom_Features_Cols = ["Top_5_Columns", "Bottom_5_Columns"]
                        TBF = pd.read_excel(TargetFile).set_index("date", drop=True)[Top_Bottom_Features_Cols]
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
        #############################################################################################################################
        LookBacksPickle = open(self.AlternativeStorageLocation + "/LookbacksPacks/_DefaultSingleLookBack_LookBacksPack",'rb')
        LookBacksPack = pickle.load(LookBacksPickle)
        LookBacksPickle.close()
        rets = LookBacksPack[0]
        #############################################################################################################################
        DEMAsPickle = open(self.AlternativeStorageLocation + "/DEMAs/_DefaultSingleLookBack_LookBacksPack_DEMAs",'rb')
        DEMAs = pickle.load(DEMAsPickle)
        DEMAsPickle.close()
        DEMAs = DEMAs.fillna(0)
        #############################################################################################################################
        sig = pd.DataFrame(0, index=DEMAs.index, columns=DEMAs.columns)
        targetSpace = "Bottom"
        #TargetAssetsList = DEMAs.columns
        TargetAssetsList = self.ActiveAssets
        for TargetAsset in tqdm(TargetAssetsList):
            if True:#TargetAsset == "NQ1 Index":#("1 Comdty" in TargetAsset)|("2 Comdty" in TargetAsset):
                print(TargetAsset)
                #############################################################################################################################
                for TargetFile in glob.glob(self.AlternativeStorageLocation + '/DT_Filters_Top_Bottom_Features/*'):
                    if '_DefaultSingleLookBack_'+TargetAsset in TargetFile:
                        #############################################################################################################################
                        TBF = pd.read_excel(TargetFile).set_index("date", drop=True)[["Top_5_Columns","Bottom_5_Columns"]]
                        ChangeDF = pd.DataFrame(TBF[targetSpace+"_5_Columns"].str.replace("[","").str.replace("]","").str.replace(","+TargetAsset+",TreeThreshold_Raw_RollWindow_250_Lower","").str.replace("'",""))
                        #print(ChangeDF.iloc[:,0].tail())
                        #f_asset = ChangeDF.iloc[:,0].str.contains('OVX Index', regex=False).astype(int)
                        #s_asset = ChangeDF.iloc[:,0].str.contains('VXN Index', regex=False).astype(int)
                        #t_asset = ChangeDF.iloc[:,0].str.contains('VIX Index', regex=False).astype(int)
                        #print(f_asset.sum(), s_asset.sum(), t_asset.sum())
                        #print(f_asset.sum()/(ChangeDF.shape[0]-500), s_asset.sum()/(ChangeDF.shape[0]-500), t_asset.sum()/(ChangeDF.shape[0]-500))
                        #fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
                        #f_asset.plot(ax=ax[0])
                        #s_asset.plot(ax=ax[1])
                        #t_asset.plot(ax=ax[2])
                        #plt.show()
                        #time.sleep(30000)
                        """
                        ChangeDF["FirstIsInPrevious"] = 0
                        ChangeDF["SecondIsInPrevious"] = 0
                        ChangeDF["ThirdIsInPrevious"] = 0
                        sChangeDF = pe.S(ChangeDF)
                        for idx,row in tqdm(ChangeDF.iterrows()):
                            try:
                                #if idx == "2023-09-05 00:00:00":
                                #    print(sBottomDF.loc[idx, "Bottom_5_Columns"])
                                #    print(BottomDF.loc[idx,"Bottom_5_Columns"].split(", ")[0])
                                #    print(BottomDF.loc[idx,"Bottom_5_Columns"].split(", ")[1])
                                #    print(BottomDF.loc[idx,"Bottom_5_Columns"].split(", ")[2])
                                #    time.sleep(30000)
                                if ChangeDF.loc[idx,targetSpace+"_5_Columns"].split(", ")[0] in sChangeDF.loc[idx,targetSpace+"_5_Columns"]:
                                    ChangeDF.loc[idx,"FirstIsInPrevious"] = 1
                                if ChangeDF.loc[idx,targetSpace+"_5_Columns"].split(", ")[1] in sChangeDF.loc[idx,targetSpace+"_5_Columns"]:
                                    ChangeDF.loc[idx,"SecondIsInPrevious"] = 1
                                if ChangeDF.loc[idx,targetSpace+"_5_Columns"].split(", ")[2] in sChangeDF.loc[idx,targetSpace+"_5_Columns"]:
                                    ChangeDF.loc[idx,"ThirdIsInPrevious"] = 1
                            except Exception as e:
                                print(e)
                        print(100*(ChangeDF[["FirstIsInPrevious","SecondIsInPrevious","ThirdIsInPrevious"]].sum()/(ChangeDF.shape[0]-500)))
                        print(ChangeDF[["FirstIsInPrevious","SecondIsInPrevious","ThirdIsInPrevious"]].tail(25))
                        ChangeDF[["FirstIsInPrevious","SecondIsInPrevious","ThirdIsInPrevious"]].plot()
                        plt.show()
                        """
                #############################################################################################################################
                specDT_RV_Filter_List = []
                for dtFile in glob.glob(self.AlternativeStorageLocation + "/DecisionTrees/RetsLive/*RollWindow_250*"):
                    if "_BeforeUpdate" not in dtFile:
                        DT_RV = pe.readPickleDF(dtFile)
                        dtFile_MacroFactor = dtFile.split("_")[-1]
                        try:
                            DT_RV_Filters = pe.sign(DT_RV.sub(self.IndicatorsDF[dtFile_MacroFactor], axis=0).sort_index())[dtFile_MacroFactor + "," + TargetAsset + ",TreeThreshold"]
                            specDT_RV_Filter_List.append(DT_RV_Filters)
                        except Exception as e:
                            print(e)
                specDT_RV_Filter_DF = pd.concat(specDT_RV_Filter_List, axis=1).fillna(0)
                #print(specDT_RV_Filter_DF.tail())
                #specDT_RV_Filter_DF.columns = [x.split(",")[1] for x in specDT_RV_Filter_DF.columns]
                #############################################################################################################################
                for idx, row in ChangeDF.iterrows():
                    #try:
                    #    sig.loc[idx,TargetAsset] = pe.sign(DEMAs.loc[idx,TargetAsset])
                    #except Exception as e:
                    #    print(e)
                    for sel in [0,1,2]:
                        try:
                            f_driver = row.str.split(", ").str[sel]
                            subFilter = specDT_RV_Filter_DF.loc[idx, f_driver + "," + TargetAsset + ",TreeThreshold"].values[0]
                            if "Upper" in TBF.loc[idx,targetSpace+"_5_Columns"].split("', '")[sel]:
                                if subFilter > 0:
                                    subFilter = 0
                            elif "Lower" in TBF.loc[idx,targetSpace+"_5_Columns"].split("', '")[sel]:
                                if subFilter < 0:
                                    subFilter = 0
                            #print(abs(subFilter))
                            #sig.loc[idx,TargetAsset] = pe.sign(DEMAs.loc[idx,TargetAsset])*abs(subFilter)
                            sig.loc[idx,TargetAsset] += pe.sign(DEMAs.loc[idx,TargetAsset])*abs(subFilter)
                            #sig.loc[idx,TargetAsset] = pe.sign(sig.loc[idx,TargetAsset]+(pe.sign(DEMAs.loc[idx,TargetAsset])*abs(subFilter)))
                        except Exception as e:
                            pass
                            #print(e)
        #############################################################################################################################
        sig = pe.sign(sig)
        #############################################################################################################################
        pnl = (pe.S(sig, nperiods=2) * rets).fillna(0)
        pnl["TOTAL"] = pe.rs(pnl)
        sh = np.sqrt(252) * pe.sharpe(pnl)
        print(sh.dropna())

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        pe.cs(DEMAs).plot(ax=ax[0])
        pe.cs(pnl).plot(ax=ax[1])
        plt.show()

            #Changes = (out != pe.S(out)).astype(float)
            #Changes["Top_5_Columns"].plot()
            #plt.show()
            #time.sleep(30000)

    #############################################################################################################################

    def DT_Filters_Alphas_Trader(self):
        #############################################################################################################################
        LookBacksPickle = open(self.AlternativeStorageLocation + "/LookbacksPacks/_DefaultSingleLookBack_LookBacksPack",'rb')
        LookBacksPack = pickle.load(LookBacksPickle)
        LookBacksPickle.close()
        rets = LookBacksPack[0]
        #############################################################################################################################
        DEMAsPickle = open(self.AlternativeStorageLocation + "/DEMAs/_DefaultSingleLookBack_LookBacksPack_DEMAs",'rb')
        DEMAs = pickle.load(DEMAsPickle)
        DEMAsPickle.close()
        DEMAs = DEMAs.fillna(0)
        #############################################################################################################################
        sh_Space = '_Raw_RollWindow_250_'
        Alphas_DT_Filters_shDF = pd.read_excel(self.AlternativeStorageLocation + "/DT_Filters_Alphas/__DefaultSingleLookBack_Alphas_DT_Filters_sh_RetsLive.xlsx")
        Alphas_DT_Filters_shDF['DT_Base'] = Alphas_DT_Filters_shDF['ID'].str.split(sh_Space).str[0]
        Alphas_DT_Filters_shDF['DT_Direction'] = Alphas_DT_Filters_shDF['ID'].str.split(sh_Space).str[1]
        Alphas_DT_Filters_shDF = Alphas_DT_Filters_shDF.set_index('DT_Base',drop=True)
        "MinMax Scale Sharpe"
        #Alphas_DT_Filters_shDF['Sharpe'] = (Alphas_DT_Filters_shDF['Sharpe'] - Alphas_DT_Filters_shDF['Sharpe'].min())/(Alphas_DT_Filters_shDF['Sharpe'].max()-Alphas_DT_Filters_shDF['Sharpe'].min())
        #############################################################################################################################
        sig = DEMAs.copy()
        #TargetAssetsList = DEMAs.columns
        TargetAssetsList = self.ActiveAssets
        for TargetAsset in tqdm(TargetAssetsList):
            if TargetAsset == "EC1 Curncy":#("1 Comdty" in TargetAsset)|("2 Comdty" in TargetAsset):
                print(TargetAsset)
                #############################################################################################################################
                specDT_RV_Filter_List = []
                for dtFile in glob.glob(self.AlternativeStorageLocation + "/DecisionTrees/RetsLive/*RollWindow_250*"):
                    try:
                        if "_BeforeUpdate" not in dtFile:
                            DT_RV = pe.readPickleDF(dtFile)
                            dtFile_MacroFactor = dtFile.split("_")[-1]
                            DT_RV_Filters = pe.sign(DT_RV.sub(self.IndicatorsDF[dtFile_MacroFactor], axis=0).sort_index())[dtFile_MacroFactor + "," + TargetAsset + ",TreeThreshold"]
                            specDT_RV_Filter_List.append(DT_RV_Filters)
                    except Exception as e:
                        print(e)
                specDT_RV_Filter_DF = pd.concat(specDT_RV_Filter_List, axis=1).fillna(0)
                #############################################################################################################################
                sigFilters_List = []
                for c in specDT_RV_Filter_DF.columns:
                    filter = specDT_RV_Filter_DF[c]
                    SigFilter = filter.copy()

                    ExPost_Sh_Specs = Alphas_DT_Filters_shDF.loc[c]
                    for idx,row in ExPost_Sh_Specs.iterrows():
                        if row['DT_Direction'] == "Upper":
                            SigFilter[filter > 0] = 0
                        elif row['DT_Direction'] == 'Lower':
                            SigFilter[filter < 0] = 0
                        ##############################################################
                        sig = (pe.sign(DEMAs[TargetAsset]) * SigFilter).fillna(0)
                        sig.name = c+'_'+row['DT_Direction']
                        sigFilters_List.append(sig)

                sigFiltersDF = pd.concat(sigFilters_List,axis=1).sort_index().fillna(0)
                PnL_Copy_DF = sigFiltersDF.mul(rets[TargetAsset],axis=0).fillna(0)
                #PnL_Copy_DF = pe.S(sigFiltersDF, nperiods=2).mul(rets[TargetAsset],axis=0).fillna(0)
                print(PnL_Copy_DF.columns)

                ExpSharpe_DF = np.sqrt(252) * pe.expanderMetric(PnL_Copy_DF, "Sharpe", 25)
                print(ExpSharpe_DF.iloc[-1,:].sort_values(ascending=False))
                ExpSharpe_DF.to_excel("ExpSharpe_DF_test.xlsx")
                #time.sleep(30000)
        #############################################################################################################################
        sig = pe.sign(sig)
        #############################################################################################################################
        pnl = (pe.S(sig, nperiods=2) * rets).fillna(0)
        pnl["TOTAL"] = pe.rs(pnl)
        sh = np.sqrt(252) * pe.sharpe(pnl)
        print(sh.dropna())

        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        pe.cs(DEMAs).plot(ax=ax[0])
        pe.cs(pnl).plot(ax=ax[1])
        plt.show()

            #Changes = (out != pe.S(out)).astype(float)
            #Changes["Top_5_Columns"].plot()
            #plt.show()
            #time.sleep(30000)

###################################################################################################

def MainRunner():

    print("Running Time Depencent Decision Trees (DT) Validator ... ")
    obj = TD_DT_Validation("RetsLive")
    print("Extracting DT Filters' Alphas ... ")
    #obj.DT_Filters_Alphas("Setup")
    obj.DT_Filters_Alphas("Update", Alphas_Sh_ExcelModeOnly="NO")
    print("Setup the Top-Bottom Features ... ")
    obj.Top_Bottom_Features(0.5)
    print("Ready!")

def AdHocRunner():
    for DT_Space in ["RetsLive", "RetsExpWindow25"]: ##diffRets_ExpWindow
        obj = TD_DT_Validation(DT_Space)
        ###############################################################################################
        obj.DT_Filters_Alphas("Setup", Alphas_Sh_ExcelModeOnly="YES")
        #obj.DT_Filters_Alphas("Setup", Alphas_Sh_ExcelModeOnly="NO")
        ###############################################################################################
        #obj.DT_Filters_Alphas_Trader()
        ###############################################################################################
        obj.Top_Bottom_Features(0.9)
    ###############################################################################################
    #obj.Expanding_Top_Bottom_Features()
    #obj.Expanding_Top_Botton_Features_Analytics()
    #obj.Expanding_ScanOptimals("excelOptimals_Analytics")
    #obj.Expanding_ScanOptimals("dynamics")
    #obj.Expanding_Top_Bottom_Features_Trader()

###################################################################################################

#MainRunner()
AdHocRunner()