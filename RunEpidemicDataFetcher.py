import pandas as pd
from tqdm import tqdm
import glob, subprocess, time, os, sys
from scipy.io import loadmat

serverEpiFolder = "C:\\Users\\lucia\\Desktop\\EpidemicModel\\"
pathRawData = serverEpiFolder+"dati-regioni\dati-regioni\\"
pathNationalData = serverEpiFolder+"dati-regioni\dati-andamento-nazionale\\"
pathWorkingData = serverEpiFolder+'WorkingDataset\\'
populationData = serverEpiFolder+"dati-regioni\dati-statistici-riferimento\\"
targetDataColumns = ["data", "ricoverati_con_sintomi", "terapia_intensiva", "totale_ospedalizzati", "isolamento_domiciliare",
                    "totale_positivi", "variazione_totale_positivi", "nuovi_positivi", "dimessi_guariti", "deceduti",
                    "casi_da_sospetto_diagnostico", "casi_da_screening", "totale_casi", "tamponi", "casi_testati"]

def DatasetBuilder(mode):

    if mode == 'population':
        populationDF = pd.read_csv(populationData+"popolazione-istat-regione-range.csv")
        print(populationDF.columns)
        grouppedRegions_populationDF = populationDF.groupby(populationDF['denominazione_regione'])["totale_generale"].sum().reset_index()
        grouppedRegions_populationDF["denominazione_regione"] = grouppedRegions_populationDF["denominazione_regione"].replace("Trento", "P.A. Trento")
        grouppedRegions_populationDF["denominazione_regione"] = grouppedRegions_populationDF["denominazione_regione"].replace("Bolzano", "P.A. Bolzano")
        grouppedRegions_populationDF.to_excel("PopulationDF.xlsx", index=False)

    elif mode == 'Raw':
        for sub_path_configs in [[pathRawData, ''], [pathNationalData, 'National']]:
            sub_path = sub_path_configs[0]
            sub_path_name = sub_path_configs[1]
            dfList = []
            for fileIn in tqdm(glob.glob(sub_path + "*.csv")):
                subDF = pd.read_csv(fileIn)
                dfList.append(subDF)
            dfAll_Raw = pd.concat(dfList, axis=0).drop_duplicates()
            dfAll_Raw['data'] = dfAll_Raw['data'].astype(str).str.split('T').str[0]
            dfAll_Raw.to_excel(pathWorkingData+sub_path_name+"DataAll.xlsx", index=False)

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
                regionDataDF = regionDataDF.sort_index()
                regionDataDF.to_excel(pathWorkingData+"PerRegionTimeSeries\\"+region+".xlsx")
                regionsDataList.append(regionDataDF)
            except Exception as e:
                print(e)
        dfAll = pd.concat(regionsDataList, axis=1).sort_index()
        dfAll.to_excel(pathWorkingData + "DataRegionsTimeSeries.xlsx")

        #dfAll_toMatlab = dfAll[[x for x in dfAll.columns if ('totale_positivi' in x or 'totale_ospedalizzati' in x or 'nuovi_positivi' in x or 'dimessi_guariti' in x or 'deceduti' in x or 'totale_casi' in x or 'tamponi' in x or 'terapia_intensiva' in x or 'ricoverati_con_sintomi' in x) & ('variazione' not in x)]]
        dfAll_toMatlab = dfAll[[x for x in dfAll.columns if 'totale_positivi' in x or 'nuovi_positivi' in x or 'dimessi_guariti' in x or 'deceduti' in x]]
        dfAll[[x for x in dfAll.columns if 'dimessi_guariti' in x or 'deceduti' in x]] = dfAll[[x for x in dfAll.columns if 'dimessi_guariti' in x or 'deceduti' in x]].diff().fillna(0)
        #dfAll_toMatlab[[x for x in dfAll_toMatlab.columns if 'totale_casi' in x or 'tamponi' in x or 'dimessi_guariti' in x or 'deceduti' in x]] = dfAll_toMatlab[[x for x in dfAll_toMatlab.columns if 'totale_casi' in x or 'tamponi' in x or 'dimessi_guariti' in x or 'deceduti' in x]].diff().fillna(0)
        dfAll_toMatlab.to_excel(pathWorkingData + "dfAll_toMatlab.xlsx")

        totale_positivi_Cols = [x for x in dfAll.columns if 'totale_positivi' in x and 'variazione' not in x]
        dfAll[totale_positivi_Cols].to_excel(pathWorkingData + "totale_positivi.xlsx")

        totale_casi_Cols = [x for x in dfAll.columns if 'totale_casi' in x and 'variazione' not in x]
        dfAll[totale_casi_Cols].to_excel(pathWorkingData + "totale_casi.xlsx")

        dfAll[totale_positivi_Cols].iloc[-100:,:].to_excel(pathWorkingData + "totale_positivi_latest.xlsx")
        nuovi_positivi_Cols = [x for x in dfAll.columns if 'nuovi_positivi' in x and 'variazione' not in x]
        dfAll[nuovi_positivi_Cols].to_excel(pathWorkingData + "nuovi_positivi.xlsx")
        dimessi_guariti_Cols = [x for x in dfAll.columns if 'dimessi_guariti' in x and 'variazione' not in x]
        dfAll[dimessi_guariti_Cols].to_excel(pathWorkingData + "dimessi_guariti.xlsx")
        deceduti_Cols = [x for x in dfAll.columns if 'deceduti' in x and 'variazione' not in x]
        dfAll[deceduti_Cols].to_excel(pathWorkingData + "deceduti.xlsx")

sys.path.append(pathRawData)
os.chdir(pathRawData)
subprocess.call([serverEpiFolder+'dati-regioni\\appRun.bat'])
DatasetBuilder("population")
DatasetBuilder('Raw')
DatasetBuilder('PerRegion')
