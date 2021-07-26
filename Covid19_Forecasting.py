from Slider import Slider as sl
import numpy as np, pickle, json
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf, pickle
import matplotlib.pyplot as plt
from scipy import stats
import glob, investpy
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
import math
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from sklearn.metrics import mean_squared_error

conn = sqlite3.connect('Covid19.db')

def PaperDatasetSiettosBuilder():
    Raw_d = pd.read_excel('ItalyData_d.xlsx').set_index('Dates', drop=True)
    Raw_r = pd.read_excel('ItalyData_r.xlsx').set_index('Dates', drop=True)
    Raw_y = pd.read_excel('ItalyData_y.xlsx').set_index('Dates', drop=True)

    for country in Raw_d.columns:
        print(country)
        df = pd.concat([Raw_d[country], Raw_r[country], Raw_y[country]], axis=1)
        df.columns = ['d', 'r', 'y']
        df.to_sql(country, conn, if_exists='replace')

def ScrapWebData(modeIn):

    if modeIn == 'InitialSetup':
        profile = webdriver.FirefoxProfile()
        driver = webdriver.Firefox(firefox_profile=profile)

        link = "https://public.flourish.studio/story/217890/embed#slide-14"

        driver.get(link)
        html = driver.execute_script("return document.body.innerHTML;")
        pickle.dump(html, open("htmlRawData.p", "wb"))

    elif modeIn == 'Run':
        """
        ["Regione","Contagi","Deceduti","Guariti","Ricoverati con sintomi","In terapia intensiva","In isolamento domiciliare","Totale attualmente positivi",
        "Totale Tamponi","Variazione deceduti in assoluto","Variazione guariti in assoluto","Variazione positivi in assoluto","Contagi giornalieri","Tamponi giornalieri",
        "Rapporto tamponi/nuovi contagi giornalieri","Spiegazione"],
        "value":["Deceduti","Guariti","Totale attualmente positivi"]
        """
        from bs4 import BeautifulSoup
        html = pickle.load(open("htmlRawData.p", "rb")).strip()
        soup = BeautifulSoup(html)
        for item in soup.findAll("script"):
            subSoup = BeautifulSoup(item.text.strip())
        rawDataList = []
        dataList = []
        prevLine = 'region'
        for l in str(subSoup).split('{"facet":'):
            if ('Flourish' in l)|('{"data_table_id"' in l):
                pass
            else:
                if 'Regione' not in l:
                    prevLine = l.split(',')[0].replace('"', '')
                else:
                    l = l.replace("Regione",prevLine,1)

                rawDataList.append(l)
                dataList.append(l.split(',"'))

        rawDF = pd.DataFrame(rawDataList)
        rawDF.to_sql('dfRawScrapData', conn, if_exists='replace')
        df = pd.DataFrame(dataList)
        df.to_sql('dfRawScrapDataSplitted', conn, if_exists='replace')
        for col in df.columns:
            df[col] = df[col].str.replace('"', '').str.replace('label', '').str.replace('metadata', '').str.replace('value', '')\
                .str.replace(':', '').str.replace('[', '').str.replace(']', '').str.replace('{', '').str.replace('}', '')\
                .str.replace(' ', '').str.replace(',', '').str.replace(' ', '').str.replace('n.d.', '')
        df.to_sql('dfScrapData', conn, if_exists='replace')

    elif modeIn == 'FinalRegionSetup':
        df = pd.read_sql('SELECT * FROM dfScrapData', conn).set_index('index', drop=True)

        for region in list(df.iloc[:,0].unique()):
            regionData = df[df.iloc[:, 0] == region].dropna(axis=1, how='all')
            headers = []
            for x in regionData[regionData.iloc[:, 1] == 'Data'].values[0]:
                if 'Totaleattualmentepositivi' in x:
                    headers.append('Totaleattualmentepositivi')
                else:
                    headers.append(x)
            regionData.columns = headers
            regionData.columns = pd.io.parsers.ParserBase({'names': regionData.columns})._maybe_dedup_names(regionData.columns)
            regionData = regionData[regionData.iloc[:,1] != 'Data']
            regionData['Data'] = pd.to_datetime(regionData['Data'])
            regionData.to_sql(region, conn, if_exists='replace')
            #regionData.to_csv(region+'.csv', index=False)

def takensEmbedding (data, delay, dimension):
    "This function returns the Takens embedding of data with delay into dimension, delay*dimension must be < len(data)"
    if delay*dimension > len(data):
        raise NameError('Delay times dimension exceed length of data!')
    embeddedData = np.array([data[0:len(data)-delay*dimension]])
    for i in range(1, dimension):
        embeddedData = np.append(embeddedData, [data[i*delay:len(data) - delay*(dimension - i)]], axis=0)
    return embeddedData;

def Analyze(country):
    df0 = pd.read_sql('SELECT * FROM '+country, conn).set_index('Data', drop=True)
    df = df0[['Contagi', 'Deceduti','Guariti']].astype(float)
    It = df['Contagi'] - df['Deceduti'] - df['Guariti']
    print(It)
    It.plot()
    plt.show()

def RunRollManifoldOnCovid19(country, manifoldIn):
    df0 = pd.read_sql('SELECT * FROM '+country, conn).set_index('Data', drop=True)
    df = df0[['Contagi', 'Deceduti', 'Guariti']].astype(float)
    #df = sl.d(df).fillna(0)

    if manifoldIn == 'PCA':
        out = sl.AI.gRollingManifold(manifoldIn, df, 7, 2, [0,1])
    elif manifoldIn == 'LLE':
        out = sl.AI.gRollingManifold(manifoldIn, df, 7, 2, [0,1], LLE_n_neighbors=2, ProjectionMode='Transpose')

    out[0].to_sql(country+'_'+manifoldIn+'_df', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql(country+'_'+manifoldIn+'_principalCompsDf_'+str(k), conn, if_exists='replace')
        (principalCompsDfList[k]*df).to_sql(country+'_'+manifoldIn+'_exAnteProjections_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql(country+'_'+manifoldIn+'_exPostProjections_'+str(k), conn, if_exists='replace')

    out[3].to_sql(country+'_'+manifoldIn+'_Lambdas', conn, if_exists='replace')
    out[4].to_sql(country+'_'+manifoldIn+'_Sigmas', conn, if_exists='replace')

def ProjectionsPlots(country, manifoldIn):
    #PrincipalComps = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_principalCompsDf_0', conn).set_index('Data', drop=True)
    #ExPostProjections = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_exPostProjections_0', conn).set_index('Data', drop=True)
    #rsExPostProjections = sl.rs(pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_exAnteProjections_0', conn).set_index('Data', drop=True))
    #Lambdas = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_Lambdas', conn).set_index('Data', drop=True)
    #Lambdas.columns = ['First Projection', 'Second Projection']
    Sigmas = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_Sigmas', conn).set_index('Data', drop=True)
    Sigmas.columns = ['First Projection', 'Second Projection']

    #PrincipalComps.plot(title=country+' - Rolling PCA Embeddings Weights')
    #ExPostProjections.plot(title=country+' - Rolling PCA Projections Contributions')
    #rsExPostProjections.plot(title=country+' - Rolling PCA Projections')
    #Lambdas.plot(title=country+' - Rolling PCA Eigenvalues - Lambdas')
    Sigmas.plot(title=country+' - Rolling PCA Explained Variance Ratios')
    plt.show()

def RNNpredict(country, manifoldIn, magicNum):

    df = pd.read_sql('SELECT * FROM ' + country + '_' + manifoldIn + '_df', conn).set_index('Data', drop=True).fillna(0)

    params = {
        "HistLag": 0,
        "TrainWindow": 5,
        "epochsIn": 100,
        "batchSIzeIn": 1,
        "LearningMode": 'static',
        "LSTMmedSpecs": [{"units": 1, "RsF": False, "Dropout": 0.01}],
        "modelNum": magicNum,
        "TrainEndPct": 0.6,
        "CompilerSettings" : ['adam', 'mean_squared_error'], # optimizer='rmsprop', loss='categorical_crossentropy'
        "writeLearnStructure":1
    }

    out = sl.AI.gRNN(df, params)
    out[0].to_sql('df_real_price_RNN_' + country + manifoldIn + '_Projections_' + str(magicNum) , conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_' + country + manifoldIn + '_Projections_' + str(magicNum), conn, if_exists='replace')
    out[2].to_sql('scoreList_RNN_' + country + manifoldIn + '_Projections_' + str(magicNum), conn, if_exists='replace')

    pickle.dump(out[3], open("Covid19_RNN_Classifier.p", "wb"))

def RNNpredictionsAnalysis(country, manifoldIn, magicNum):

    df_real_price = pd.read_sql(
        'SELECT * FROM df_real_price_RNN_' + country + manifoldIn + '_Projections_' + str(magicNum), conn).set_index('Data', drop=True)

    df_predicted_price = pd.read_sql(
        'SELECT * FROM df_predicted_price_RNN_' + country + manifoldIn + '_Projections_'+ str(magicNum), conn).set_index('Data', drop=True)

    try:
        df_predicted_price.columns = df_real_price.columns
    except Exception as e:
        df_predicted_price.name = df_real_price.name

    for col in df_real_price.columns:
        allData = pd.concat([df_real_price[col], df_predicted_price[col]], axis=1)
        allData.columns = [col, col+'_Predicted']
        allData.plot()

    plt.show()

def LoadRNNclassifier():
    modelNum = '1'
    clFier = pickle.load(open("Covid19_RNN_Classifier"+modelNum+".p", "rb"))

#PaperDatasetBuilder()

#ScrapWebData('InitialSetup')
#ScrapWebData('Run')
#ScrapWebData('FinalRegionSetup')

#Analyze('Lombardia')
#RunRollManifoldOnCovid19('Lombardia', 'PCA')
#RunRollManifoldOnCovid19('Lombardia', 'LLE')
#ProjectionsPlots('Lombardia', 'PCA')

#RNNpredict('Lombardia', 'PCA', 1)
RNNpredictionsAnalysis('Lombardia', 'PCA', 1)

"""
def globalDatasetBuilder():
    df = pd.read_excel('COVID-19-geographic-disbtribution-worldwide-2020-03-22.xlsx').set_index('DateRep', drop=True)
    df.to_sql('Covid19RawData', conn, if_exists='replace')
    print(df.columns)
    dataAllCases = []
    dataAllDeaths = []
    uniqueCountriesList = df['Countries and territories'].unique().tolist()
    print(uniqueCountriesList)
    for country in uniqueCountriesList:
        indCases = df[df['Countries and territories']==country]['Cases']
        indCases.name = country
        indDeaths = df[df['Countries and territories']==country]['Deaths']
        indDeaths.name = country
        dataAllCases.append(indCases)
        dataAllDeaths.append(indDeaths)
    dataCasesDF = pd.concat(dataAllCases, axis=1)
    dataCasesDF = dataCasesDF.rename(columns={'CANADA': 'Canada2'})
    dataDeathsDF = pd.concat(dataAllDeaths, axis=1)
    dataDeathsDF = dataDeathsDF.rename(columns={'CANADA': 'Canada2'})
    print(dataDeathsDF)
    dataCasesDF.fillna(0).to_sql('CountryCases', conn, if_exists='replace')
    dataDeathsDF.fillna(0).to_sql('CountryDeaths', conn, if_exists='replace')

    fig, axes = plt.subplots(nrows=2, ncols=1)
    dataCasesDF.plot(ax=axes[0],title='Cases per Country')
    dataDeathsDF.plot(ax=axes[1], title='Deaths per Country')
    plt.show()
"""