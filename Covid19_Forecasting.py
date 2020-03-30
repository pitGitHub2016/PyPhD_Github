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
from sklearn.metrics import mean_squared_error

conn = sqlite3.connect('Covid19.db')

def PaperDatasetBuilder():
    Raw_d = pd.read_excel('ItalyData_d.xlsx').set_index('Dates', drop=True)
    Raw_r = pd.read_excel('ItalyData_r.xlsx').set_index('Dates', drop=True)
    Raw_y = pd.read_excel('ItalyData_y.xlsx').set_index('Dates', drop=True)

    for country in Raw_d.columns:
        print(country)
        df = pd.concat([Raw_d[country], Raw_r[country], Raw_y[country]], axis=1)
        df.columns = ['d', 'r', 'y']
        df.to_sql(country, conn, if_exists='replace')

def Analyze(country):
    df = pd.read_sql('SELECT * FROM '+country, conn).set_index('Dates', drop=True)
    df = sl.d(df)
    df.plot(title='Lombardia - dry')
    plt.show()

def RunRollManifoldOnCovid19(country, manifoldIn):
    df = pd.read_sql('SELECT * FROM '+country, conn).set_index('Dates', drop=True)
    df = sl.d(df).fillna(0)

    if manifoldIn == 'PCA':
        out = sl.AI.gRollingManifold(manifoldIn, df, 7, 2, [0,1])
    elif manifoldIn == 'LLE':
        out = sl.AI.gRollingManifold(manifoldIn, df, 7, 2, [0,1], LLE_n_neighbors=1, ProjectionMode='Transpose')

    out[0].to_sql(country+'_'+manifoldIn+'_df', conn, if_exists='replace')
    principalCompsDfList = out[1]; exPostProjectionsList = out[2]
    for k in range(len(principalCompsDfList)):
        principalCompsDfList[k].to_sql(country+'_'+manifoldIn+'_principalCompsDf_'+str(k), conn, if_exists='replace')
        (principalCompsDfList[k]*df).to_sql(country+'_'+manifoldIn+'_exAnteProjections_'+str(k), conn, if_exists='replace')
        exPostProjectionsList[k].to_sql(country+'_'+manifoldIn+'_exPostProjections_'+str(k), conn, if_exists='replace')

    out[3].to_sql(country+'_'+manifoldIn+'_Lambdas', conn, if_exists='replace')
    out[4].to_sql(country+'_'+manifoldIn+'_Sigmas', conn, if_exists='replace')

def ProjectionsPlots(country, manifoldIn):
    #PrincipalComps = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_principalCompsDf_0', conn).set_index('Dates', drop=True)
    #ExPostProjections = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_exPostProjections_0', conn).set_index('Dates', drop=True)
    #rsExPostProjections = sl.rs(pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_exAnteProjections_0', conn).set_index('Dates', drop=True))
    #Lambdas = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_Lambdas', conn).set_index('Dates', drop=True)
    Sigmas = pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_Sigmas', conn).set_index('Dates', drop=True)

    #PrincipalComps.plot(title=country+' - Rolling PCA Embeddings Weights')
    #rsExPostProjections.plot(title=country+' - Rolling PCA Projections')
    #ExPostProjections.plot(title=country+' - Rolling PCA Projections Contributions')
    #Lambdas.plot(title=country+' - Rolling PCA Eigenvalues - Lambdas')
    Sigmas.plot(title=country+' - Rolling PCA Explained Variance Ratios')
    plt.show()

def RNNpredict(country, manifoldIn):

    df = sl.rs(pd.read_sql('SELECT * FROM '+country+'_'+manifoldIn+'_exAnteProjections_0', conn).set_index('Dates', drop=True)).fillna(0)

    params = {
        "HistLag": 0,
        "TrainWindow": 7,
        "epochsIn": 100,
        "batchSIzeIn": 1,
        "LearningMode": 'static',
        "LSTMmedSpecs": [{"units": 'xShape1', "RsF": False, "Dropout": 0.1}],
        "modelNum": 1,
        "TrainEndPct": 0.6,
        "CompilerSettings" : ['adam', 'mean_squared_error'], # optimizer='rmsprop', loss='categorical_crossentropy'
        "writeLearnStructure":1
    }

    out = sl.AI.gRNN(df, params)
    out[0].to_sql('df_real_price_RNN_' + country + manifoldIn + '_Projections_' + json.dumps(params) , conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_' + country + manifoldIn + '_Projections_' + json.dumps(params), conn, if_exists='replace')
    out[2].to_sql('scoreList_RNN_' + country + manifoldIn + '_Projections_' + json.dumps(params), conn, if_exists='replace')

    pickle.dump(out[3], open("Covid19_RNN_Classifier.p", "wb"))

    #historyOut = out[4]
    #print(historyOut.history.keys())
    #plt.plot(historyOut.history['loss'])
    #plt.title('model loss')
    #plt.ylabel('loss')
    #plt.xlabel('epoch')
    #plt.legend(['train', 'test'], loc='upper left')
    #plt.show()

def RNNpredictionsAnalysis(whatToRun, exPostMode):

    df_real_price = pd.read_sql(
        'SELECT * FROM df_real_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + LearningMode + '_' + str(modelNum), conn).set_index('DateRep', drop=True)

    df_predicted_price = pd.read_sql(
        'SELECT * FROM df_predicted_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + LearningMode + '_' + str(modelNum), conn).set_index('DateRep', drop=True)

    df_predicted_price_errors = pd.read_sql(
        'SELECT * FROM scoreList_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + LearningMode + '_' + str(modelNum), conn).set_index('index', drop=True)

    #df_predicted_price_errors.plot(title='Loss Function Values on Test Set')
    #plt.show()

    df_predicted_price.columns = df_real_price.columns

    MSE = mean_squared_error(df_real_price, df_predicted_price)
    testScore = math.sqrt(MSE)
    print('Test Score: %.2f MSE' % (MSE))
    print('Test Score: %.2f RMSE' % (testScore))

    fig, axes = plt.subplots(nrows=2, ncols=1)
    df_real_price['Italy'].plot(ax=axes[0], title='df_real_price')
    df_predicted_price['Italy'].plot(ax=axes[1], title='df_predicted_price')
    plt.show()

def LoadRNNclassifier():
    modelNum = '1'
    clFier = pickle.load(open("Covid19_RNN_Classifier"+modelNum+".p", "rb"))

#PaperDatasetBuilder()
#Analyze('Lombardia')
#RunRollManifoldOnCovid19('Lombardia', 'PCA')
#RunRollManifoldOnCovid19('Lombardia', 'LLE')
#ProjectionsPlots('Lombardia', 'PCA')
RNNpredict('Lombardia', 'PCA')

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