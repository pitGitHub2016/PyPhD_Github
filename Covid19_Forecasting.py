from Slider import Slider as sl
import numpy as np, pickle
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

def DatasetBuilder():
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

def Analyze():
    dataCasesDF = pd.read_sql('SELECT * FROM CountryCases', conn).set_index('DateRep', drop=True)
    #print(df)
    dataCasesDF.plot(title='Cases per Country')
    #sl.cs(dataCasesDF).plot()
    plt.show()

HistLag = 0
TrainWindow = 7
epochsIn = 200
batchSIzeIn = 1
medBatchTrain = 1
HistoryPlot = 1
PredictionsPlot = [1, 'cs']  # PredictionsPlot = [1, 'NoCs']
LearningMode = 'static'
#LearningMode = 'online'
modelNum = '2'
TrainEndPct = 0.7

def RNNpredict(whatToRun, exPostMode):

    df = pd.read_csv('testDataset.csv').set_index('Month', drop=True)
    #df = pd.read_sql('SELECT * FROM '+whatToRun, conn).set_index('DateRep', drop=True)[['China', 'Italy', 'France']]
    #df = sl.cs(df)

    out = sl.AI.gRNN(df, [HistLag, TrainWindow, epochsIn, batchSIzeIn, medBatchTrain, HistoryPlot, PredictionsPlot,
                          LearningMode, modelNum, TrainEndPct])
    out[0].to_sql(
        'df_real_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(
            batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')
    out[1].to_sql('df_predicted_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
        TrainWindow) + '_' + str(
        epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
        PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')
    out[2].to_sql(
        'scoreList_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(TrainWindow) + '_' + str(
            epochsIn) + '_' + str(
            batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn, if_exists='replace')

    pickle.dump(out[3], open("Covid19_RNN_Classifier"+modelNum+".p", "wb"))

    historyOut = out[4]
    print(historyOut.history.keys())
    plt.plot(historyOut.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def RNNpredictionsAnalysis(whatToRun, exPostMode):

    df_real_price = pd.read_sql(
        'SELECT * FROM df_real_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn).set_index('DateRep', drop=True)

    df_predicted_price = pd.read_sql(
        'SELECT * FROM df_predicted_price_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn).set_index('DateRep', drop=True)

    df_predicted_price_errors = pd.read_sql(
        'SELECT * FROM scoreList_RNN_' + whatToRun + exPostMode + '_Projections_' + str(HistLag) + '_' + str(
            TrainWindow) + '_' + str(
            epochsIn) + '_' + str(batchSIzeIn) + '_' + str(medBatchTrain) + '_' + str(HistoryPlot) + '_' + str(
            PredictionsPlot[0]) + '_' + LearningMode + '_' + modelNum, conn).set_index('index', drop=True)

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
    print(clFier)

#DatasetBuilder()
#Analyze()
RNNpredict('CountryCases', '')
#RNNpredictionsAnalysis('CountryCases', '')
