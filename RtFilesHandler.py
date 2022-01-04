import glob, os, time
import pandas as pd

RtFolder = 'D:\Dropbox\EpidemicModel\WorkingDataset\ItalyInteractiveMap_Active\it-js-map\RtFolder\\'
os.chdir(RtFolder)

for fileIn in glob.glob(RtFolder + "*.xlsx"):
    df = pd.read_excel(fileIn, header=None)
    df.columns = ['dates', 'cases', 'deaths']
    df['deaths'] = df['deaths'].cumsum()
    df['dates'] = pd.to_datetime(df['dates'])
    df['day'] = df['dates'].dt.day
    df['month'] = df['dates'].dt.month
    dfToTXT = df[['day', 'month', 'cases', 'deaths']]
    dfToTXT.to_csv(fileIn.replace('xlsx', 'txt'), sep=' ', index=False, header=None)

