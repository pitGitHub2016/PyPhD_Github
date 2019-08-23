import pandas as pd, sqlite3

df0 = pd.read_csv('RollingPCA/fxEODdata.csv', header=None)

st = 50
for i in range(st, len(df0) + 1):
    df = df0.iloc[i - st:i, :]
    print(df)
    break

print(df0.iloc[st-1:])