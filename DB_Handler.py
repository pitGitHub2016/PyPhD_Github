import time

import pandas as pd, sqlite3

def to_other_DB(DB, kws_str):
    db = sqlite3.connect(DB+'.db')
    to_db = sqlite3.connect(DB+'_'+kws_str+'.db')
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table_name in tables:
        table_name = table_name[0]
        if kws_str in table_name:
            try:
                print(table_name)
                table = pd.read_sql_query("SELECT * from %s" % table_name, db)
                table = table.set_index(table.columns[0], drop=True)
                table.to_sql(table_name, to_db, if_exists='replace')
            except Exception as e:
                print(e)
        #table.to_csv(table_name + '.csv', index_label='index')
    cursor.close()
    db.close()

for kw in ["allProjectionsDF", "FxData", "IRD", "principalCompsDf", "sema","ARIMA"]:#
    to_other_DB('FXeodData', kw)
