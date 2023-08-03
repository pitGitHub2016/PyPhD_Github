from PyEurobankBloomberg.PySystems.PyLiveTradingSystems.Skeleton import Skeleton
#from PyEurobankBloomberg.PySystems.PyLiveTradingSystems.DataDeck import DataDeck
import warnings, pandas as pd, sqlite3
warnings.filterwarnings("ignore")

class Valley:

    def __init__(self):
        self.Assets = pd.read_excel("AssetsDashboard.xlsx", sheet_name="ActiveStrategies",engine='openpyxl')[__class__.__name__].dropna().tolist()
        self.strategylev = pd.read_excel("AssetsDashboard.xlsx", sheet_name="Live Strategies Control Panel",engine='openpyxl').set_index("Strategy Name", drop=True)

    def Run(self):

        Robot = Skeleton(__class__.__name__+".db", __class__.__name__, "PRODUCTION",__class__.__name__, self.Assets, 1000000, "EUR", self.strategylev.loc[[__class__.__name__],"Leverage"].values[0])
        Robot.StrategyRun(2, ShowPnL='No')
        #Robot.PositionsMonitor()
        #Robot.OrderHandler()

#DataDeck("DataDeck.db").Run()
Valley().Run()
