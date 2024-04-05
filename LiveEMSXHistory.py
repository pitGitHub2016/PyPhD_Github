import blpapi, sys, datetime, pandas as pd, sqlite3, numpy as np, time, glob
from optparse import OptionParser
from pyerb import pyerb as pe
import pdblp
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

SESSION_STARTED = blpapi.Name("SessionStarted")
SESSION_STARTUP_FAILURE = blpapi.Name("SessionStartupFailure")
SERVICE_OPENED = blpapi.Name("ServiceOpened")
SERVICE_OPEN_FAILURE = blpapi.Name("ServiceOpenFailure")
ERROR_INFO = blpapi.Name("ErrorInfo")
GET_FILLS_RESPONSE = blpapi.Name("GetFillsResponse")

d_service = "//blp/emsx.history"
d_host = "localhost"
d_port = 8194
bEnd = False

class LiveEMSXHistory:

    def __init__(self, DB):
        self.DB = DB
        self.conn = sqlite3.connect(self.DB)
        self.ActiveAssetsReferenceDataExcel = pd.read_sql('SELECT * FROM ActiveAssetsReferenceData', sqlite3.connect("DataDeck.db")).set_index("ticker", drop=True)
        self.FUT_NOTICE_FIRST = self.ActiveAssetsReferenceDataExcel["FUT_NOTICE_FIRST"]
        self.CURR_GENERIC_FUTURES_SHORT_NAME = self.ActiveAssetsReferenceDataExcel["CURR_GENERIC_FUTURES_SHORT_NAME"]
        self.EMSXFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/PyLiveTradingSystems/EMSX/"
        self.Bloomberg_EMSX = "F:/Dealing/Panagiotis Papaioannou/pyerb/PyEurobankBloomberg/PySystems/Bloomberg_EMSX/"
        self.GreenBoxFolder = "F:/Dealing/Panagiotis Papaioannou/pyerb/GreenBox/"

        ### CHECK IF EMSX MAIN EMSX HISTORY TABLE EXISTS ###
        c = self.conn.cursor()
        # get the count of tables with the name
        c.execute(''' SELECT count(name) FROM sqlite_master WHERE type='table' AND name='MAIN_EMSX_HISTORY' ''')
        # if the count is 1, then table exists
        if c.fetchone()[0] == 1:
            print('Table exists.')
            self.MAIN_EMSX_HISTORY = pd.read_sql('SELECT * FROM MAIN_EMSX_HISTORY', self.conn).set_index("index", drop=True)
        else:
            self.MAIN_EMSX_HISTORY = pd.DataFrame()
        # commit the changes to db
        self.conn.commit()

    def processEMSX_Blotter(self):
        dfList = []
        for name in glob.glob(self.EMSXFolder+'*.csv'):
            print(name)
            sub_df = pd.read_csv(name)
            dfList.append(sub_df)
        df = pd.concat(dfList)
        df["BinarySide"] = None
        df.loc[df["Side"] == "Buy", "BinarySide"] = 1
        df.loc[df["Side"] == "Sell", "BinarySide"] = -1
        df["Position"] = df["Qty"] * df["BinarySide"]
        df = df.sort_values(by="Create Time (As of)")
        df.to_sql("EMSX_Blotter_Raw", self.conn, if_exists='replace')

        df = df[df["Status"] == "Filled"].dropna(subset=["P/L (T) Currency vs. Lst Trd/Lst Px EUR"])
        df["Sec ID w/ YK Parsky"] = df["Sec ID w/ YK Parsky"].str.replace(" COMB ", " ")
        df = df.fillna(0)
        df.to_sql("EMSX_Blotter_Filled", self.conn, if_exists='replace')
        df_Groupped = df.groupby(["Sec ID w/ YK Parsky"]).sum().replace([np.inf, -np.inf], 0)
        df_Groupped.to_sql("EMSX_Blotter_Groupped", self.conn, if_exists='replace')

        print(df)

    def main(self):

        dataOut = []

        class SessionEventHandler():

            def processEvent(self, event, session):
                try:
                    if event.eventType() == blpapi.Event.SESSION_STATUS:
                        self.processSessionStatusEvent(event, session)

                    elif event.eventType() == blpapi.Event.SERVICE_STATUS:
                        self.processServiceStatusEvent(event, session)

                    elif event.eventType() == blpapi.Event.RESPONSE or event.eventType() == blpapi.Event.PARTIAL_RESPONSE:
                        self.processResponseEvent(event)

                    else:
                        self.processMiscEvents(event)

                except:
                    print("Exception:  %s" % sys.exc_info()[0])

                return False

            def processSessionStatusEvent(self, event, session):
                print("Processing SESSION_STATUS event")

                for msg in event:
                    if msg.messageType() == SESSION_STARTED:
                        print("Session started...")
                        session.openServiceAsync(d_service)

                    elif msg.messageType() == SESSION_STARTUP_FAILURE:
                        print(sys.stderr, ("Error: Session startup failed"))

                    else:
                        print(msg)

            def processServiceStatusEvent(self, event, session):
                print("Processing SERVICE_STATUS event")

                for msg in event:

                    if msg.messageType() == SERVICE_OPENED:
                        print("Service opened...")

                        service = session.getService(d_service)

                        request = service.createRequest("GetFills")
                        today = datetime.datetime.now()
                        if today.month <= 6:
                            requestMonth = 1
                        else:
                            requestMonth = 6
                        request.set("FromDateTime", str(today.year)+"-0"+str(requestMonth)+"-01T00:00:00.000+00:00")
                        request.set("ToDateTime", (today+datetime.timedelta(days=1)).isoformat())

                        scope = request.getElement("Scope")
                        scope.setElement("Team", "DERV_TRADING")
                        print("Request: %s" % request.toString())

                        self.requestID = blpapi.CorrelationId()

                        session.sendRequest(request, correlationId=self.requestID)

                    elif msg.messageType() == SERVICE_OPEN_FAILURE:
                        print(sys.stderr, ("Error: Service failed to open"))

            def processResponseEvent(self, event):
                print("Processing RESPONSE event")

                for msg in event:

                    if msg.correlationIds()[0].value() == self.requestID.value():
                        print("MESSAGE TYPE: %s" % msg.messageType())

                        if msg.messageType() == ERROR_INFO:
                            errorCode = msg.getElementAsInteger("ErrorCode")
                            errorMessage = msg.getElementAsString("ErrorMsg")
                            print("ERROR CODE: %d\tERROR MESSAGE: %s" % (errorCode, errorMessage))
                        elif msg.messageType() == GET_FILLS_RESPONSE:

                            fills = msg.getElement("Fills")

                            for fill in fills.values():
                                #print(fill)
                                try:
                                    try:
                                        account = fill.getElement("Account").getValueAsString()
                                    except:
                                        account = None

                                    try:
                                        amount = fill.getElement("Amount").getValueAsFloat()
                                    except:
                                        amount = None

                                    try:
                                        assetClass = fill.getElement("AssetClass").getValueAsString()
                                    except:
                                        assetClass = None

                                    try:
                                        basketId = fill.getElement("BasketId").getValueAsInteger()
                                    except:
                                        basketId = None

                                    try:
                                        bbgid = fill.getElement("BBGID").getValueAsString()
                                    except:
                                        bbgid = None

                                    try:
                                        blockId = fill.getElement("BlockId").getValueAsString()
                                    except:
                                        blockId = None

                                    try:
                                        broker = fill.getElement("Broker").getValueAsString()
                                    except:
                                        broker = None

                                    try:
                                        clearingAccount = fill.getElement("ClearingAccount").getValueAsString()
                                    except:
                                        clearingAccount = None

                                    try:
                                        clearingFirm = fill.getElement("ClearingFirm").getValueAsString()
                                    except:
                                        clearingFirm = None

                                    try:
                                        contractExpDate = fill.getElement("ContractExpDate").getValueAsString()
                                    except:
                                        contractExpDate = None

                                    try:
                                        correctedFillId = fill.getElement("CorrectedFillId").getValueAsInteger()
                                    except:
                                        correctedFillId = None

                                    try:
                                        currency = fill.getElement("Currency").getValueAsString()
                                    except:
                                        currency = None

                                    try:
                                        cusip = fill.getElement("Cusip").getValueAsString()
                                    except:
                                        cusip = None

                                    try:
                                        dateTimeOfFill = fill.getElement("DateTimeOfFill").getValueAsString()
                                    except:
                                        dateTimeOfFill = None

                                    try:
                                        exchange = fill.getElement("Exchange").getValueAsString()
                                    except:
                                        exchange = None

                                    try:
                                        execPrevSeqNo = fill.getElement("ExecPrevSeqNo").getValueAsInteger()
                                    except:
                                        execPrevSeqNo = None

                                    try:
                                        execType = fill.getElement("ExecType").getValueAsString()
                                    except:
                                        execType = None

                                    try:
                                        executingBroker = fill.getElement("ExecutingBroker").getValueAsString()
                                    except:
                                        executingBroker = None

                                    try:
                                        fillId = fill.getElement("FillId").getValueAsInteger()
                                    except:
                                        fillId = None

                                    try:
                                        fillPrice = fill.getElement("FillPrice").getValueAsFloat()
                                    except:
                                        fillPrice = None

                                    try:
                                        fillShares = fill.getElement("FillShares").getValueAsFloat()
                                    except:
                                        fillShares = None

                                    try:
                                        investorId = fill.getElement("InvestorID").getValueAsString()
                                    except:
                                        investorId = None

                                    try:
                                        isCFD = fill.getElement("IsCfd").getValueAsBool()
                                    except:
                                        isCFD = None

                                    try:
                                        isin = fill.getElement("Isin").getValueAsString()
                                    except:
                                        isin = None

                                    try:
                                        isLeg = fill.getElement("IsLeg").getValueAsBool()
                                    except:
                                        isLeg = None

                                    try:
                                        lastCapacity = fill.getElement("LastCapacity").getValueAsString()
                                    except:
                                        lastCapacity = None

                                    try:
                                        lastMarket = fill.getElement("LastMarket").getValueAsString()
                                    except:
                                        lastMarket = None

                                    try:
                                        limitPrice = fill.getElement("LimitPrice").getValueAsFloat()
                                    except:
                                        limitPrice = None

                                    try:
                                        liquidity = fill.getElement("Liquidity").getValueAsString()
                                    except:
                                        liquidity = None

                                    try:
                                        localExchangeSymbol = fill.getElement("LocalExchangeSymbol").getValueAsString()
                                    except:
                                        localExchangeSymbol = None

                                    try:
                                        locateBroker = fill.getElement("LocateBroker").getValueAsString()
                                    except:
                                        locateBroker = None

                                    try:
                                        locateId = fill.getElement("LocateId").getValueAsString()
                                    except:
                                        locateId = None

                                    try:
                                        locateRequired = fill.getElement("LocateRequired").getValueAsBool()
                                    except:
                                        locateRequired = None

                                    try:
                                        multiLedId = fill.getElement("MultilegId").getValueAsString()
                                    except:
                                        multiLedId = None

                                    try:
                                        occSymbol = fill.getElement("OCCSymbol").getValueAsString()
                                    except:
                                        occSymbol = None

                                    try:
                                        orderExecutionInstruction = fill.getElement(
                                            "OrderExecutionInstruction").getValueAsString()
                                    except:
                                        orderExecutionInstruction = None

                                    try:
                                        orderHandlingInstruction = fill.getElement(
                                            "OrderHandlingInstruction").getValueAsString()
                                    except:
                                        orderHandlingInstruction = None

                                    try:
                                        orderId = fill.getElement("OrderId").getValueAsInteger()
                                    except:
                                        orderId = None

                                    try:
                                        orderInstruction = fill.getElement("OrderInstruction").getValueAsString()
                                    except:
                                        orderInstruction = None

                                    try:
                                        orderOrigin = fill.getElement("OrderOrigin").getValueAsString()
                                    except:
                                        orderOrigin = None

                                    try:
                                        orderReferenceId = fill.getElement("OrderReferenceId").getValueAsString()
                                    except:
                                        orderReferenceId = None

                                    try:
                                        originatingTraderUUId = fill.getElement("OriginatingTraderUuid").getValueAsInteger()
                                    except:
                                        originatingTraderUUId = None

                                    try:
                                        reroutedBroker = fill.getElement("ReroutedBroker").getValueAsString()
                                    except:
                                        reroutedBroker = None

                                    try:
                                        routeCommissionAmount = fill.getElement("RouteCommissionAmount").getValueAsFloat()
                                    except:
                                        routeCommissionAmount = None

                                    try:
                                        routeCommissionRate = fill.getElement("RouteCommissionRate").getValueAsFloat()
                                    except:
                                        routeCommissionRate = None

                                    try:
                                        routeExecutionInstruction = fill.getElement(
                                            "RouteExecutionInstruction").getValueAsString()
                                    except:
                                        routeExecutionInstruction = None

                                    try:
                                        routeHandlingInstruction = fill.getElement(
                                            "RouteHandlingInstruction").getValueAsString()
                                    except:
                                        routeHandlingInstruction = None

                                    try:
                                        routeId = fill.getElement("RouteId").getValueAsInteger()
                                    except:
                                        routeId = None

                                    try:
                                        routeNetMoney = fill.getElement("RouteNetMoney").getValueAsFloat()
                                    except:
                                        routeNetMoney = None

                                    try:
                                        routeNotes = fill.getElement("RouteNotes").getValueAsString()
                                    except:
                                        routeNotes = None

                                    try:
                                        routeShares = fill.getElement("RouteShares").getValueAsFloat()
                                    except:
                                        routeShares = None

                                    try:
                                        securityName = fill.getElement("SecurityName").getValueAsString()
                                    except:
                                        securityName = None

                                    try:
                                        sedol = fill.getElement("Sedol").getValueAsString()
                                    except:
                                        sedol = None

                                    try:
                                        settlementDate = fill.getElement("SettlementDate").getValueAsString()
                                    except:
                                        settlementDate = None

                                    try:
                                        side = fill.getElement("Side").getValueAsString()
                                    except:
                                        side = None

                                    try:
                                        stopPrice = fill.getElement("StopPrice").getValueAsFloat()
                                    except:
                                        stopPrice = None

                                    try:
                                        strategyType = fill.getElement("StrategyType").getValueAsString()
                                    except:
                                        strategyType = None

                                    try:
                                        ticker = fill.getElement("Ticker").getValueAsString()
                                    except:
                                        ticker = None

                                    try:
                                        tif = fill.getElement("TIF").getValueAsString()
                                    except:
                                        tif = None

                                    try:
                                        traderName = fill.getElement("TraderName").getValueAsString()
                                    except:
                                        traderName = None

                                    try:
                                        traderUUId = fill.getElement("TraderUuid").getValueAsInteger()
                                    except:
                                        traderUUId = None

                                    try:
                                        type = fill.getElement("Type").getValueAsString()
                                    except:
                                        type = None

                                    try:
                                        userCommissionAmount = fill.getElement("UserCommissionAmount").getValueAsFloat()
                                    except:
                                        userCommissionAmount = None

                                    try:
                                        userCommissionRate = fill.getElement("UserCommissionRate").getValueAsFloat()
                                    except:
                                        userCommissionRate = None

                                    try:
                                        userFees = fill.getElement("UserFees").getValueAsFloat()
                                    except:
                                        userFees = None

                                    try:
                                        userNetMoney = fill.getElement("UserNetMoney").getValueAsFloat()
                                    except:
                                        userNetMoney = None

                                    try:
                                        yellowKey = fill.getElement("YellowKey").getValueAsString()
                                    except:
                                        yellowKey = None

                                    dataOut.append(
                                        [account, amount, assetClass, basketId, bbgid, blockId, broker, clearingAccount,
                                         clearingFirm, contractExpDate,
                                         correctedFillId, currency, cusip, dateTimeOfFill, exchange, execPrevSeqNo,
                                         execType, executingBroker, fillId, fillPrice, routeShares, fillShares, investorId, isCFD,
                                         isin, isLeg, lastCapacity, lastMarket, limitPrice, liquidity, localExchangeSymbol,
                                         locateBroker, locateId, locateRequired, multiLedId, occSymbol,
                                         orderExecutionInstruction, orderHandlingInstruction, orderId, orderInstruction,
                                         orderOrigin, orderReferenceId, originatingTraderUUId, reroutedBroker,
                                         routeCommissionAmount,
                                         routeCommissionRate, routeExecutionInstruction, routeHandlingInstruction, routeId,
                                         routeNetMoney, routeNotes, securityName, sedol, settlementDate,
                                         side, stopPrice, strategyType, ticker, tif, traderName, traderUUId, type,
                                         userCommissionAmount, userCommissionRate, userFees, userNetMoney, yellowKey])

                                except Exception as e:
                                    print(e)

                        global bEnd
                        bEnd = True

            def processMiscEvents(self, event):

                print("Processing " + event.eventType() + " event")

                for msg in event:
                    print("MESSAGE: %s" % (msg.tostring()))

        sessionOptions = blpapi.SessionOptions()
        sessionOptions.setServerHost(d_host)
        sessionOptions.setServerPort(d_port)

        print("Connecting to %s:%d" % (d_host, d_port))

        eventHandler = SessionEventHandler()

        session = blpapi.Session(sessionOptions, eventHandler.processEvent)

        if not session.startAsync():
            print("Failed to start session.")
            return

        global bEnd
        while bEnd == False:
            pass

        recentDF = pd.DataFrame(dataOut, columns=["account", "amount", "assetClass", "basketId", "bbgid", "blockId", "broker", "clearingAccount",
                                         "clearingFirm", "contractExpDate", "correctedFillId", "currency", "cusip",
                                        "dateTimeOfFill", "exchange", "execPrevSeqNo", "execType", "executingBroker",
                                            "fillId", "fillPrice", "routeShares", "fillShares", "investorId", "isCFD",
                                         "isin", "isLeg", "lastCapacity", "lastMarket", "limitPrice", "liquidity",
                                            "localExchangeSymbol", "locateBroker", "locateId", "locateRequired",
                                            "multiLedId", "occSymbol", "orderExecutionInstruction",
                                            "orderHandlingInstruction", "orderId", "orderInstruction",
                                         "orderOrigin", "orderReferenceId", "originatingTraderUUId", "reroutedBroker",
                                         "routeCommissionAmount", "routeCommissionRate", "routeExecutionInstruction",
                                            "routeHandlingInstruction", "routeId", "routeNetMoney", "routeNotes",
                                            "securityName", "sedol", "settlementDate", "side", "stopPrice", "strategyType", "ticker", "tif",
                                         "traderName", "traderUUId", "type", "userCommissionAmount", "userCommissionRate",
                                            "userFees", "userNetMoney", "yellowKey"])

        "Correction on Exposures / Get to the next period (Roll History)"
        #recentDF = recentDF[(recentDF["contractExpDate"].str.split("-").str[0].astype(float)==2022)&(recentDF["contractExpDate"].str.split("-").str[1].astype(float)!=9)]

        recentDF.to_sql("LiveEMSXHistory", self.conn, if_exists='replace')

        ### UPDATE / APPEND MAIN ESMX TABLE ###
        print("MAIN_EMSX_HISTORY = ", len(self.MAIN_EMSX_HISTORY))
        print("recentDF = ", len(recentDF))
        df = pd.concat([self.MAIN_EMSX_HISTORY, recentDF], axis=0).drop_duplicates()
        #df = recentDF.copy()
        print("df = ", len(df))
        df.to_sql("MAIN_EMSX_HISTORY", self.conn, if_exists='replace')

        session.stop()

    def ProcessFills(self):
        def my_agg(x):
            names = {'weighted_ave_price': (x['exposureShares'] * x['fillPrice']).sum() / x['exposureShares'].sum()}
            return pd.Series(names, index=['weighted_ave_price'])

        processedFills = pd.read_sql('SELECT * FROM MAIN_EMSX_HISTORY', self.conn).set_index("index", drop=True)

        processedFills["Asset"] = processedFills["ticker"] + " " + processedFills["yellowKey"]
        processedFills["BinarySide"] = 0
        processedFills["BinarySide"][processedFills["side"] == "B"] = 1
        processedFills["BinarySide"][processedFills["side"] == "S"] = -1
        processedFills["exposureShares"] = processedFills["BinarySide"] * processedFills["fillShares"]

        processedFills.to_sql("processedFills", self.conn, if_exists='replace')

        avgFillPrices = processedFills.groupby(["Asset"]).apply(my_agg).replace([np.inf, -np.inf], 0)

        """Aggregate"""
        for trader in processedFills["traderName"].unique():
            print(trader)
            if trader == "PPAPAIOANNO1":
                trader_processedFills = processedFills[processedFills["traderName"] == trader]
                trader_processedFills.to_sql(trader+"_processedFills", self.conn, if_exists='replace')
                ########################################################################################################################
                "External Strategies Exposures"
                ExternalStrategies = ['Galileo']
                for strategy in ExternalStrategies:
                    externalStratExposuresDF = trader_processedFills[trader_processedFills['orderInstruction']==strategy][['Asset','exposureShares']]
                    externalStratExposuresDF = externalStratExposuresDF.rename(columns={"exposureShares":strategy})
                    externalStratExposuresDF['BaseAsset'] = externalStratExposuresDF['Asset'].str[:2] + '1 ' + externalStratExposuresDF['Asset'].str.split(" ").str[1]
                    externalStratExposuresDF = externalStratExposuresDF.set_index('Asset',drop=True)
                    externalStratExposuresDF = externalStratExposuresDF.groupby('BaseAsset').sum()
                    externalStratExposuresDF.to_sql(trader+"_ExposuresDF_"+strategy, self.conn, if_exists='replace')
                ########################################################################################################################
                trader_processedFills_Excel = trader_processedFills.copy()
                trader_processedFills_Excel['dateTimeOfFill'] = pd.to_datetime((trader_processedFills_Excel['dateTimeOfFill'].astype(str).str.split(" ").str[0]).str.split("T").str[0])
                trader_processedFills_Excel.to_sql(trader+"_processedFills_Excel_ToKPlus", self.conn, if_exists='replace')

                trader_processedFills_Excel['TodayTimeDiff'] = (trader_processedFills_Excel['dateTimeOfFill']-datetime.datetime.now()).dt.days
                trader_processedFills_Excel_ToKPlus = trader_processedFills_Excel[trader_processedFills_Excel['TodayTimeDiff']>=-2]
                trader_processedFills_Excel_ToKPlus = trader_processedFills_Excel_ToKPlus[["Asset", "side", "TodayTimeDiff", "fillPrice", "fillShares"]]
                trader_processedFills_Excel_ToKPlus["K_Plus_Ticker"] = None
                trader_processedFills_Excel_ToKPlus["K_Plus_Maturity"] = None
                for idx, row in trader_processedFills_Excel_ToKPlus.iterrows():
                    trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Ticker"] = pe.EMSX_Kondor_Dict(row['Asset'][:2]+"1 "+row["Asset"].split(" ")[-1])
                    ########################################################################################################################
                    if row['Asset'][2]=="H":
                        trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Maturity"] = '01/03/'+str(datetime.datetime.now().year)
                    if row['Asset'][2]=="M":
                        trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Maturity"] = '01/06/'+str(datetime.datetime.now().year)
                    if row['Asset'][2]=="U":
                        trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Maturity"] = '01/09/'+str(datetime.datetime.now().year)
                    if row['Asset'][2]=="Z":
                        trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Maturity"] = '01/12/'+str(datetime.datetime.now().year)
                    ########################################################################################################################
                    if trader_processedFills_Excel_ToKPlus.loc[idx, "K_Plus_Ticker"] in ["DXY_FUTURE", "F_JPY", "F_NZD", "F_GBP", "F_CAD"]:
                        trader_processedFills_Excel_ToKPlus.loc[idx,"fillPrice"] /= 100
                ########################################################################################################################
                trader_processedFills_Excel_ToKPlus.to_excel(self.Bloomberg_EMSX+trader+"_processedFills_Excel_ToKPlus_Today.xlsx")
                ########################################################################################################################
                trader_aggregatedFills = trader_processedFills.copy()
                "HARDCODED EMSX EXCEPTIONS!!!"
                trader_aggregatedFills = trader_aggregatedFills[trader_aggregatedFills["bbgid"] != ""]
                trader_aggregatedFills =trader_aggregatedFills[~trader_aggregatedFills["bbgid"].isin(["BBG01FVC2626"])]
                "############################"
                trader_aggregatedFills = trader_aggregatedFills.groupby("Asset").agg({'exposureShares' : "sum", 'ticker' : "last"})
                trader_aggregatedFills['Generics'] = trader_aggregatedFills["ticker"].str[:-2]+"1 "+trader_aggregatedFills.index.to_series().str.split(" ").str[1]
                trader_aggregatedFills[['FirstNotice','FUTURES_NAME']] = None
                for idx, row in trader_aggregatedFills.iterrows():
                    if row['Generics'][:2] == "G ": #"Handle strange naming of Gilts"
                        genericToScan = "G 1 Comdty"
                    elif row['Generics'][:2] == "Z ": #"Handle strange naming of Gilts"
                        genericToScan = "Z 1 Index"
                    else:
                        genericToScan = row['Generics']
                    try:
                        trader_aggregatedFills.loc[idx, "FirstNotice"] = self.FUT_NOTICE_FIRST[genericToScan]
                        trader_aggregatedFills.loc[idx, "FUTURES_NAME"] = self.CURR_GENERIC_FUTURES_SHORT_NAME[genericToScan]
                    except Exception as e:
                        print(e)
                trader_aggregatedFills['FirstNotice'] = pd.to_datetime(trader_aggregatedFills['FirstNotice'].str.split('+').str[0])
                trader_aggregatedFills['DaysToExpire'] = (trader_aggregatedFills['FirstNotice'] - datetime.datetime.now()).dt.days
                trader_aggregatedFills['RollAction'] = ''
                trader_aggregatedFills.loc[trader_aggregatedFills['DaysToExpire'] < 0, 'RollAction'] = 'Expired'
                trader_aggregatedFills.loc[(trader_aggregatedFills['DaysToExpire'] > 0) & (trader_aggregatedFills['DaysToExpire'].abs() <= 2), 'RollAction'] = 'NEED TO ROLL !!!'
                ### Concatenate with Avg Prices ###
                trader_aggregatedFills = pd.concat([trader_aggregatedFills, avgFillPrices], axis=1)
                if trader == "PPAPAIOANNO1":
                    EMSX_Blotter_Groupped = pd.read_sql('SELECT * FROM EMSX_Blotter_Groupped', self.conn)
                    EMSX_Blotter_Groupped = EMSX_Blotter_Groupped.rename(columns={"Sec ID w/ YK Parsky":"index", "Position":"exportedEMSXBlotterPosition"}).set_index("index", drop=True)
                    trader_aggregatedFills = pd.concat([trader_aggregatedFills, EMSX_Blotter_Groupped["exportedEMSXBlotterPosition"]], axis=1)
                    trader_aggregatedFills["exportedEMSXBlotterPosition"] = trader_aggregatedFills["exportedEMSXBlotterPosition"].fillna(0)
                    trader_aggregatedFills['exposureShares'] += trader_aggregatedFills["exportedEMSXBlotterPosition"]
                    trader_aggregatedFills['exposureShares'] = trader_aggregatedFills['exposureShares'].fillna(0)

                trader_aggregatedFills.to_sql(trader+"_aggregatedFills", self.conn, if_exists='replace')
                trader_aggregatedFillsHtml = trader_aggregatedFills.reset_index()
                trader_aggregatedFillsHtml = trader_aggregatedFillsHtml[trader_aggregatedFillsHtml['exposureShares'] != 0]
                pe.RefreshableFile([[trader_aggregatedFillsHtml, 'trader_aggregatedFills']],
                                   self.GreenBoxFolder + trader + '_trader_aggregatedFills.html', 5,
                                   cssID='QuantitativeStrategies', addButtons="aggregatedFills")

        #filledAssets = processedFills['Asset'].tolist()
        #con = pdblp.BCon(debug=True, port=8194, timeout=20000).start()
        #con.ref(filledAssets, 'LAST_TRADEABLE_DT')

#obj = LiveEMSXHistory("LiveEMSXHistory.db")
#obj.main()
#obj.ProcessFills()


"""NOTES"""
#obj.processEMSX_Blotter()