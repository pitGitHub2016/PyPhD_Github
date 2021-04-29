from Slider import Slider as sl
import seaborn as sns
import numpy as np, investpy, time, pickle
import pandas as pd
from tqdm import tqdm
import pymc3 as pm
import arviz as az
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')

conn = sqlite3.connect('FXeodData.db')
GraphsFolder = '/home/gekko/Desktop/PyPhD/RollingManifoldLearning/Graphs/'

twList = [25, 100, 150, 250, 'ExpWindow25']
lagList = [2, 3, 5, 10, 15, 25, 50, 100, 150, 200, 250]
kernelList = ["RBF"] #"RBF","DotProduct","Matern","RationalQuadratic","WhiteKernel","Matern_WhiteKernel"

calcMode = 'run'
pnlCalculator = 0

def GPRlocal(argList):
    selection = argList[0]
    df = argList[1]
    trainLength = argList[2]
    kernelIn = argList[3]
    rw = argList[4]
    print(selection, ",", trainLength, ",", kernelIn, ", ", rw)
    try:
        if calcMode == 'run':
            GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)

            GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
            GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

            pickle.dump(GPR_Results[2], open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) +".p", "wb"))

        elif calcMode == 'read':

            GPR_Results = [pd.read_sql('SELECT * FROM ' + selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn).set_index('Dates', drop=True),
                           pd.read_sql('SELECT * FROM ' + selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn).set_index('Dates', drop=True)]

        if pnlCalculator == 0:
            sig = sl.sign(GPR_Results[1])
            pnl = sig * GPR_Results[0]
        elif pnlCalculator == 1:
            sig = sl.S(sl.sign(GPR_Results[1]), nperiods=-1)
            pnl = sig * GPR_Results[0]
        elif pnlCalculator == 2:
            sig = sl.sign(GPR_Results[1])
            pnl = sig * sl.S(GPR_Results[0], nperiods=-1)

        reportSh = np.sqrt(252) * sl.sharpe(pnl)
        print(reportSh)
        #time.sleep(30000)

        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

    except Exception as e:
        print(e)

def GPRonPortfolios(Portfolios, scanMode, mode):
    if Portfolios == 'Projections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)
    elif Portfolios == 'ClassicPortfolios':
        LOportfolio = pd.read_sql('SELECT * FROM LongOnlyEWPEDf', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["LO"]
        RPportfolio = pd.read_sql('SELECT * FROM RiskParityEWPrsDf_tw_250', conn).set_index('Dates', drop=True)
        LOportfolio.columns = ["RP"]
        allProjectionsDF = pd.concat([LOportfolio, RPportfolio], axis=1)
    elif Portfolios == 'globalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
             globalProjectionsList.append(pd.read_sql('SELECT * FROM globalProjectionsDF_'+manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)
    elif Portfolios == 'Specific':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[
            set(["PCA_250_0","PCA_250_1","PCA_250_2","PCA_250_19",
             "PCA_ExpWindow25_1","PCA_ExpWindow25_2",
             "PCA_150_19","PCA_100_19","PCA_ExpWindow25_18",
             "PCA_25_6","PCA_ExpWindow25_13","PCA_250_8", "PCA_150_18",
             "PCA_25_16","PCA_150_8", "PCA_ExpWindow25_4", "PCA_250_13", "PCA_ExpWindow25_11",
             "PCA_100_7", "PCA_150_5", "PCA_150_10",
             "LLE_250_0","LLE_250_1","LLE_250_2",
             "LLE_ExpWindow25_1", "LLE_ExpWindow25_2",
             "LLE_ExpWindow25_17", "LLE_25_11", "LLE_250_18","LLE_ExpWindow25_14",
             "LLE_150_1", "LLE_250_9", "LLE_25_15", "LLE_ExpWindow25_4",
             "LLE_ExpWindow25_11", "LLE_100_11", "LLE_150_3", "LLE_150_2", "LLE_100_4",
             "LLE_250_5", "LLE_250_11", "LLE_150_6", "LLE_25_7"])] #"PCA_ExpWindow25_0","PCA_ExpWindow25_19","LLE_ExpWindow25_0","LLE_ExpWindow25_18",
    elif Portfolios == 'FinalistsProjections':
        allProjectionsDF = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[['PCA_ExpWindow25_0', 'PCA_ExpWindow25_19', 'LLE_ExpWindow25_0', "LLE_ExpWindow25_18"]]
    elif Portfolios == 'FinalistsGlobalProjections':
        globalProjectionsList = []
        for manifoldIn in ["PCA", "LLE"]:
            globalProjectionsList.append(
                pd.read_sql('SELECT * FROM globalProjectionsDF_' + manifoldIn, conn).set_index('Dates', drop=True))
        allProjectionsDF = pd.concat(globalProjectionsList, axis=1)[["PCA_ExpWindow25_5_Head", "PCA_ExpWindow25_5_Tail", "LLE_ExpWindow25_5_Head", "LLE_ExpWindow25_5_Tail",
                                                                     "PCA_ExpWindow25_3_Head","PCA_ExpWindow25_3_Tail", "LLE_ExpWindow25_3_Head", "LLE_ExpWindow25_3_Tail"]]

    if scanMode == 'Main':

        if mode == "run":
            processList = []
            rw = 250
            for kernelIn in kernelList:
                for selection in allProjectionsDF.columns:
                    processList.append([selection, allProjectionsDF[selection], 0.3, kernelIn, rw])

            p = mp.Pool(mp.cpu_count())
            result = p.map(GPRlocal, tqdm(processList))
            p.close()
            p.join()

        elif mode == "report":
            notProcessed = []
            rw = 250
            shList = []
            for kernelIn in kernelList:
                for selection in allProjectionsDF.columns:
                    try:
                        pnl = pd.read_sql('SELECT * FROM ' + selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw),
                                          conn).set_index('Dates', drop=True).iloc[round(0.3*len(allProjectionsDF)):]
                        pnl.columns = [selection]
                        pnl['RW'] = sl.S(sl.sign(allProjectionsDF[selection])) * allProjectionsDF[selection]

                        #pnl /= pnl.std() * 100

                        sh = (np.sqrt(252) * sl.sharpe(pnl)).round(2)
                        MEANs = (252 * pnl.mean() * 100).round(2)
                        tConfDf = sl.tConfDF(pd.DataFrame(pnl).fillna(0), scalingFactor=252 * 100).set_index("index",drop=True).round(2)
                        STDs = (np.sqrt(250) * pnl.std() * 100).round(2)

                        statsMat = pd.concat([sh, MEANs, tConfDf, STDs], axis=1)
                        stats = pd.concat([statsMat.iloc[0, :], statsMat.iloc[1, :]], axis=0)
                        stats.index = ["GPR_sh", "GPR_Mean", "GPR_tConf", "GPR_Std", "RW_sh", "RW_Mean",
                                       "RW_tConf", "RW_Std"]
                        stats[["GPR_tConf", "RW_tConf"]] = stats[["GPR_tConf", "RW_tConf"]].astype(str)
                        stats["selection"] = selection
                        stats["kernel"] = kernelIn

                        shList.append(stats)
                    except Exception as e:
                        print(e)
                        notProcessed.append(selection + '_GPR_pnl_'+kernelIn+ '_' + str(rw))
            shDF = pd.concat(shList, axis=1).T.set_index("selection", drop=True).round(4)
            shDF.to_sql(Portfolios + '_sh_GPR_pnl_' + str(rw), conn, if_exists='replace')
            notProcessedDF = pd.DataFrame(notProcessed, columns=['NotProcessedProjection'])
            notProcessedDF.to_sql(Portfolios+'_notProcessedDF_GPR_' + str(rw), conn, if_exists='replace')

    elif scanMode == 'ScanNotProcessed':
        processList = []
        rw = 250
        notProcessedDF = pd.read_sql('SELECT * FROM '+Portfolios+'_notProcessedDF_GPR_' + str(rw), conn).set_index('index', drop=True)
        for idx, row in notProcessedDF.iterrows():
            splitInfo = row['NotProcessedProjection'].split("_GPR_pnl_")
            selection = splitInfo[0]
            kernelIn = str(splitInfo[1]).split("_")[0] + "_" + str(splitInfo[1]).split("_")[1]
            processList.append([selection, allProjectionsDF[selection], 0.3, kernelIn, rw])

        print("#GPR Processes = ", len(processList))
        p = mp.Pool(mp.cpu_count())
        result = p.map(GPRlocal, tqdm(processList))
        p.close()
        p.join()

    elif scanMode == 'ReportStatistics':
        shGPR = []
        rw = 250
        for kernelIn in kernelList:
            for selection in tqdm(allProjectionsDF.columns):
                try:
                    pnl = pd.read_sql('SELECT * FROM ' + selection + '_GPR_pnl_' + kernelIn + '_' + str(rw),
                                      conn).set_index('Dates', drop=True).iloc[round(0.3 * len(allProjectionsDF)):]
                    pnlSharpes = (np.sqrt(252) * sl.sharpe(pnl).round(4)).reset_index()
                    pnlSharpes['kernelIn'] = kernelIn

                    tConfDf_gpr = sl.tConfDF(pnl.fillna(0)).set_index("index", drop=True)

                    pnlSharpes = pnlSharpes.set_index("index", drop=True)
                    pnlSharpes = pd.concat(
                        [pnlSharpes, pnl.mean() * 100, tConfDf_gpr.astype(str), pnl.std() * 100], axis=1)
                    pnlSharpes.columns = ["pnlSharpes", "kernelIn", "pnl_mean", "tConfDf_sema", "pnl_std"]
                    pnlSharpes['selection'] = selection
                    pnlSharpes = pnlSharpes.set_index("selection", drop=True)
                    shGPR.append(pnlSharpes)
                except:
                    pass

        shGprDF = pd.concat(shGPR).round(4)
        shGprDF.to_sql('GPR_pnlSharpes_' + Portfolios, conn, if_exists='replace')

    elif scanMode == 'ReportSpecificStatistics':
        stats = pd.read_sql('SELECT * FROM GPR_pnlSharpes_'+Portfolios, conn)
        stats = stats[(stats['selection'].str.split("_").str[2].astype(float)<5)&(stats['kernelIn']=="RBF_DotProduct")].set_index("selection", drop=True)
        stats.to_sql('GPR_SpecificStatistics_' + Portfolios, conn, if_exists='replace')

def Test(mode):
    if mode == 'GPC':
        selection = 'PCA_ExpWindow25_2'
        trainLength = 0.3
        tw = 250
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection]
        rwDF = pd.read_sql('SELECT * FROM PCA_randomWalkPnlRSprojections_tw_ExpWindow25', conn).set_index('Dates',
                                                                                                          drop=True).iloc[
               round(0.3 * len(df)):, 2]
        medSh = (np.sqrt(252) * sl.sharpe(rwDF)).round(4)
        print("Random Walk Sharpe : ", medSh)
        # GaussianProcess_Results = sl.GPC_Walk(df, trainLength, tw)
        magicNum = 1
        params = {
            "TrainWindow": 5,
            "LearningMode": 'static',
            "Kernel": "DotProduct",
            "modelNum": magicNum,
            "TrainEndPct": 0.3,
            "writeLearnStructure": 0
        }
        out = sl.AI.gGPC(df, params)

        out[0].to_sql('df_real_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                      if_exists='replace')
        out[1].to_sql('df_predicted_price_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                      if_exists='replace')
        out[2].to_sql('df_predicted_proba_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn,
                      if_exists='replace')
        df_real_price = out[0]
        df_predicted_price = out[1]
        df_predicted_price.columns = df_real_price.columns
        # Returns Prediction
        sig = sl.sign(df_predicted_price)
        pnl = sig * df_real_price
        pnl.to_sql('pnl_GPC_TEST_' + params["Kernel"] + "_" + selection + str(magicNum), conn, if_exists='replace')
        print("pnl_GPC_TEST_sharpe = ", np.sqrt(252) * sl.sharpe(pnl))
        sl.cs(pnl).plot()
        print(out[2].tail(10))
        out[2].plot()
        plt.show()

    elif mode == 'GPR':
        selection = 'LLE_250_0'
        trainLength = 0.1
        kernelIn = "RBF"
        rw = 250
        df = pd.read_sql('SELECT * FROM allProjectionsDF', conn).set_index('Dates', drop=True)[selection].iloc[-250:]
        GPR_Results = sl.GPR_Walk(df, trainLength, kernelIn, rw)

        GPR_Results[0].to_sql(selection + '_GPR_testDF_' + kernelIn + '_' + str(rw), conn, if_exists='replace')
        GPR_Results[1].to_sql(selection + '_GPR_PredictionsDF_' + kernelIn + '_' + str(rw), conn,
                              if_exists='replace')

        pd.concat([sl.cs(GPR_Results[0]), sl.cs(GPR_Results[1])], axis=1).plot()
        plt.show()

        pickle.dump(GPR_Results[2],open(selection + '_GPR_gprparamList_' + kernelIn + '_' + str(rw) + ".p", "wb"))

        sig = sl.sign(GPR_Results[1])

        pnl = sig * GPR_Results[0]
        pnl_sh = np.sqrt(252) * sl.sharpe(pnl)
        print("pnl Sharpe = ", pnl_sh)
        pnl.to_sql(selection + '_GPR_pnl_' + kernelIn + '_' + str(rw), conn, if_exists='replace')

    elif mode == 'GPR_template_pymc3':
        sns.set_style(
            style='darkgrid',
            rc={'axes.facecolor': '.9', 'grid.color': '.8'}
        )
        sns.set_palette(palette='deep')
        sns_c = sns.color_palette(palette='deep')

        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['figure.dpi'] = 100

        np.random.seed(42)

        # Generate seasonal variables.
        def seasonal(t, amplitude, period):
            """Generate a sinusoidal curve."""
            return amplitude * np.sin((2 * np.pi * t) / period)

        def generate_data(n, sigma_n=0.3):
            """Generate sample data.
            Two seasonal components, one linear trend and gaussian noise.
            """
            # Define "time" variable.
            t = np.arange(n)
            data_df = pd.DataFrame({'t': t})
            # Add components:
            data_df['epsilon'] = np.random.normal(loc=0, scale=sigma_n, size=n)
            data_df['s1'] = data_df['t'].apply(lambda t: seasonal(t, amplitude=2, period=40))
            data_df['s2'] = data_df['t'].apply(lambda t: seasonal(t, amplitude=1, period=13.3))
            data_df['tr1'] = 0.01 * data_df['t']
            return data_df.eval('y = s1 + s2 + tr1 + epsilon')

        # Number of samples.
        # Generate data.
        #data_df = generate_data(n=n)
        data_df_raw = pd.read_sql('SELECT * FROM allProjectionsDF', conn).reset_index()
        data_df_raw['t'] = data_df_raw['index']
        data_df_raw['y'] = data_df_raw['LLE_250_0'].fillna(0)
        n = len(data_df_raw)
        #data_df_raw['y'] = sl.cs(data_df_raw['LLE_250_0'].fillna(0))
        data_df = data_df_raw[['t', 'y']]

        # Plot.
        fig, ax = plt.subplots()
        sns.lineplot(x='t', y='y', data=data_df, color=sns_c[0], label='y', ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set(title='Sample Data', xlabel='t', ylabel='')

        x = data_df['t'].values.reshape(n, 1)
        y = data_df['y'].values.reshape(n, 1)

        prop_train = 0.1
        n_train = round(prop_train * n)

        x_train = x[:n_train]
        y_train = y[:n_train]

        x_test = x[n_train:n_train+1]
        y_test = y[n_train:n_train+1]

        # Plot.
        #fig, ax = plt.subplots()
        #sns.lineplot(x=x_train.flatten(), y=y_train.flatten(), color=sns_c[0], label='y_train', ax=ax)
        #sns.lineplot(x=x_test.flatten(), y=y_test.flatten(), color=sns_c[1], label='y_test', ax=ax)
        #ax.axvline(x=x_train.flatten()[-1], color=sns_c[7], linestyle='--', label='train-test-split')
        #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        #ax.set(title='y train-test split ', xlabel='t', ylabel='')
        #plt.show()

        with pm.Model() as model:
            # First seasonal component.
            ls_1 = pm.Gamma(name='ls_1', alpha=2.0, beta=1.0)
            period_1 = pm.Gamma(name='period_1', alpha=80, beta=2)
            gp_1 = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_1, ls=ls_1))
            # Second seasonal component.
            #ls_2 = pm.Gamma(name='ls_2', alpha=2.0, beta=1.0)
            #period_2 = pm.Gamma(name='period_2', alpha=30, beta=2)
            #gp_2 = pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1, period=period_2, ls=ls_2))
            # Linear trend.
            #c_3 = pm.Normal(name='c_3', mu=np.mean(x_train), sigma=np.std(x_train))
            #gp_3 = pm.gp.Marginal(cov_func=pm.gp.cov.Linear(input_dim=1, c=c_3))
            # Define gaussian process.
            #gp = gp_1 + gp_2 + gp_3
            gp = gp_1
            print(gp)
            # Noise.
            sigma = pm.HalfNormal(name='sigma', sigma=10)
            # Likelihood.
            y_pred = gp.marginal_likelihood('y_pred', X=x_train, y=y_train.flatten(), noise=sigma)
            print(y_pred)
            # Sample.
            trace = pm.sample(draws=10, chains=1, tune=10)
            print(trace)

            az.plot_trace(trace)
            pm.summary(trace)
            plt.show()

        with model:
            x_train_conditional = gp.conditional('x_train_conditional', x_train)
            y_train_pred_samples = pm.sample_posterior_predictive(trace, vars=[x_train_conditional], samples=1)

            x_test_conditional = gp.conditional('x_test_conditional', x_test)
            y_test_pred_samples = pm.sample_posterior_predictive(trace, vars=[x_test_conditional], samples=1)

        # Train
        y_train_pred_samples_mean = y_train_pred_samples['x_train_conditional'].mean(axis=0)
        y_train_pred_samples_std = y_train_pred_samples['x_train_conditional'].std(axis=0)
        y_train_pred_samples_mean_plus = y_train_pred_samples_mean + 2 * y_train_pred_samples_std
        y_train_pred_samples_mean_minus = y_train_pred_samples_mean - 2 * y_train_pred_samples_std
        # Test
        y_test_pred_samples_mean = y_test_pred_samples['x_test_conditional'].mean(axis=0)
        y_test_pred_samples_std = y_test_pred_samples['x_test_conditional'].std(axis=0)
        y_test_pred_samples_mean_plus = y_test_pred_samples_mean + 2 * y_test_pred_samples_std
        y_test_pred_samples_mean_minus = y_test_pred_samples_mean - 2 * y_test_pred_samples_std

        print(y_test_pred_samples_mean)
        print(len(y_test_pred_samples_mean))
        print(y_test_pred_samples_std)
        print(len(y_test_pred_samples_std))

        time.sleep(3000)

        fig, ax = plt.subplots()
        sns.lineplot(x=x_train.flatten(), y=y_train.flatten(), color=sns_c[0], label='y_train', ax=ax)
        sns.lineplot(x=x_test.flatten(), y=y_test.flatten(), color=sns_c[1], label='y_test', ax=ax)
        ax.fill_between(
            x=x_train.flatten(),
            y1=y_train_pred_samples_mean_minus,
            y2=y_train_pred_samples_mean_plus,
            color=sns_c[2],
            alpha=0.2,
            label='credible_interval (train)'
        )
        sns.lineplot(x=x_train.flatten(), y=y_train_pred_samples_mean, color=sns_c[2], label='y_pred_train', ax=ax)
        ax.fill_between(
            x=x_test.flatten(),
            y1=y_test_pred_samples_mean_minus,
            y2=y_test_pred_samples_mean_plus,
            color=sns_c[3],
            alpha=0.2,
            label='credible_interval (test)'
        )
        sns.lineplot(x=x_test.flatten(), y=y_test_pred_samples_mean, color=sns_c[3], label='y_pred_test', ax=ax)
        ax.axvline(x=x_train.flatten()[-1], color=sns_c[7], linestyle='--', label='train-test-split')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set(title='Model Predictions', xlabel='t', ylabel='')
        plt.show()

#####################################################
#GPRonPortfolios("ClassicPortfolios", 'Main', "run")
#GPRonPortfolios("ClassicPortfolios", 'Main', "report")
#GPRonPortfolios("Projections", 'Main', "run")
#GPRonPortfolios("Projections", 'Main', "report")
#GPRonPortfolios("Projections", "ScanNotProcessed", "")
#GPRonPortfolios("globalProjections", 'Main', "run")
#GPRonPortfolios("globalProjections", 'Main', "report")
#GPRonPortfolios("globalProjections", "ScanNotProcessed", "")
#GPRonPortfolios("Projections", "ReportStatistics", "")
#GPRonPortfolios("Projections", "ReportSpecificStatistics", "")
#GPRonPortfolios("FinalistsProjections", 'Main', "run")
#GPRonPortfolios("FinalistsProjections", 'Main', "report")
#GPRonPortfolios("FinalistsProjections", 'ScanNotProcessed', "")
#GPRonPortfolios("FinalistsGlobalProjections", 'Main', "run")
#GPRonPortfolios("FinalistsGlobalProjections", 'Main', "report")
#GPRonPortfolios("FinalistsGlobalProjections", 'ScanNotProcessed', "")
#GPRonPortfolios("Specific", 'Main', "run")
#GPRonPortfolios("Specific", 'Main', "report")
#GPRonPortfolios("Specific", "ScanNotProcessed", "")

Test("GPR")
#Test("GPR_template_pymc3")
