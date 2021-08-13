from Slider import Slider as sl
import numpy as np, time, pickle
import pandas as pd
from tqdm import tqdm
import warnings, sqlite3, os, tensorflow as tf
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats as st
import seaborn as sns

def Run():
    outList = pickle.load(open("F:\Dealing\Panagiotis Papaioannou\AcademicsCode\PyPhD_Github-master\PyPhD_Github-master\DMAP_Lift_GeometricHarmonics_500.p","rb"))
    print("Len Outlist = ", len(outList))
    #df = outList[0]
    df = pd.read_sql('SELECT * FROM FxDataAdjRets', sqlite3.connect('FXeodData_FxData.db')).set_index('Dates', drop=True)
    #sl.cs(elem0).plot()
    #plt.show()

    LO = sl.E(df)
    print("LO Sharpe = ", np.sqrt(252) * sl.sharpe(LO))
    RP = sl.rs(sl.rp(df))
    print("RP Sharpe = ", np.sqrt(252) * sl.sharpe(RP))

    #print(elem0)
    elem1 = outList[1]
    #print(elem1)
    Loadings_TemporalResidual = elem1[0]
    psi_all = elem1[1]

    psi_all_hat_var_array = elem1[2]
    psi_all_hat_var_pvals_array = elem1[3]
    extrapolatedPsi_to_X_var = elem1[4]
    psi_all_hat_gpr_array = elem1[5]
    psi_all_hat_gpr_score_array = elem1[6]
    extrapolatedPsi_to_X_gpr = elem1[7]
    psi_all_hat_nn1_array = elem1[8]
    psi_all_hat_nn1_score_array = elem1[9]
    extrapolatedPsi_to_X_nn1 = elem1[10]
    psi_all_hat_nn2_array = elem1[11]
    psi_all_hat_nn2_score_array = elem1[12]
    extrapolatedPsi_to_X_nn2 = elem1[13]

    fig, ax = plt.subplots(sharex=True, nrows=4, ncols=1)
    sl.cs(extrapolatedPsi_to_X_var[1]).plot(ax=ax[0], legend=None)
    sl.cs(extrapolatedPsi_to_X_gpr[1]).plot(ax=ax[1], legend=None)
    sl.cs(extrapolatedPsi_to_X_nn1[1]).plot(ax=ax[2], legend=None)
    sl.cs(extrapolatedPsi_to_X_nn2[1]).plot(ax=ax[3], legend=None)

    for trSigSA in [[extrapolatedPsi_to_X_var[1], "var"], [extrapolatedPsi_to_X_gpr[1], "gpr"], [extrapolatedPsi_to_X_nn1[1], "nn1"], [extrapolatedPsi_to_X_nn2[1], "nn2"]]:
        #pnl = sl.sign(trSigSA[0]) * df
        #pnl = sl.sign(trSigSA[0]) * sl.rp(df)
        #pnl = sl.S(sl.sign(trSig[0])) * df
        pnl = sl.S(sl.sign(trSigSA[0])) * sl.rp(df)
        #pnl = sl.sign(trSig[0]) * sl.S(df)
        rs_pnl = sl.rs(pnl)
        sh_pnl = np.sqrt(252) * sl.sharpe(rs_pnl)
        print(trSigSA[1], sh_pnl)

    for trSigSB in [[psi_all_hat_var_array, "var"], [psi_all_hat_gpr_array, "gpr"], [psi_all_hat_nn1_array, "nn1"], [psi_all_hat_nn2_array, "nn2"]]:
        pnl_LO = sl.sign(trSigSB[0][1]).mul(LO, axis=0)
        print("LO : ", trSigSA[1], np.sqrt(252) * sl.sharpe(pnl_LO))
        pnl_RP = sl.sign(trSigSB[0][1]).mul(RP, axis=0)
        print("RP : ", trSigSA[1], np.sqrt(252) * sl.sharpe(pnl_RP))

    plt.show()

    #print(elem1)
    #elem2 = outList[2]
    #print(elem2)
    #elem3 = outList[3]
    #print(elem3)

Run()