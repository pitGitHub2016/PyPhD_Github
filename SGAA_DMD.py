from Slider import Slider as sl
from scipy.linalg import svd
import numpy as np, investpy, json, time, pickle, glob
from tqdm import tqdm
import pandas as pd
import warnings, sqlite3, os, tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
from sklearn.linear_model import LinearRegression
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from pydiffmap import kernel
import pyriemann as pr
from pyriemann.utils import distance as mt
from scipy import stats as st
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 20
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_digits
from sklearn.manifold import LocallyLinearEmbedding
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore')
pd.set_option('display.width', 320)
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',200)

conn = sqlite3.connect('SmartGlobalAssetAllocation.db')

