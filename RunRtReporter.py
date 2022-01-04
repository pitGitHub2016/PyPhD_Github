import pandas as pd
import numpy as np
import time
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import stats as sps
from scipy.interpolate import interp1d
import multiprocessing as mp
import os, itertools
import seaborn as sns

# Working Paths
serverEpiFolder = "C:\\Users\\lucia\\Desktop\\EpidemicModel\\"
os.chdir(serverEpiFolder+"WorkingDataset\\")
pathRawData = serverEpiFolder+"dati-regioni\dati-regioni\\"
pathWorkingData = ""
RtFolder = "ItalyInteractiveMap_Active/it-js-map/RtFolder/"
totalCases = pd.read_excel(pathWorkingData + "totale_casi.xlsx").ffill()
totalCases['data'] = pd.to_datetime(totalCases['data'])
totalCases = totalCases.set_index('data', drop=True)

# Main parameters
GAMMA = 1/7
# We create an array for every possible value of Rt
R_T_MAX = 10
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX * 100 + 1)

def highest_density_interval(pmf, p=.9, debug=False):

    # If we pass a DataFrame, just call this recursively on the columns
    if (isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col], p=p) for col in pmf],
                            index=pmf.columns)

    cumsum = np.cumsum(pmf.values)

    # N x N matrix of total probability mass for each low, high
    total_p = cumsum - cumsum[:, None]

    # Return all indices with total_p > p
    lows, highs = (total_p > p).nonzero()
    #print(total_p)
    #print(lows)
    #print(highs)
    # Find the smallest range (highest density)
    best = (highs - lows).argmin()

    low = pmf.index[lows[best]]
    high = pmf.index[highs[best]]

    return pd.Series([low, high],
                     index=['Low', 'High'])

def mainTheory():

    FILTERED_REGION_CODES = ['AS', 'GU', 'PR', 'VI', 'MP']

    # Column vector of k
    k = np.arange(0, 70)[:, None]

    # Different values of Lambda
    lambdas = [10, 20, 30, 40]

    # Evaluated the Probability Mass Function (remember: poisson is discrete)
    y = sps.poisson.pmf(k, lambdas)

    # Show the resulting shape
    print(y.shape)

    fig, ax = plt.subplots(figsize=(6,2.5))
    ax.set(title='Poisson Distribution of Cases\n $p(k|\lambda)$')
    plt.plot(k, y,
             marker='o',
             markersize=3,
             lw=0)
    plt.legend(title="$\lambda$", labels=lambdas)

    k = 20
    lam = np.linspace(1, 45, 90)
    likelihood = pd.Series(data=sps.poisson.pmf(k, lam),
                           index=pd.Index(lam, name='$\lambda$'),
                           name='lambda')
    fig1, ax1 = plt.subplots(figsize=(6,2.5))
    likelihood.plot(title=r'Likelihood $P\left(k_t=20|\lambda\right)$', figsize=(6,2.5))
    k = np.array([20, 40, 55, 90])

    # Gamma is 1/serial interval
    # https://wwwnc.cdc.gov/eid/article/26/7/20-0282_article
    # https://www.nejm.org/doi/full/10.1056/NEJMoa2001316


    # Map Rt into lambda so we can substitute it into the equation below
    # Note that we have N-1 lambdas because on the first day of an outbreak
    # you do not know what to expect.
    lam = k[:-1] * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # Evaluate the likelihood on each day and normalize sum of each day to 1.0
    likelihood_r_t = sps.poisson.pmf(k[1:], lam)
    likelihood_r_t /= np.sum(likelihood_r_t, axis=0)

    # Plot it
    fig2, ax2 = plt.subplots(figsize=(6,2.5))
    ax = pd.DataFrame(
        data = likelihood_r_t,
        index = r_t_range
    ).plot(
        title='Likelihood of $R_t$ given $k$',
        xlim=(0,10),
        figsize=(6,2.5)
    )

    ax.legend(labels=k[1:], title='New Cases')
    ax.set_xlabel('$R_t$')

    posteriors = likelihood_r_t.cumprod(axis=1)
    posteriors = posteriors / np.sum(posteriors, axis=0)

    columns = pd.Index(range(1, posteriors.shape[1]+1), name='Day')
    posteriors = pd.DataFrame(
        data = posteriors,
        index = r_t_range,
        columns = columns)

    fig3, ax3 = plt.subplots(figsize=(6,2.5))
    ax = posteriors.plot(
        title='Posterior $P(R_t|k)$',
        xlim=(0,10),
        figsize=(6,2.5)
    )
    ax.legend(title='Day')
    ax.set_xlabel('$R_t$')

    most_likely_values = posteriors.idxmax(axis=0)
    print(most_likely_values)

    hdi = highest_density_interval(posteriors, debug=True)

    fig4, ax4 = plt.subplots(figsize=(6,2.5))
    ax = most_likely_values.plot(marker='o',
                                 label='Most Likely',
                                 title=f'$R_t$ by day',
                                 c='k',
                                 markersize=4)

    ax.fill_between(hdi.index,
                    hdi['Low_90'],
                    hdi['High_90'],
                    color='k',
                    alpha=.1,
                    lw=0,
                    label='HDI')

    ax.legend()
    plt.show()

def TemplateDataUS():
    url = 'https://covidtracking.com/api/v1/states/daily.csv'
    states = pd.read_csv(url,
                         usecols=['date', 'state', 'positive'],
                         parse_dates=['date'],
                         index_col=['state', 'date'],
                         squeeze=True).sort_index()

    state_name = 'NY'
    cases = states.xs(state_name).rename(f"{state_name} cases")

# ///////////////////////////////////////////////////////////////////////////////////////////////////

def prepare_cases(cases, cutoff=25):
    new_cases = cases.diff().rolling(3).mean()
    #original = new_cases.copy()

    smoothed = new_cases.rolling(7,
                                 win_type='gaussian',
                                 min_periods=1,
                                 center=True).mean(std=2).round()

    idx_start = np.searchsorted(smoothed, cutoff)
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    return original, smoothed

def get_posteriors(sr, sigma=0.25):
    # (1) Calculate Lambda
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # (2) Calculate each day's likelihood
    likelihoods = pd.DataFrame(
        data=sps.poisson.pmf(sr[1:].values, lam),
        index=r_t_range,
        columns=sr.index[1:])

    # (3) Create the Gaussian Matrix
    process_matrix = sps.norm(loc=r_t_range,
                              scale=sigma
                              ).pdf(r_t_range[:, None])

    # (3a) Normalize all rows to sum to 1
    process_matrix /= process_matrix.sum(axis=0)

    # (4) Calculate the initial prior
    # prior0 = sps.gamma(a=4).pdf(r_t_range)
    prior0 = np.ones_like(r_t_range) / len(r_t_range)
    prior0 /= prior0.sum()

    # Create a DataFrame that will hold our posteriors for each day
    # Insert our prior as the first posterior.
    posteriors = pd.DataFrame(
        index=r_t_range,
        columns=sr.index,
        data={sr.index[0]: prior0}
    )

    # We said we'd keep track of the sum of the log of the probability
    # of the data for maximum likelihood calculation.
    log_likelihood = 0.0

    # (5) Iteratively apply Bayes' rule
    for previous_day, current_day in zip(sr.index[:-1], sr.index[1:]):
        # (5a) Calculate the new prior
        current_prior = process_matrix @ posteriors[previous_day]

        # (5b) Calculate the numerator of Bayes' Rule: P(k|R_t)P(R_t)
        numerator = likelihoods[current_day] * current_prior

        # (5c) Calcluate the denominator of Bayes' Rule P(k)
        denominator = np.sum(numerator)

        # Execute full Bayes' Rule
        posteriors[current_day] = numerator / denominator

        # Add to the running sum of log likelihoods
        log_likelihood += np.log(denominator)

    return posteriors, log_likelihood

def plot_rt(result, ax, fig):

    # Colors
    ABOVE = [1, 0, 0]
    MIDDLE = [1, 1, 1]
    BELOW = [0, 0, 0]
    cmap = ListedColormap(np.r_[
                              np.linspace(BELOW, MIDDLE, 25),
                              np.linspace(MIDDLE, ABOVE, 25)
                          ])
    color_mapped = lambda y: np.clip(y, .5, 1.5) - .5

    index = result['ML'].index.get_level_values('data')
    values = result['ML'].values

    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)

    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')

    highfn = interp1d(date2num(index),
                      result['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')

    extended = pd.date_range(start=index[0], end=index[-1] + pd.Timedelta(days=1))

    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25)

    # Formatting
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    #ax.xaxis.set_minor_locator(mdates.DayLocator())
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))

    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    #ax.set_ylim(0.0, 5.0)
    ax.set_xlim(pd.Timestamp(index[0]), result.index.get_level_values('data')[-1] + pd.Timedelta(days=1))
    fig.set_facecolor('w')

def RegionRt(processParamsIn):
    TargetRegion = processParamsIn[0]
    CI_thr = processParamsIn[1]

    cases = totalCases[[x for x in totalCases.columns if TargetRegion in x]].iloc[:,0]

    original, smoothed = prepare_cases(cases)

    #original.plot(title=f"{TargetRegion} New Cases per Day", c='k', linestyle=':', alpha=.5, label='Actual', legend=True)
    #ax = smoothed.plot(label='Smoothed', legend=True)
    #ax.get_figure().set_facecolor('w')

    # Note that we're fixing sigma to a value just for the example
    posteriors, log_likelihood = get_posteriors(smoothed, sigma=.1)

    #ax = posteriors.plot(title=f'{TargetRegion} - Daily Posterior for $R_t$', legend=False, lw=1, c='k', alpha=.3, xlim=(0.4, 6))
    #ax.set_xlabel('$R_t$')

    posteriors = posteriors.ffill(axis=1)
    #print(posteriors)

    # Note that this takes a while to execute - it's not the most efficient algorithm
    hdis = highest_density_interval(posteriors, p=CI_thr/100)
    most_likely = posteriors.idxmax().rename('ML')
    result = pd.concat([most_likely, hdis], axis=1)
    result = result.iloc[-30*6:, :]

    "get last month"
    #result = result.iloc[-30:,:]

    fig, ax = plt.subplots(figsize=(500/72,400/72))
    plot_rt(result, ax, fig)
    valuesLabeled = str(result.round(2).values[-1,0]) #+ ' (' + str(result.round(2).values[-1,1]) + ', ' + str(result.round(2).values[-1,2]) + ')'
    ax.set_title('Latest $R_t$ = ' + valuesLabeled)
    #ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    for label in ax.get_xticklabels():
        label.set_fontsize(10)
        label.set_ha("left")
        label.set_rotation(45)
    plt.savefig(RtFolder+TargetRegion+".png", dpi=65)
    #plt.show()

    out = [TargetRegion, result.round(2).values[-1,0]]
    return out

def vals2colors(vals, cmap='icefire',res=100):

    """Maps values to colors
    Args:
    values (list or list of lists) - list of values to map to colors
    cmap (str) - color map (default is 'Spectral')
    res (int) - resolution of the color map (default: 100)
    Returns:
    list of rgb tuples
    """
    # flatten if list of lists
    if any(isinstance(el, list) for el in vals):
        vals = list(itertools.chain(*vals))

    # get palette from seaborn
    palette = np.array(sns.color_palette(cmap, res).as_hex())
    #ranks = np.digitize(vals, np.linspace(np.min(vals), np.max(vals)+1, res+1)) - 1
    ranks = np.digitize(vals, np.linspace(0, 3, res+1)) - 1
    return palette[ranks]

# ///////// RUNNERS ///////
#TemplateDataUS()
#outAllDF = pd.read_excel(RtFolder + "ALL_Rt_Values.xlsx")

if __name__ == '__main__':
    processList = []
    for regionRaw in totalCases.columns:
        region = regionRaw.replace("totale_casi_", "")
        processParams = [region, 9]
        processList.append(processParams)

    p = mp.Pool(mp.cpu_count())
    outAll = p.map(RegionRt, tqdm(processList))
    p.close()
    p.join()

    outAllDF = pd.DataFrame(outAll, columns=['Region' + ' ' + str(totalCases.index[-1]).split(" ")[0], 'Rt'])
    outAllDF['color'] = pd.Series(vals2colors(outAllDF['Rt'].values))
    outAllDF.to_excel(RtFolder+"ALL_Rt_Values.xlsx", index=False)
    outAllDF.to_csv(RtFolder + "ALL_Rt_Values_Colored.csv", index=False)
    outAllDF.to_json(RtFolder+"ALL_Rt_Values_Colored.json", orient="split")

    with open(RtFolder+"ALL_Rt_Values_Colored.json", 'r') as file:
        data = json.load(file)
        data = 'var colorData = '+ str(data)
        with open(RtFolder+"ALL_Rt_Values_Colored.js", "w") as text_file:
            text_file.write(data)
