import math
import numpy as np
import xlwings as xw

def divGtN0(n,n0):
    # Find all divisors of the natural number N greater or equal to N0
    xx = np.arange(n0,math.floor(n/2)+1)
    return xx[(n/xx)==np.floor(n/xx)]

def rscalc(z, n):
    m = int(z.shape[0]/n)
    y = np.reshape(z, (m, n)).T
    mu= np.mean(y, axis=0)
    sigma = np.std(y, ddof=1, axis=0)
    y=np.cumsum(y-mu, axis=0)
    yrng = np.max(y, axis=0)-np.min(y, axis=0)
    return np.mean(yrng/sigma)

def hurstExponent(x,d=50):
    # Find such a natural number OptN that possesses the largest number of 
    # divisors among all natural numbers in the interval [0.99*N,N] 
    dmin, N, N0 = d, x.shape[0], math.floor(0.99*x.shape[0])
    dv = np.zeros( (N-N0+1,) )
    for i in range(N0, N+1):
        dv[i-N0] = divGtN0(i, dmin).shape[0]
    optN = N0+np.max(np.arange(0,N-N0+1)[max(dv)==dv])
    # Use the first OptN values of x for further analysis
    x = x[:optN]
    d = divGtN0(optN, dmin)

    N = d.shape[0]
    RSe, ERS = np.zeros( (N,)), np.zeros( (N,))

    # Calculate empirical R/S
    for i in range(N):
        RSe[i]= rscalc(x, d[i])

    # Compute Anis-Lloyd [1] and Peters [3] corrected theoretical E(R/S)
    # (see [4] for details)
    for i in range(N):
        n = d[i]
        K = np.arange(1,n)
        ratio = (n-0.5)/n * np.sum(np.sqrt((np.ones((n-1))*n-K)/K))
        if n>340:
            ERS[i]=ratio/math.sqrt(0.5*math.pi*n)
        else:
            ERS[i]=(math.gamma(0.5*(n-1))*ratio) / (math.gamma(0.5*n)*math.sqrt(math.pi))

    # Calculate the Anis-Lloyd/Peters corrected Hurst exponent
    # Compute the Hurst exponent as the slope on a loglog scale
    ERSal = np.sqrt(0.5*math.pi*d)
    Pal=np.polyfit(np.log10(d), np.log10(RSe-ERS+ERSal),1)
    Hal=Pal[0]

    # Calculate the empirical and theoretical Hurst exponents
    Pe = np.polyfit(np.log10(d),np.log10(RSe),1)
    He = Pe[0]
    P = np.polyfit(np.log10(d),np.log10(ERS),1)
    Ht = P[0]

    # Compute empirical confidence intervals (see [4])
    L = math.log2(optN)
    # R/S-AL (min(divisor)>50) two-sided empirical confidence intervals
    #pval95 = np.array([0.5-exp(-7.33*log(log(L))+4.21) exp(-7.20*log(log(L))+4.04)+0.5])
    lnlnL = math.log(math.log(L))
    c1 = [0.5-math.exp(-7.35*lnlnL+4.06), math.exp(-7.07*lnlnL+3.75)+0.5, 0.90]
    c2 = [0.5-math.exp(-7.33*lnlnL+4.21), math.exp(-7.20*lnlnL+4.04)+0.5, 0.95]
    c3 = [0.5-math.exp(-7.19*lnlnL+4.34), math.exp(-7.51*lnlnL+4.58)+0.5, 0.99]
    C= np.array([c1, c2, c3])
    
    detail = (d, optN, RSe, ERS, ERSal)
    return (Hal, He, Ht, C, detail)

@xw.func
@xw.arg('x', pd.DataFrame, index=False, header=False)
def hurst(x):
    Hal, He, Ht, C, detail = hurstExponent(xx)
    return He

 
if __name__ == "__main__":
    z=np.sin(np.linspace(0.0, 50.0, 1000))
    Hal, He, Ht, C, detail = hurstExponent(z)

    d, optN, RSe, ERS, ERSal = detail
    print('-----------------------------------------------------------------')
    print('R/S-AL using %d divisors (%d ,..., %d) for sample of %d values' % (d.shape[0], d[0], d[-1], optN) )
    print('Corrected theoretical Hurst exponent    %.3f' % 0.5)
    print('Corrected empirical Hurst exponent      %.3f' % Hal)
    print('Theoretical Hurst exponent              %.3f' % Ht )
    print('Empirical Hurst exponent                %.3f' % He)
    print('-----------------------------------------------------------------')
    print('R/S-AL (min(divisor)>50) two-sided empirical confidence intervals')
    print('conf_lo  conf_hi level ------------------------------------------')
    with np.printoptions(precision=4, suppress=True):
        print(C)
    print('-----------------------------------------------------------------')

    import matplotlib.pyplot as plt
    plt.plot(np.log10(d), np.log10(ERSal/(ERS[0]/RSe[0])), 'b-')
    plt.plot(np.log10(d), np.log10(RSe-ERS+ERSal), 'go')
    plt.show()
