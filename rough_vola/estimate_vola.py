import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import os
from hurst import compute_Hc

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    
    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]

class volatility():
    
    def __init__(self, x):
        self.ts = x
        self.est_RV_power = None
        self.est_RV_bipower = None
        self.est_naive_RV = None
        self.n = len(x)
    
    def RV_power(self, order=2):
        self.est_RV = (abs(self.ts.dX.values)**order).sum()
        return self.est_RV_power
    
    def RV_bipower(self, k=300):
        log_ret = []
        for i in range(2*k, len(self.ts.Close), k):
            log_ret.append(abs(np.log(self.ts.Close[i]) - np.log(self.ts.Close[i-k])) * abs(np.log(self.ts.Close[i-k]) - np.log(self.ts.Close[i-2*k])))
        self.est_RV = np.array(log_ret).sum()
        return self.est_RV_bipower
    
    def naive_RV(self, k=300):
        self.est_naive_RV = np.diff(np.log(self.ts.Close)).std()
        return 

class OU_process():
    
    def __init__(self, RV):
        self.vola = RV
        self.n = len(RV)
        self.H = None
        self.OU_params = None
    
    def Hurst_diff_of_diff(self):
        z = 0
        n = 0
        for i in range(self.n-4):
            z += (self.vola[i+4] - 2*self.vola[i+2] + self.vola[i])**2
        for i in range(self.n-2):
            n += (self.vola[i+2] - 2*self.vola[i+1] + self.vola[i])**2
        self.H = 0.5*math.log(z/n, 2)
        return self.H
    
    def Hurst_r_s(self):
        self.H = get_hurst_exponent(self.vola)[0]
    
    def second_stage_approx(self, delta=1):
        if not self.H:
            print("Please estimate the Hurst exponent first.")
            return None
        diff_of_diff = 0
        for i in range(self.n-2):
            diff_of_diff += (self.vola[i+2] - 2*self.vola[i+1] + self.vola[i])**2
        
        sigma = math.sqrt(diff_of_diff / (self.n*(4-2**(2*self.H))*delta**(2*self.H)))
        mu = np.array(self.vola).mean()
        kappa = ((self.n*(np.array(self.vola)**2).sum()-np.array(self.vola).sum()**2) / (self.n**2*sigma**2*self.H*math.gamma(2*self.H)))**(-1/(2*self.H))
        
        self.OU_params = [sigma, mu, kappa]
        return self.OU_params
    
    def sim_vola(self, n, start_value = self.OU_params[1]):
        N = 86400  # time steps
        paths = 2 # number of paths
        T = 5
        T_vec, dt = np.linspace(0, T, N, retstep=True)
        
        sigma = self.OU_params[0]
        mu = self.OU_params[1]
        kappa = self.OU_params[2]
        std_asy = np.sqrt(sigma**2 / (2 * kappa))

        X0 = start_value
        X = np.zeros((N, paths))
        X[0, :] = X0
        W = scipy.stats.norm.rvs(loc=0, scale=1, size=(N - 1, paths))

        std_dt = np.sqrt(sigma**2 / (2 * kappa) * (1 - np.exp(-2 * kappa * dt)))
        for t in range(0, N - 1):
            X[t + 1, :] = mu + np.exp(-kappa * dt) * (X[t, :] - mu) + std_dt * W[t, :]
        
        return X

        


if __name__ == '__main__':
    
    
    all_files = []
    folder_path = 'path/to/logret/files.csv' # must have a column called dX
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            all_files.append(file_path)
    
    est_RV_power = []
    est_RV_bipower = []
    est_RV = []
    for i, file_path in enumerate(all_files):
        data = pd.read_csv(file_path)
        vola = volatility(data)
        est_RV_power.append(vola.RV_power(k=300))
        est_RV_bipower.append(vola.RV_bipower(k=300))
        est_RV = np.diff(np.log(data.Close[::300])).std()
        if i%25 == 0:
            print(str(i+1)+". files done. "+str(len(all_files))+" total.")
    
    plt.rcParams["figure.figsize"] = (25,6)
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(np.array(est_RV_power))
    axs[0].plot(np.array(est_RV_bipower))
    axs[0].set_title("Estimated Daily Realized Volatility", fontsize = 20)
    axs[0].set_xlabel("Days", fontsize = 20)
    axs[0].set_ylabel("Realized Volatility", fontsize = 20)
    axs[0].tick_params(axis='both', which='major', labelsize=20)

    axs[1].plot(np.diff(np.log(est_RV_power)))
    axs[1].plot(np.diff(np.log(est_RV_bipower)))
    axs[1].set_title("Difference in Log-Volatility", fontsize = 20)
    axs[1].set_xlabel("Days", fontsize = 20)
    axs[1].set_ylabel(r'$\Delta$ Log-Volatility', fontsize = 20)
    axs[1].tick_params(axis='both', which='major', labelsize=20)
    
    plt.show()
    
    vola = OU_process(est_RV_power_jl)
    print(compute_Hc(np.log(np.array(vola.vola)))[0])
    plt.plot(np.array(vola.vola))
    plt.show()
    
    vola.Hurst_diff_of_diff()
    vola.second_stage_approx()
    
