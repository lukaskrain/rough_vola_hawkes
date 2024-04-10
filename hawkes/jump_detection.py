import numpy as np
import pandas as pd
import math
from math import ceil, sqrt
import matplotlib.pyplot as plt
import os
import warnings
import Hawkes as hk
import scipy.stats
from scipy.stats import invweibull


class jump_detection():
    
    def __init__(self, x, window):
        self.ts = x
        self.rets = x.dX.values
        self.window = window
        self.jumps = None
        self.jump_indices = None
        self.sizes = None
        self.jumpless_ts = None
        self.hawkes_param = None
        self.p_up = None
        self.frechet_param = None
    
    def peak_over_threshold(self, n_sigma=2, return_type=False):
        mu = self.rets.mean()
        sigma = self.rets.std()
        threshold = mu + n_sigma * sigma
        jump_indices = np.where(abs(self.rets) > threshold)[0]
        jump_sizes = self.rets[jump_indices]
        jumpless_ts = self.ts.copy()
        jumpless_ts.values[jump_indices] = 0
        
        self.PoTjump_indices = jump_indices
        self.PoTsizes = jump_sizes
        
        if return_type:
            return jumpless_ts, jump_indices, jump_sizes
        
    @staticmethod
    def movmean(v, kb, kf):
        m = len(v) * [np.nan]
        for i in range(kb, len(v)-kf):
            m[i] = np.mean(v[i-kb:i+kf+1])
        return m

    def LeeMykland(self, sampling, significance_level=0.01):
        tm = 86400//self.window
        k = sampling
        r = np.append(np.nan, self.rets)
        bpv = np.multiply(np.absolute(r[:]), np.absolute(np.append(np.nan, r[:-1])))
        bpv = np.append(np.nan, bpv[0:-1]).reshape(-1,1) 
        sig = np.sqrt(self.movmean(bpv, k-3, 0)) 
        L   = r/sig
        n   = np.size(self.rets) 
        c   = (2/np.pi)**0.5
        Sn  = c*(2*np.log(n))**0.5
        Cn  = (2*np.log(n))**0.5/c - np.log(np.pi*np.log(n))/(2*c*(2*np.log(n))**0.5)
        beta_star   = -np.log(-np.log(1-significance_level)) # Jump threshold
        T   = (abs(L)-Cn)*Sn
        J   = (T > beta_star).astype(float)
        J   = J*np.sign(r)
        # First k rows are NaN involved in bipower variation estimation are set to NaN.
        J[0:k] = np.nan
        
        LMjumps = pd.DataFrame({'L': L,'sig': sig, 'T': T,'J':J})
        self.LMjump_indices = LMjumps.index[(LMjumps.J > 0) | (LMjumps.J < 0)].values
        self.LMsizes = self.rets[self.LMjump_indices]
        
        return LMjumps
    
    def write_jump_data(self, file_name_jumpless, file_name_jumps, jump_test="PoT"):
        jumpless_ts = self.ts.copy(deep = True)
        if jump_test == "PoT":
            j_ind = self.PoTjump_indices
            j_sizes = self.PoTsizes
        elif jump_test == "LM":
            j_ind = self.LMjump_indices
            j_sizes = self.LMsizes
        else:
            print("Unvalid Jump Test")
            return None
        if len(j_ind) > 0:
            jumpless_ts.dX.loc[j_ind] = np.zeros(len(j_ind))
        
        self.jumpless_ts = jumpless_ts
        jumpless_ts.to_csv(file_name_jumpless, index = True)
        
        jump_data = pd.DataFrame([j_ind, j_sizes], index = ["Jump_Indices", "Jump_Sizes"]).T
        jump_data.to_csv(file_name_jumps, index = True)
        
    def estimate_hawkes(self, kernel="exp", baseline="const", jump_test="PoT"):
        if jump_test == "PoT":
            j_ind = self.PoTjump_indices
        elif jump_test == "LM":
            j_ind = self.LMjump_indices
        else:
            print("Unvalid Jump Test")
            return None
        model = hk.estimator()
        model.set_kernel(kernel)
        model.set_baseline(baseline)
        interval = [0, self.window]
        model.fit(np.array(j_ind).astype(np.float64), interval)
        self.hawkes_param = model.parameter
        
    def estimate_size(self, jump_test="PoT"):
        if jump_test == "PoT":
            j_sizes = self.PoTsizes
        elif jump_test == "LM":
            j_sizes = self.LMsizes
        else:
            print("Unvalid Jump Test")
            return None
        self.p_up = (j_sizes > 0).sum() / len(j_sizes)
        self.frechet_param = invweibull.fit(np.abs(j_sizes))
        
    def simulate_jump_times(self, kernel="exp", baseline="const"):
        ind_sim = hk.simulator().set_kernel(kernel).set_baseline(baseline)
        ind_sim.set_parameter(self.hawkes_param)
        interval = [0, self.window]
        return ind_sim.simulate(interval)
    
    def simulate_jumps(self):
        sim_indices = np.unique(self.simulate_jump_times().astype(int))
        sim_n = len(sim_indices)
        sim_jumps = np.multiply(invweibull.rvs(self.frechet_param[0],
                                               loc=self.frechet_param[1],
                                               scale=self.frechet_param[2],
                                               size=sim_n),
                                np.random.choice([1, -1],
                                                 size = sim_n,
                                                 p=[self.p_up, 1-self.p_up]))
        return sim_jumps
        
        
        
    
if __name__ == '__main__':
    path_name = 'path/to/raw/data'
    file_name = 'sample/name/of/logret/file.csv' # must have one column called 'dX'
    ts = pd.read_csv(path_name + "LogReturns/" + file_name)
    ts.head()
    
    jumps = jump_detection(ts, window = 288)
    
    jumps.peak_over_threshold()
    jumps.estimate_hawkes()
    jumps.estimate_size()
    
    all_files = []
    folder_path = 'path/to/directory/with/logret/data'
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            all_files.append(file_path)
    
    jump_parameter = pd.DataFrame(columns = ["Day", "hawkes_mu", "hawkes_alpha", "hawkes_beta", "frechet_c", "frechet_loc", "frechet_scale", "p_up"])
    
    for i, filename in enumerate(sorted(os.listdir(folder_path))):
        n = len(sorted(os.listdir(folder_path)))
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            ts = pd.read_csv(path_name + "LogReturns/" + filename)
            ts.head()

            jumps = jump_detection(ts, window = 288)

            jumps.peak_over_threshold(n_sigma=2)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            jumps.estimate_hawkes()
            jumps.estimate_size()

            
            j_param = pd.DataFrame([[filename[19:-4]] + list(jumps.hawkes_param.values()) + list(jumps.frechet_param) + [jumps.p_up]])
            j_param.columns = ["Day", "hawkes_mu", "hawkes_alpha", "hawkes_beta", "frechet_c", "frechet_loc", "frechet_scale", "p_up"]
            if len(jumps.PoTsizes) < 5:
                print(filename + " has too few jumps.")
            elif jumps.hawkes_param["alpha"] * jumps.hawkes_param["beta"] > 1:
                print(filename + " has parameters that are too high")
            
            warnings.filterwarnings("ignore", category=FutureWarning)
            jump_parameter = jump_parameter.append(j_param, ignore_index = True)
            
            if i%25 == 0:
                print(str(i+1)+". files done. "+str(len(all_files))+" total.")        

    warnings.resetwarnings()
    jump_parameter.to_csv(path_name + "jump_parameters.csv", index=False)
