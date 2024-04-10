import numpy as np
import pandas as pd
import os



all_files = []
folder_path = 'path/to/raw/data'
for i, filename in enumerate(sorted(os.listdir(folder_path))):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        all_files.append(file_path)

        
        
open_close_file = []
directory = 'path/to/raw/data'
k = 300

for i, filename in enumerate(sorted(os.listdir(folder_path))):
    n = len(sorted(os.listdir(folder_path)))
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        ts = pd.read_csv(folder_path+filename, header = None)[[0,4]]
        if ts.shape[0] != 86400:
            print(filename + " does not have the right length.")
            continue
        ts.columns = ["Time", "Close"]
        x0 = ts.Close.iloc[0]
        xT = ts.Close.iloc[86399]
        day = filename[11:-4]
        open_close_file.append([day, x0, xT])
        log_rets = np.append(np.diff(np.log(ts.Close[::k])), np.log(ts.Close[86399]) - np.log(ts.Close[86400-k]))
        log_rets = pd.DataFrame(log_rets, columns = ["dX"])
        log_rets.to_csv(directory + "LogReturns/logrets_"+filename, index = False)
        
        if i%50 == 0:
            print(str(i+1)+". files done. "+str(len(all_files))+" total.")
print("DONE")
        
open_close_file = pd.DataFrame(open_close_file, columns = ["Day", "Open", "Close"])
open_close_file.to_csv(directory + "open_close_prices.csv", index=False)
        