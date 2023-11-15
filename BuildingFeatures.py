import pandas as pd
import numpy as np
from datetime import datetime
import glob 
import os
import random
import scipy
from scipy import spatial 
import os 
from pathlib import Path  

class BuildingFeatures: 
    def __init__(self, alg):
        # alg is a string that can be 'Euclidean' or 'KNN' or 'Decision Tree'
        self.alg = alg
    
    # process_alg takes the algorithm input and calls the appropriate method
    def process_alg(self, meter_file_path, sim_job_file_path, sq_ft):
        if self.alg == 'Euclidean':
            self.Euclidean()
            self.format_simdata(meter_file_path, sim_job_file_path, sq_ft)
        elif self.alg == 'KNN':
            self.KNN_classifiers()
            self.format_simdata(meter_file_path, sim_job_file_path, sq_ft)
        elif self.alg == 'Decision Tree':
            self.DT_classifiers()
            self.format_simdata(meter_file_path, sim_job_file_path, sq_ft)
        else: 
            print("Invalid Algorithm Input. Please provide 'Euclidean', 'KNN', or 'Decision Tree.'")
    
    def Euclidean(self):
        print("Calculating Euclidean Distance")
    
    def KNN_classifiers(self): 
        print("Calculating KNN")

    def DT_classifiers(self): 
        print("Calculating the Decision Tree(s)")

    def format_simdata(self, meter_file_path, sim_job_file_path, sq_ft):
        meter_files = glob.glob(os.path.join(meter_file_path, "*.csv"))
        print("Meter files:" , meter_files)
        #load simulation data, transform to "wide" format where each row is a simulation and columns are each hour of the year
        date_str = '12/31/2014'
        start = pd.to_datetime(date_str) - pd.Timedelta(days=364)
        hourly_periods = 8760
        drange = pd.date_range(start, periods=hourly_periods, freq='H')
        df_sim = pd.DataFrame(0., index=np.arange(len(meter_files)), columns=drange.astype(str).tolist())#+['Job_ID'])
        i=0
        names = []
        for f in meter_files:
            name = os.path.basename(f)
            name = os.path.splitext(name)[0]
            names.append(name)
            df = pd.read_csv(f)
            df = pd.DataFrame(df)
            df = df.transpose()
            df = df / 3.6e+6

            # if sq_ft is already accounted for - have the user input 0 for the sq_ft field 
            if sq_ft == 0: 
                df = df 
            else: 
                df = df / sq_ft #secondary SF = 210887, primary = 73959

            df.columns = drange.astype(str)
            df_sim.iloc[i] = df
            i=i+1
            
        df_sim['Job_ID'] = names 
        print(df_sim)    
        simjob = pd.read_csv(sim_job_file_path)
        simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
        print("building params")
        print(simjob.head)
        simjob_cols = list(simjob.columns)
        simjob_cols.remove(simjob_cols[0])
        print(simjob_cols)

        return df_sim, simjob

#testing 
bf = BuildingFeatures('Euclidean')
meter_files = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/subset test run/meter files"
sim_job_file = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/subset test run/SimJobIndex.csv"
sq_ft = 210887
bf.process_alg(meter_files, sim_job_file, sq_ft)
