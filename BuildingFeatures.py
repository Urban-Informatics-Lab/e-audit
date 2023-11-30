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
    def process_alg(self, meter_file_path, sim_job_file_path, date_str, output_files_path, actual, sq_ft, J_conversion):
        if self.alg == 'Euclidean':
            df_sim, simjob = self.format_simdata(meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion)
            df_actual_t = self.format_actualdata(actual) 
            self.Euclidean(df_sim, simjob, output_files_path, df_actual_t)

        elif self.alg == 'KNN':
            self.KNN_classifiers()
            self.format_simdata(meter_file_path, sim_job_file_path, date_str, sq_ft)
        elif self.alg == 'Decision Tree':
            self.DT_classifiers()
            self.format_simdata(meter_file_path, sim_job_file_path, date_str, sq_ft)
        else: 
            print("Invalid Algorithm Input. Please provide 'Euclidean', 'KNN', or 'Decision Tree.'")
    
    def Euclidean(self, df_sim, simjob, output_files_path, df_actual_t):
        #compute Euclidean distance - test set
        print("DF sim loaded: ")
        print(df_sim)
        np.random.seed(1)
        ridx = np.random.permutation(np.arange(len(df_sim)))
        cidx = int(len(df_sim)*0.8)
        train = df_sim.iloc[ridx[0:cidx]] #subset training set (80%)
        test = df_sim.iloc[ridx[cidx:]] #subset test set (20%)
        # print(train['Job_ID'])
        train2 = train.iloc[:, :8760] #remove Job_ID column 
        train2 = train2.to_numpy() #make sure all data is numeric
        test2 = test.iloc[:, :8760] #remove Job_ID column
        test2 = test2.to_numpy() #make sure all data is numeric
        #calculate euclidean distance between each time series in the training and test sets
        print("calculating test matrix")

        euc_dist_test = scipy.spatial.distance.cdist(test2,train2,metric = 'euclidean') 
        euc_dist_test = pd.DataFrame(euc_dist_test) #resulting df - each row is job from the test set, each column is a job from the training set
        euc_dist_test = scipy.spatial.distance.cdist(test2,train2,metric = 'euclidean') 
        euc_dist_test = pd.DataFrame(euc_dist_test) #resulting df - each row is job from the test set, each column is a job from the training set
        print(euc_dist_test)

        euc_dist_test.columns = train['Job_ID']
        euc_dist_test['Job_ID'] = euc_dist_test.apply(lambda x: x.idxmin(), axis=1) #select minimum distance as the closest match
        euc_dist_test['Job_ID_actual'] = test['Job_ID'].tolist()

        output_test_path = "/".join([output_files_path, "euc_dist_test_mat.csv"])
        filepath = Path(output_test_path)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        euc_dist_test.to_csv(filepath, index=False)
        print("euc_dist_test_mat saved")
        
        euc_dist_test_preds = euc_dist_test[['Job_ID']] #predicted match
        euc_dist_test_truth = euc_dist_test[['Job_ID_actual']] #actual job ID
        euc_dist_test_preds = euc_dist_test_preds.merge(simjob, on='Job_ID', how='left') #merge with building parameters
        euc_dist_test_truth = euc_dist_test_truth.rename(columns={'Job_ID_actual': 'Job_ID'})
        euc_dist_test_truth = euc_dist_test_truth.merge(simjob, on='Job_ID', how='left') #merge with building parameters

        output_test_preds_path = "/".join([output_files_path, "euc_dist_test_preds.csv"])
        euc_dist_test_preds.to_csv(output_test_preds_path, index=False)

        output_test_truth_path = "/".join([output_files_path, "euc_dist_test_truth.csv"])
        euc_dist_test_preds.to_csv(output_test_truth_path, index=False)
        print("saved test results")

        #compute Euclidean distance - validation, after
        actual2 = df_actual_t.iloc[:, :8760]
        actual2 = actual2.to_numpy()
        df_sim2 = df_sim.iloc[:, :8760]
        df_sim2 = df_sim2.to_numpy()
        print("calculating after matrix")
        euc_dist_after = scipy.spatial.distance.cdist(actual2,df_sim2,metric = 'euclidean')
        euc_dist_after = pd.DataFrame(euc_dist_after)

        euc_dist_after.columns = df_sim['Job_ID']
        euc_dist_after['Job_ID'] = euc_dist_after.apply(lambda x: x.idxmin(), axis=1)
        euc_dist_after['char_prem_id'] = df_actual_t['school_id'].tolist()
        euc_dist_after_path = "/".join([output_files_path, "euc_dist_test_after_mat.csv"])
        euc_dist_after.to_csv(euc_dist_after_path, index=False)
        euc_dist_preds_after = euc_dist_after[["Job_ID"]]
        euc_dist_truth_after = euc_dist_after[["char_prem_id"]]
        print("saving after results")
        euc_dist_preds_after_path = "/".join([output_files_path, "euc_dist_preds_after.csv"])
        euc_dist_preds_after.to_csv(euc_dist_preds_after_path, index=False)
        euc_dist_truth_after_path = "/".join([output_files_path, "euc_dist_truth_after.csv"])
        euc_dist_truth_after.to_csv(euc_dist_truth_after_path, index=False)

        preds = euc_dist_test_preds
        truth = euc_dist_test_truth
        simjob_cols = simjob_cols = list(simjob.columns)
        simjob_cols.remove(simjob_cols[0])

        # for loop for categories 
        for col in simjob_cols:
            if col in ['lightsched', 'roof', 'shgc', 'boiler']:
                preds[col] = preds[col].astype('str')
                truth[col] = truth[col].astype('str')
            else:
                preds[col] = preds[col].astype('category')
                truth[col] = truth[col].astype('category')

        list_features = simjob_cols 
        class_correct = pd.DataFrame(columns=list_features)
        for feature in list_features:
            preds_str = preds[feature].astype(str)
            truth_str = truth[feature].astype(str)
            class_correct[feature] = (preds_str == truth_str).astype(int)
        correct_rate = class_correct.mean()
        correct_rate_path = "/".join([output_files_path, "test_rate.csv"])
        correct_rate.to_csv(correct_rate_path, index=False)
    
    def KNN_classifiers(self): 
        print("Calculating KNN")

    def DT_classifiers(self): 
        print("Calculating the Decision Tree(s)")

    def format_simdata(self, meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion):
        if os.path.isfile(meter_file_path):
            print("meter file inputed")
            start = pd.to_datetime(date_str)
            hourly_periods = 8760
            drange = pd.date_range(start, periods=hourly_periods, freq='H')
            df = pd.read_csv(meter_file_path)
            unique_ids = df['Job_ID'].unique()
            df_sim = pd.DataFrame(0., index=np.arange(len(unique_ids)), columns=drange.astype(str).tolist())#+['Job_ID'])
            # if J is accounted for - have the user input 0 for the J field 
            if J_conversion == 0: 
                pass
            else:
                # J conversion 
                df['Electricity:Facility [J](Hourly)']=df['Electricity:Facility [J](Hourly)']/(3.6e+6) 
            # if sq_ft is already accounted for - have the user input 0 for the sq_ft field 
            if sq_ft == 0: 
                df = df 
            else: 
                df['Electricity:Facility [J](Hourly)']=df['Electricity:Facility [J](Hourly)']/ sq_ft #secondary SF = 210887, primary = 73959
            # df = df.set_index(['Electricity:Facility [J](Hourly)', 'Job_ID'])
            i = 0 
            # iterate through the unique Job IDs 
            for job_id in unique_ids: 
                # gather rows with the same Job ID 
                df_job = df[df['Job_ID'] == job_id]
                # extract the electricity data 
                df_job = df_job['Electricity:Facility [J](Hourly)']
                # set the columns to be the drange and row data to be the electricity data 
                df_job = df_job.transpose()
                df_job.columns = drange.astype(str)
                # transfer the data from the rows to df_sim 
                df_sim.iloc[i] = df_job 
                i += 1 
            # assign the unique Job_ID column to df_sim 
            df_sim['Job_ID'] = unique_ids

        if os.path.isdir(meter_file_path):
            meter_files = glob.glob(os.path.join(meter_file_path, "*.csv"))
            #load simulation data, transform to "wide" format where each row is a simulation and columns are each hour of the year
            start = pd.to_datetime(date_str)
            hourly_periods = 8760
            drange = pd.date_range(start, periods=hourly_periods, freq='H')
            df_sim = pd.DataFrame(0., index=np.arange(len(meter_files)), columns=drange.astype(str).tolist())#+['Job_ID'])
            i=0
            names = []
            for f in meter_files:
                # handle the name of the file input 
                name = os.path.basename(f)
                name = os.path.splitext(name)[0]
                names.append(name)
                df = pd.read_csv(f)
                if J_conversion == 0: 
                    pass
                else:
                    # J conversion 
                    df['Electricity:Facility [J](Hourly) ']=df['Electricity:Facility [J](Hourly) ']/(3.6e+6) 
                # if sq_ft is already accounted for - have the user input 0 for the sq_ft field 
                if sq_ft == 0: 
                    df = df 
                else: 
                    df['Electricity:Facility [J](Hourly) ']=df['Electricity:Facility [J](Hourly) ']/ sq_ft #secondary SF = 210887, primary = 73959
                # extract only the electricity data we need 
                df = df['Electricity:Facility [J](Hourly) ']
                # transform the data to the "wide" format 
                df = df.transpose()
                df.columns = drange.astype(str)
                df_sim.iloc[i] = df
                i=i+1
            # assign df_sim to be each of the file names that contains the Job_ID 
            df_sim['Job_ID'] = names 
        
        simjob = pd.read_csv(sim_job_file_path)
        simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
        simjob_cols = list(simjob.columns)
        simjob_cols.remove(simjob_cols[0])
        return df_sim, simjob
    
    def format_actualdata(self, df_actual_path):
        df_actual = pd.read_csv(df_actual_path)
        start = pd.to_datetime(date_str) 
        hourly_periods = 8760
        drange = pd.date_range(start, periods=hourly_periods, freq='H')
        # change the index to be the length of the dataframe 
        df_actual_t = pd.DataFrame(0., index=np.arange(309), columns=drange.astype(str).tolist()+['school_id'])
        ids = df_actual['ID'].unique()
        i=0
        for school_id in ids:
            df = df_actual[df_actual['ID'] == school_id]
            date_range = df['Date.Time']
            df = df[['kWh_norm_sf']].transpose()
            df.columns = date_range.astype(str)
            df['school_id'] = school_id
            df_actual_t.iloc[i] = df.iloc[0]
            i=i+1
        return df_actual_t
    
#testing 
meter_files_dir = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example_IndividualFiles"
# meter_file = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example.csv"
sim_job = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/SimJobIndex_Example.csv"
output_files_path = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Euc_Results_Class" 
actual_data = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Sample Building Electricity Data.csv"

date_str = "01/01/2014"
sq_ft = 210887
J_conversion = 1 
bf = BuildingFeatures('Euclidean')
bf.process_alg(meter_files_dir, sim_job, date_str, output_files_path, actual_data, sq_ft, J_conversion)