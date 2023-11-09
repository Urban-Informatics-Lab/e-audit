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


# load meter files directory input 
meter_files_path = input('Enter a directory path for the meter files: ')
print("meter directory inputed")
meter_files = glob.glob(os.path.join(meter_files_path, "*.csv"))

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
    df = df / 210887 #secondary SF = 210887, primary = 73959
    df.columns = drange.astype(str)
    df_sim.iloc[i] = df
    i=i+1
df_sim['Job_ID'] = names 
print("simulation data loaded")
print(df_sim.head)


sim_job_file_path = input('Enter a file path for the building simulation: ')
print("sim job file inputed")
if os.path.exists(sim_job_file_path):
    print('The sim job file exists.')

    with open(sim_job_file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

        print("The sim job file has been read.")
else:
    print('The specified sim job file does NOT exist')

simjob = pd.read_csv(sim_job_file_path)
simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
print("building params")
print(simjob.head)
simjob_cols = list(simjob.columns)
simjob_cols.remove(simjob_cols[0])
print(simjob_cols)


#compute Euclidean distance - test set
np.random.seed(1)
ridx = np.random.permutation(np.arange(len(df_sim)))
cidx = int(len(df_sim)*0.8)
train = df_sim.iloc[ridx[0:cidx]] #subset training set (80%)
test = df_sim.iloc[ridx[cidx:]] #subset test set (20%)
train2 = train.iloc[:, :8760] #remove Job_ID column 
train2 = train2.to_numpy() #make sure all data is numeric
test2 = test.iloc[:, :8760] #remove Job_ID column
test2 = test2.to_numpy() #make sure all data is numeric
#calculate euclidean distance between each time series in the training and test sets
print("calculating test matrix")
euc_dist_test = scipy.spatial.distance.cdist(test2,train2,metric = 'euclidean') 
euc_dist_test = pd.DataFrame(euc_dist_test) #resulting df - each row is job from the test set, each column is a job from the training set
print(euc_dist_test)

euc_dist_test.columns = train['Job_ID']
euc_dist_test['Job_ID'] = euc_dist_test.apply(lambda x: x.idxmin(), axis=1) #select minimum distance as the closest match
euc_dist_test['Job_ID_actual'] = test['Job_ID'].tolist()

output_files_path = input('Enter a directory path for the output files: ')
print(output_files_path + "output file path loaded")
 

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

##################################################################################################
#read in actual data - before retrofits were installed, transform to "wide" format where each row is a simulation and columns are each hour of the year
print("loading before data")
df_actual_before_path = input('Enter a file path for the before data: ')
print("before data inputed")
if os.path.exists(df_actual_before_path):
    print('The before data file exists.')

    with open(df_actual_before_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

        print("The before data has been read.")
else:
    print('The before data file does NOT exist')

df_actual_before = pd.read_csv(df_actual_before_path)
df_actual_before_t = pd.DataFrame(0., index=np.arange(309), columns=drange.astype(str).tolist()+['school_id'])
ids = df_actual_before['char_prem_id'].unique()
i=0
for school_id in ids:
    df = df_actual_before[df_actual_before['char_prem_id'] == school_id]
    date_range = df['date_time']
    df = df[['kWh_norm_sf']]
    df = df.transpose()
    df.columns = date_range.astype(str)
    df['school_id'] = school_id
    df_actual_before_t.iloc[i] = df
    i=i+1
#compute Euclidean distance - validation, before
actual2 = df_actual_before_t.iloc[:, :8760] #remove school ID column
actual2 = actual2.to_numpy() #ensure all data is numeric
df_sim2 = df_sim.iloc[:, :8760]
df_sim2 = df_sim2.to_numpy()
#calculate euclidean distance between each simulation and each actual time series
print("calculating before matrix")
euc_dist_before = scipy.spatial.distance.cdist(actual2,df_sim2,metric = 'euclidean') #resulting df - each row is an actual school, each column is a simulation
euc_dist_before = pd.DataFrame(euc_dist_before)
euc_dist_before.columns = df_sim['Job_ID']
euc_dist_before['Job_ID'] = euc_dist_before.apply(lambda x: x.idxmin(), axis=1) #select minimum distance as closest match
euc_dist_before['char_prem_id'] = df_actual_before_t['school_id'].tolist() #join with the actual school ids
# save the euc dist before file 
euc_dist_before_path = "/".join([output_files_path, "euc_dist_test_before_mat.csv"])
euc_dist_before.to_csv(euc_dist_before_path, index=False)
# euc_dist_before.to_csv("/Users/dipashreyasur/Desktop/Classifying code/Euclidean_results_DS/euc_dist_before_mat.csv", index=False)
euc_dist_preds_before = euc_dist_before[["Job_ID"]] #predictions
euc_dist_truth_before = euc_dist_before[["char_prem_id"]] #truth

print("saving before results")
euc_dist_preds_before_path = "/".join([output_files_path, "euc_dist_preds_before.csv"])
euc_dist_preds_before.to_csv(euc_dist_preds_before_path, index=False)
euc_dist_truth_before_path = "/".join([output_files_path, "euc_dist_truth_before.csv"])
euc_dist_truth_before.to_csv(euc_dist_truth_before_path, index=False)

#repeat the same process for post-retrofit installation
#read in actual data - after retrofits were installed
df_actual_after_path = input('Enter a file path for the after data: ')
if os.path.exists(df_actual_after_path):
    print('The after data file exists.')

    with open(df_actual_after_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

        print("The after data has been read.")
else:
    print('The after data file does NOT exist')
df_actual_after = pd.read_csv(df_actual_after_path)
print("loading after data")
date_str = '12/31/2017'  
start = pd.to_datetime(date_str) - pd.Timedelta(days=364)
hourly_periods = 8760
drange = pd.date_range(start, periods=hourly_periods, freq='H')
df_actual_after_t = pd.DataFrame(0., index=np.arange(325), columns=drange.astype(str).tolist()+['school_id'])
ids = df_actual_after['char_prem_id'].unique()
i=0
for school_id in ids:
    df = df_actual_after[df_actual_after['char_prem_id'] == school_id]
    date_range = df['date_time']
    df = df[['kWh_norm_sf']]
    df = df.transpose()
    df.columns = date_range.astype(str)
    df['school_id'] = school_id
    df_actual_after_t.iloc[i] = df
    i=i+1
#compute Euclidean distance - validation, after
actual2 = df_actual_after_t.iloc[:, :8760]
actual2 = actual2.to_numpy()
df_sim2 = df_sim.iloc[:, :8760]
df_sim2 = df_sim2.to_numpy()
print("calculating after matrix")
euc_dist_after = scipy.spatial.distance.cdist(actual2,df_sim2,metric = 'euclidean')
euc_dist_after = pd.DataFrame(euc_dist_after)

euc_dist_after.columns = df_sim['Job_ID']
euc_dist_after['Job_ID'] = euc_dist_after.apply(lambda x: x.idxmin(), axis=1)
euc_dist_after['char_prem_id'] = df_actual_after_t['school_id'].tolist()
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

# for loop for categories 
for col in simjob_cols:
    if col in ['lightsched', 'roof', 'shgc', 'boiler']:
        preds[col] = preds[col].astype('str')
    else:
        preds[col] = preds[col].astype('category')

for col in simjob_cols:
    if col in ['lightsched', 'roof', 'shgc', 'boiler']:
        truth[col] = truth[col].astype('str')
    else:
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
