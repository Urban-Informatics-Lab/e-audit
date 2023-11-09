import pandas as pd
import numpy as np
from datetime import datetime
import glob 
import os
import random
import scipy

#load simulation data, transform to "wide" format where each row is a simulation and columns are each hour of the year
path = "/scratch/groups/risheej/abigail-lauren-Sec3C/S3C_csv"
meter_files = glob.glob(os.path.join(path, "*.csv"))
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

#read in simulation building parameters 
simjob = pd.read_csv("/scratch/groups/risheej/abigail-lauren-Sec3C/sec_3c/Secondary3C/SimJobIndex.csv")
simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
print("building params")
print(simjob.head)

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
euc_dist_test.to_csv("/scratch/users/lexcell/S3C/euc_dist_test_mat.csv", index=False)
euc_dist_test_preds = euc_dist_test[['Job_ID']] #predicted match
euc_dist_test_truth = euc_dist_test[['Job_ID_actual']] #actual job ID
euc_dist_test_preds = euc_dist_test_preds.merge(simjob, on='Job_ID', how='left') #merge with building parameters
euc_dist_test_truth = euc_dist_test_truth.rename(columns={'Job_ID_actual': 'Job_ID'})
euc_dist_test_truth = euc_dist_test_truth.merge(simjob, on='Job_ID', how='left') #merge with building parameters
print("saving test results")
euc_dist_test_preds.to_csv("/scratch/users/lexcell/S3C/euc_dist_test_preds.csv", index=False)
euc_dist_test_truth.to_csv("/scratch/users/lexcell/S3C/euc_dist_test_truth.csv", index=False)
##################################################################################################
#read in actual data - before retrofits were installed, transform to "wide" format where each row is a simulation and columns are each hour of the year
print("loading before data")
df_actual_before = pd.read_csv('/scratch/groups/risheej/abigail-lauren-Sec3C/actual_before_2014_309schools.csv')
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
euc_dist_before.to_csv("/scratch/users/lexcell/S3C/euc_dist_before_mat.csv", index=False)
euc_dist_preds_before = euc_dist_before[["Job_ID"]] #predictions
euc_dist_truth_before = euc_dist_before[["char_prem_id"]] #truth
print("saving before results")
euc_dist_preds_before.to_csv("/scratch/users/lexcell/S3C/euc_dist_preds_before.csv", index=False)
euc_dist_truth_before.to_csv("/scratch/users/lexcell/S3C/euc_dist_truth_before.csv", index=False)

#repeat the same process for post-retrofit installation
#read in actual data - after retrofits were installed
print("loading after data")
df_actual_after = pd.read_csv('/scratch/groups/risheej/abigail-lauren-Sec3C/actual_after_2017_325schools.csv')
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
euc_dist_after.to_csv("/scratch/users/lexcell/S3C/euc_dist_after_mat.csv", index=False)
euc_dist_preds_after = euc_dist_after[["Job_ID"]]
euc_dist_truth_after = euc_dist_after[["char_prem_id"]]
print("saving after results")
euc_dist_preds_after.to_csv("/scratch/users/lexcell/S3C/euc_dist_preds_after.csv", index=False)
euc_dist_truth_after.to_csv("/scratch/users/lexcell/S3C/euc_dist_truth_after.csv", index=False)
