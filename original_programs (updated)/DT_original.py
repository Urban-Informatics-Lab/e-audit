import pandas as pd
import numpy as np
from datetime import datetime
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn import metrics 
import glob 
import os
import re 
from pathlib import Path  

# the testing in this file tests whether the attribute was classified correctly (0,1) rather than calculating an MSE, because we're selecting between two attribute options
# we can then calculate a correct classification rate for each method, and for each attribute in the mutli-tree method
def correct_date_str(d):
  # '_mm/dd__hh:mm:ss'
  date, _, time = d.strip(' ').split(' ')
  hour, mins, secs = time.split(':')
  hour = '%02d' % (int(hour) - 1)
  return f' {date}  {hour}:{mins}:{secs}'

def extract_datetime(date_str):
  # date_str = correct_date_str(date_str)
  date = datetime.strptime(date_str, ' %m/%d/%Y  %H:%M:%S')
  return date #.replace(year=2014)

def extract_datetime_actual(date_str1):
    date_str1 = date_str1.replace(' 24:00:00', ' 00:00:00')
    date1 = datetime.strptime(date_str1, ' %m/%d  %H:%M:%S')
    return date1


def time_stats(group):
  new_grp = group
  new_grp['date'] = new_grp['Date.Time'] #.apply(extract_datetime)
  new_grp['month'] = new_grp['date'].apply(lambda d: d.month)
  new_grp['week'] = new_grp['date'].apply(lambda d: d.isocalendar()[1])
  new_grp['day'] = new_grp['date'].apply(lambda d: d.timetuple().tm_yday)
  return new_grp
  
def time_stats_actual(group1):
    new_grp1 = group1
    new_grp1['date'] = new_grp1['Date.Time'].apply(extract_datetime_actual)
    new_grp1['month'] = new_grp1['date'].apply(lambda d: d.month)
    new_grp1['week'] = new_grp1['date'].apply(lambda d: d.isocalendar()[1])
    new_grp1['day'] = new_grp1['date'].apply(lambda d: d.timetuple().tm_yday)
    return new_grp1

def feature_grp(new_group):
  grp_month = new_group[['kWh_norm_sf','month']].groupby('month').agg(func = ['mean', 'min','max','median','std'])
  numpy_month = grp_month['kWh_norm_sf'].to_numpy().flatten()
  grp_year = new_group['kWh_norm_sf'].agg(func = ['mean', 'min','max','median','std'])
  numpy_year = grp_year.to_numpy().flatten()
  grp_week = new_group[['kWh_norm_sf','week']].groupby('week').agg(func = ['mean', 'min', 'max', 'median', 'std'])
  numpy_week = grp_week['kWh_norm_sf'].to_numpy().flatten()
  final_array = np.concatenate((numpy_month, numpy_year, numpy_week))
  return final_array

def feature_grp_actual(new_group1):
  grp_month1 = new_group1[['kWh_norm_sf','month']].groupby('month').agg(func = ['mean', 'min','max','median','std'])
  numpy_month1 = grp_month1['kWh_norm_sf'].to_numpy().flatten()
  grp_year1 = new_group1['kWh_norm_sf'].agg(func = ['mean', 'min','max','median','std'])
  numpy_year1 = grp_year1.to_numpy().flatten()
  grp_week1 = new_group1[['kWh_norm_sf','week']].groupby('week').agg(func = ['mean', 'min', 'max', 'median', 'std'])
  numpy_week1 = grp_week1['kWh_norm_sf'].to_numpy().flatten()
  final_array1 = np.concatenate((numpy_month1, numpy_year1, numpy_week1))
  return final_array1

# read in simulated data - columns = Job_ID, Date.Time, kWh_norm_sf
meter_files_path = input('Enter a path for the meter files: ')
   
meter_files = glob.glob(os.path.join(meter_files_path, "*.csv"))

date_str = '12/31/2014'
start = pd.to_datetime(date_str) - pd.Timedelta(days=364)
hourly_periods = 8760
drange = pd.date_range(start, periods=hourly_periods, freq='H')
df_sim = []
i=0

J_kWh_conversion = input('The electricity units are in J in the meter files (T/F): ')
norm_sf_conversion = input('The data is normalized by square footage in the meter files (T/F): ') 
if norm_sf_conversion == 'F': 
        sq_ft = input('Enter the square footage: ')
        sq_ft = int(sq_ft)

for f in meter_files:
    name = os.path.basename(f)
    name = os.path.splitext(name)[0]
    df = pd.read_csv(f)
    df['Job_ID']=name
    df['Date.Time']=drange
    df = df.rename({'Electricity:Facility': 'Electricity_kWh'}, axis='columns')

  # normalize by J or sq footage 
    if J_kWh_conversion == 'T': 
        df['Electricity_kWh']=df['Electricity_kWh']/(3.6e+6) 
    else:
        df['Electricity_kWh']=df['Electricity_kWh']
    
    if norm_sf_conversion == 'F': 
        df['kWh_norm_sf']=df['Electricity_kWh']/sq_ft #normalize by square footage: Primary School Square Footage = 73959, Secondary School Square Footage = 210887
    else:
        df['kWh_norm_sf']=df['Electricity_kWh']
        print(type(df['kWh_norm_sf']))
    df_sim.append(df)
    del df['Electricity_kWh']  

df_sim=pd.concat(df_sim) 

#create time series features for each Job ID 
grouped_id = df_sim.groupby('Job_ID')
feature_list = []
job_id = []
for name, group in list(grouped_id):
  final_group = time_stats(group)
  feature_list.append(feature_grp(final_group))
  job_id.append(name)
  feature_vector = np.array(feature_list)
print("simulation data loaded")
print(df_sim.head)

# get file path for building simulation 
sim_job_file_path = input('Enter a file path for the building simulation: ')
print("sim job file inputed")
if os.path.exists(sim_job_file_path):
    print('The sim job file exists.')

    with open(sim_job_file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()

        print("The sim job file has been read.")
else:
    print('The specified sim job file does NOT exist')
    input('Enter a file path for the building simulation: ')

simjob = pd.read_csv(sim_job_file_path)
simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
simjob_str = simjob.astype(str) 
# simjob_cols = list(simjob.columns)
# simjob_cols.remove(simjob_cols[0])
simjob_str.index = np.arange(1, len(simjob_str)+1)
building_params = simjob_str.reset_index()
print("Sim Job_Str:")
print(simjob_str)
building_params = simjob_str
# create a new index for merging purposes 
building_params.index = np.arange(1, len(building_params)+1)
building_params = building_params.reset_index()

# simjob = pd.read_csv(sim_job_file_path)
# simjob = simjob.drop(columns = ['WeatherFile','ModelFile'])
# simjob_str = simjob.astype(str)
# #create new index for merging purposes 
# simjob_str.index = np.arange(1, len(simjob_str)+1)

# building_params = simjob_str.reset_index()
# print("Sim Job_Str:")
# print(simjob_str)
# building_params = simjob_str
# # create a new index for merging purposes 
# building_params.index = np.arange(1, len(building_params)+1)
# building_params = building_params.reset_index()
# print(building_params)

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
actual_feat = []
grouped_actual_after = df_actual_after.groupby('ID')
for name, group in list(grouped_actual_after): #create time series features
  fin_actual = time_stats_actual(group)
  actual_feat.append(feature_grp_actual(fin_actual))
  actual_feature_after = np.array(actual_feat)
print("actual data loaded")
print(df_actual_after.head)

output_files_path = input('Enter a directory path for the output files: ')
print(output_files_path + "output file path loaded")

# multiple decision trees
# split the data - 80/20 train/test split
print("Feature vector length: ")
print(len(feature_vector))
print(feature_vector)
print("SimJob_Str length:")
print(len(simjob_str))
print(simjob_str)
X_train, X_test, y_train, y_test = train_test_split(feature_vector, building_params,random_state=203,test_size=0.2,shuffle=True)
print("Y_train value counts", y_train.value_counts())
# y_train["#"] = pd.to_numeric(y_train["#"], errors='coerce')
# y_train.dropna(subset=["#"], inplace=True)
# print("Y train #", y_train["#"].unique())


print("trees y test")
print(y_test)
print("trees x test")
print(X_test)
y_test = y_test.reset_index(inplace=False)
list_features = list(building_params.columns)
list_features.remove('Job_ID') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
list_features.remove('index') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
list_features.remove('#') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
print(list_features) #check that it's only the features we want
#create empty dataframes before for loop
mult_tree_preds_before = pd.DataFrame(columns=list_features)
mult_tree_preds_after = pd.DataFrame(columns=list_features)
multi_class_correct = pd.DataFrame(columns=list_features)
multi_class_test_preds = pd.DataFrame(columns=list_features)
#create column with actual building IDs
mult_tree_preds_after["ID"] = df_actual_after.ID.unique()
#set hyperparameters for tuning each decision tree
max_depth_range = [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
sample_split_range = list(range(2, 50))
leaf_range = list(range(1,40))
tree_param = [{'criterion': ['gini'], 'max_depth': max_depth_range, 'splitter': ['random','best'],
               'min_samples_split': sample_split_range, 'min_samples_leaf': leaf_range}]

#this for loop saves the results after each feature in case running the code times out and the script fails
for feature in list_features:
    print(feature)
    print("Y train: ")
    print(y_train["{}".format(feature)].value_counts())
    clf = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=2, scoring='accuracy') #hyperparameter tuning 
    clf_feature = clf.fit(X_train, y_train["{}".format(feature)]) #create a decision tree for each building feature
    print(clf_feature.best_estimator_)
    y_predicted = clf_feature.predict(X_test)
    print("y predicted")
    print(feature)
    print(y_predicted)
    multi_class_test_preds[feature] = y_predicted #predictions on test set
    mult_tree_preds_after[feature] = clf_feature.predict(actual_feature_after) #predictions on post-retrofit data
    
    multi_class_test_preds_path = f"{output_files_path}/multiple_trees_test_preds_{feature}.csv"
    multi_class_test_preds[[feature]].to_csv(multi_class_test_preds_path, index=False)
    
    y_test_path = f"{output_files_path}/multiple_trees_test_true_{feature}.csv"
    y_test[[feature]].to_csv(y_test_path, index=False)

    multiple_tree_preds_before_path = f"{output_files_path}/multiple_trees_validation_preds_before_{feature}.csv"
    mult_tree_preds_before[[feature]].to_csv(multiple_tree_preds_before_path, index=False)

    multiple_tree_preds_after_path = f"{output_files_path}/multiple_trees_validation_preds_after_{feature}.csv"
    mult_tree_preds_after[[feature]].to_csv(multiple_tree_preds_after_path, index=False)
#save all of the results (for each feature tree) in one file
print("saving results...(trees)")

multi_class_test_preds_path = "/".join([output_files_path, "multiple_trees_test_preds.csv"])
multi_class_test_preds.to_csv(multi_class_test_preds_path, index=False)
y_test_preds_path = "/".join([output_files_path, "multiple_trees_test_true.csv"])
y_test.to_csv(y_test_preds_path, index=False)
print("trees test results saved!")

mult_tree_preds_after["ID"] = df_actual_after.ID.unique()
mult_tree_preds_after_path = "/".join([output_files_path, "multiple_trees_validation_preds_after.csv"])
mult_tree_preds_after.to_csv(path_or_buf = mult_tree_preds_after_path, index=False)
print("trees validation results saved!")