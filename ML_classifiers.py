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
  date1 = datetime.strptime(date_str1, '%Y-%m-%d %H:%M:%S %Z')
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
  new_grp1['date'] = new_grp1['date_time'].apply(extract_datetime_actual)
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

#read in simulated data - columns = Job_ID, Date.Time, kWh_norm_sf
path = "/scratch/groups/risheej/abigail-lauren-Sec3C/P3C_csv" #folder where all jEPlus csv outputs are
meter_files = glob.glob(os.path.join(path, "*.csv"))
date_str = '12/31/2014'
start = pd.to_datetime(date_str) - pd.Timedelta(days=364)
hourly_periods = 8760
drange = pd.date_range(start, periods=hourly_periods, freq='H')
#drange = drange.strftime('%m/%d %H:%M:%S')
df_sim = []
i=0
for f in meter_files:
    name = os.path.basename(f)
    name = os.path.splitext(name)[0]
    df = pd.read_csv(f)
    df['Job_ID']=name
    df['Date.Time']=drange
    df = df.rename({'Electricity:Facility': 'Electricity_kWh'}, axis='columns')
    df['Electricity_kWh']=df['Electricity_kWh']/(3.6e+6) #convert Joules to kWh
    df['kWh_norm_sf']=df['Electricity_kWh']/73959 #normalize by square footage: Primary School Square Footage = 73959, Secondary School Square Footage = 210887
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

#import building features
simjob = pd.read_csv("/scratch/groups/risheej/abigail-lauren-Sec3C/SimJobIndexPrimary.csv")
simjob = simjob.drop(columns = ['WeatherFile','ModelFile',"#"])
simjob_str = simjob.astype(str)
print("simjob_str columns")
print(simjob_str.columns)
building_params = simjob
building_params["Job_ID"] = building_params["Job_ID"].str.slice(5,13) #keep only the numeric part of the job id for easier matching
building_params["Job_ID"] = building_params["Job_ID"].astype('int64')
print("building params")
print(building_params.head)

#read in actual data - this should be normalized by square footage, columns = date_time, char_prem_id, kWh_norm_sf
df_actual_before = pd.read_csv('/scratch/groups/risheej/abigail-lauren-Sec3C/actual_before_2014_309schools.csv') #before retrofits were installed
actual_feat = []
grouped_actual_before = df_actual_before.groupby('char_prem_id')
for name, group in list(grouped_actual_before): #create time series features
  fin_actual = time_stats_actual(group)
  actual_feat.append(feature_grp_actual(fin_actual))
  actual_feature_before = np.array(actual_feat)

df_actual_after = pd.read_csv('/scratch/groups/risheej/abigail-lauren-Sec3C/actual_after_2017_325schools.csv') #after retrofits were installed
actual_feat = []
grouped_actual_after = df_actual_after.groupby('char_prem_id')
for name, group in list(grouped_actual_after): #create time series features
  fin_actual = time_stats_actual(group)
  actual_feat.append(feature_grp_actual(fin_actual))
  actual_feature_after = np.array(actual_feat)
print("actual data loaded")
print(df_actual_after.head)

### kNN classifying
le = preprocessing.LabelEncoder()
label=le.fit_transform(job_id)
## split the data - 80/20 training/testing
X_train, X_test, y_train, y_test = train_test_split(feature_vector, label,random_state=135,test_size=0.2,shuffle=True)
print("kNN y test")
print(y_test)
print("kNN x test")
print(X_test)
## train the model
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train,y_train)
## test on the subset
y_predicted = model.predict(X_test)
print("kNN y predicted")
print(y_predicted)
preds = pd.DataFrame(y_predicted.T, columns = ['Job_ID'])
truth = pd.DataFrame(y_test.T, columns = ['Job_ID'])
preds = pd.merge(preds, building_params, how="left",on="Job_ID") #merge with actual building parameters to check how close the match was
truth = pd.merge(truth, building_params, how="left",on="Job_ID")
print("saving results... (kNN)")
preds.to_csv(path_or_buf="/scratch/users/lexcell/P3C/kNN_test_preds.csv", index=False) #predictions on test set
truth.to_csv(path_or_buf="/scratch/users/lexcell/P3C/kNN_test_true.csv", index=False) #truth from test set
list_features = list(simjob_str.columns)
kNN_class_correct = pd.DataFrame(columns=list_features)
for feature in list_features:
    kNN_class_correct[feature] =  np.array(preds[feature] == truth[feature], dtype=int) #check whether the feature was classified correctly
kNN_rate = kNN_class_correct.mean() #calculate the correct classification rate for each feature
kNN_class_correct.to_csv(path_or_buf="/scratch/users/lexcell/P3C/kNN_test_class_correct.csv", index=False) #binary classifications (1 = correct)
kNN_rate.to_csv(path_or_buf="/scratch/users/lexcell/P3C/kNN_test_rate.csv") #correct classification rate
print("kNN test results saved!")
## predict on actual data and save to csv
kNN_preds_before = pd.DataFrame(columns=["char_prem_id","Job_ID"])
kNN_preds_before["char_prem_id"] = df_actual_before.char_prem_id.unique()
kNN_preds_before["Job_ID"] = model.predict(actual_feature_before)
kNN_preds_before.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/kNN_validation_preds_before.csv",index = False)

kNN_preds_after = pd.DataFrame(columns=["char_prem_id","Job_ID"])
kNN_preds_after["char_prem_id"] = df_actual_after.char_prem_id.unique()
kNN_preds_after["Job_ID"] = model.predict(actual_feature_after)
kNN_preds_after.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/kNN_validation_preds_after.csv",index = False)
print("kNN validation results saved!")


# multiple decision trees
## split the data - 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(feature_vector, simjob_str,random_state=203,test_size=0.2,shuffle=True)
print("trees y test")
print(y_test)
print("trees x test")
print(X_test)
y_test = y_test.reset_index(inplace=False)
list_features = list(simjob_str.columns)
list_features.remove('Job_ID') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
print(list_features) #check that it's only the features we want
#create empty dataframes before for loop
mult_tree_preds_before = pd.DataFrame(columns=list_features)
mult_tree_preds_after = pd.DataFrame(columns=list_features)
multi_class_correct = pd.DataFrame(columns=list_features)
multi_class_test_preds = pd.DataFrame(columns=list_features)
#create column with actual building IDs
mult_tree_preds_before["char_prem_id"] = df_actual_before.char_prem_id.unique()
mult_tree_preds_after["char_prem_id"] = df_actual_after.char_prem_id.unique()
#set hyperparameters for tuning each decision tree
max_depth_range = [4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
sample_split_range = list(range(2, 50))
leaf_range = list(range(1,40))
tree_param = [{'criterion': ['gini'], 'max_depth': max_depth_range, 'splitter': ['random','best']},
              {'min_samples_split': sample_split_range, 'min_samples_leaf': leaf_range}]
#this for loop saves the results after each feature in case running the code times out and the script fails
for feature in list_features:
    print(feature)
    clf = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=2, scoring='accuracy') #hyperparameter tuning 
    clf_feature = clf.fit(X_train, y_train["{}".format(feature)]) #create a decision tree for each building feature
    print(clf_feature.best_estimator_)
    y_predicted = clf_feature.predict(X_test)
    print("y predicted")
    print(feature)
    print(y_predicted)
    multi_class_correct[feature] =  np.array(y_predicted == y_test[feature], dtype=int) #binary classifications (1 = correct)
    multi_class_test_preds[feature] = y_predicted #predictions on test set
    mult_tree_preds_before[feature] = clf_feature.predict(actual_feature_before) #predictions on pre-retrofit data
    mult_tree_preds_after[feature] = clf_feature.predict(actual_feature_after) #predictions on post-retrofit data
    multi_class_correct.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_class_correct_{}.csv".format(feature),index = False)
    multi_class_test_preds.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_preds_{}.csv".format(feature), index = False)
    y_test.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_true_{}.csv".format(feature), index = False)
    multiple_trees_rate = multi_class_correct.mean() #correct classification rate
    multiple_trees_rate.to_csv(path_or_buf="/scratch/users/lexcell/P3C/multiple_trees_test_rate_{}.csv".format(feature))
    mult_tree_preds_before.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_validation_preds_before_{}.csv".format(feature),index = False)
    mult_tree_preds_after.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_validation_preds_after_{}.csv".format(feature),index = False)
#save all of the results (for each feature tree) in one file
print("saving results...(trees)")
multi_class_correct.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_class_correct.csv",index = False)
multi_class_test_preds.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_preds.csv", index = False)
y_test.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_test_true.csv", index = False)
multiple_trees_rate = multi_class_correct.mean()
multiple_trees_rate.to_csv(path_or_buf="/scratch/users/lexcell/P3C/multiple_trees_test_rate.csv")
print("trees test results saved!")
mult_tree_preds_before["char_prem_id"] = df_actual_before.char_prem_id.unique()
mult_tree_preds_before.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_validation_preds_before.csv",index = False)
mult_tree_preds_after["char_prem_id"] = df_actual_after.char_prem_id.unique()
mult_tree_preds_after.to_csv(path_or_buf = "/scratch/users/lexcell/P3C/multiple_trees_validation_preds_after.csv",index = False)
print("trees validation results saved!")
