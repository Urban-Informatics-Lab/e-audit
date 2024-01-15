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

def extract_datetime(date_str):
  # date_str = correct_date_str(date_str)
  date = datetime.strptime(date_str, ' %m/%d  %H:%M:%S')
  return date #.replace(year=2014)

def time_stats(group):
  new_grp = group
  new_grp['date'] = new_grp['Date/Time'] #.apply(extract_datetime)
  new_grp['month'] = new_grp['date'].apply(lambda d: d.month)
  new_grp['week'] = new_grp['date'].apply(lambda d: d.isocalendar()[1])
  new_grp['day'] = new_grp['date'].apply(lambda d: d.timetuple().tm_yday)
  return new_grp

def time_stats_2(group):
  new_grp = group
  new_grp['Date/Time'] = new_grp['Date/Time'].str.replace(' 24:00:00', ' 00:00:00')
  new_grp['date'] = new_grp['Date/Time'].apply(extract_datetime)
  new_grp['month'] = new_grp['date'].apply(lambda d: d.month)
  new_grp['week'] = new_grp['date'].apply(lambda d: d.isocalendar()[1])
  new_grp['day'] = new_grp['date'].apply(lambda d: d.timetuple().tm_yday)
  return new_grp

def feature_grp_meter_file(new_group):
  grp_month = new_group[['Electricity:Facility [J](Hourly)','month']].groupby('month').agg(func = ['mean', 'min','max','median','std'])
  numpy_month = grp_month['Electricity:Facility [J](Hourly)'].to_numpy().flatten()
  grp_year = new_group['Electricity:Facility [J](Hourly)'].agg(func = ['mean', 'min','max','median','std'])
  numpy_year = grp_year.to_numpy().flatten()
  grp_week = new_group[['Electricity:Facility [J](Hourly)','week']].groupby('week').agg(func = ['mean', 'min', 'max', 'median', 'std'])
  numpy_week = grp_week['Electricity:Facility [J](Hourly)'].to_numpy().flatten()
  final_array = np.concatenate((numpy_month, numpy_year, numpy_week))
  return final_array

def feature_grp_meter_dir(new_group):
  grp_month = new_group[['Electricity:Facility [J](Hourly) ','month']].groupby('month').agg(func = ['mean', 'min','max','median','std'])
  numpy_month = grp_month['Electricity:Facility [J](Hourly) '].to_numpy().flatten()
  grp_year = new_group['Electricity:Facility [J](Hourly) '].agg(func = ['mean', 'min','max','median','std'])
  numpy_year = grp_year.to_numpy().flatten()
  grp_week = new_group[['Electricity:Facility [J](Hourly) ','week']].groupby('week').agg(func = ['mean', 'min', 'max', 'median', 'std'])
  numpy_week = grp_week['Electricity:Facility [J](Hourly) '].to_numpy().flatten()
  final_array = np.concatenate((numpy_month, numpy_year, numpy_week))
  return final_array

def time_stats_actual(group1):
    new_grp1 = group1
    new_grp1['date'] = new_grp1['Date.Time'].apply(extract_datetime_actual)
    new_grp1['month'] = new_grp1['date'].apply(lambda d: d.month)
    new_grp1['week'] = new_grp1['date'].apply(lambda d: d.isocalendar()[1])
    new_grp1['day'] = new_grp1['date'].apply(lambda d: d.timetuple().tm_yday)
    return new_grp1

def extract_datetime_actual(date_str1):
    date_str1 = date_str1.replace(' 24:00:00', ' 00:00:00')
    date1 = datetime.strptime(date_str1, ' %m/%d  %H:%M:%S')
    return date1

def feature_grp_actual(new_group1):
  grp_month1 = new_group1[['kWh_norm_sf','month']].groupby('month').agg(func = ['mean', 'min','max','median','std'])
  numpy_month1 = grp_month1['kWh_norm_sf'].to_numpy().flatten()
  grp_year1 = new_group1['kWh_norm_sf'].agg(func = ['mean', 'min','max','median','std'])
  numpy_year1 = grp_year1.to_numpy().flatten()
  grp_week1 = new_group1[['kWh_norm_sf','week']].groupby('week').agg(func = ['mean', 'min', 'max', 'median', 'std'])
  numpy_week1 = grp_week1['kWh_norm_sf'].to_numpy().flatten()
  final_array1 = np.concatenate((numpy_month1, numpy_year1, numpy_week1))
  return final_array1

class BuildingFeatures: 
    def __init__(self, alg):
        # alg is a string that can be 'Euclidean' or 'KNN' or 'Decision Tree'
        self.alg = alg
    
    # process_alg takes the algorithm input and calls the appropriate method
    def process_alg(self, meter_file_path, sim_job_file_path, date_str, output_files_path, actual, sq_ft, J_conversion):
        if self.alg == 'Euclidean':
            df_sim, simjob = self.format_simdata(meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion)
            df_actual_t = self.format_sim_actualdata(actual) 
            self.Euclidean(df_sim, simjob, output_files_path, df_actual_t)

        elif self.alg == 'KNN':
            df_sim, buildingparams, feature_vector, job_id = self.format_MLdata(meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion)
            df_actual_t = self.format_ML_actualdata(actual)
            self.KNN(df_sim, output_files_path, feature_vector, job_id)

        elif self.alg == 'Decision Tree':
            df_sim, buildingparams, feature_vector, job_id = self.format_MLdata(meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion)
            df_actual_t = self.format_ML_actualdata(actual)
            self.DecisionTrees(buildingparams, output_files_path, feature_vector, job_id)

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

    def format_simdata(self, meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion):
        if os.path.isfile(meter_file_path):
            print("meter file inputed")
            start = pd.to_datetime(date_str)
            if start.is_leap_year:
                hourly_periods = 8784 
            else: 
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
            if start.is_leap_year:
                hourly_periods = 8784 
            else: 
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
                    df['Electricity:Facility [J](Hourly)']=df['Electricity:Facility [J](Hourly)']/(3.6e+6) 
                # if sq_ft is already accounted for - have the user input 0 for the sq_ft field 
                if sq_ft == 0: 
                    df = df 
                else: 
                    df['Electricity:Facility [J](Hourly)']=df['Electricity:Facility [J](Hourly)']/ sq_ft #secondary SF = 210887, primary = 73959
                # extract only the electricity data we need 
                df = df['Electricity:Facility [J](Hourly)']
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
    
    def format_sim_actualdata(self, df_actual_path):
        df_actual = pd.read_csv(df_actual_path)
        start = pd.to_datetime(date_str) 
        if start.is_leap_year:
                hourly_periods = 8784 
        else: 
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
        print("Actual data: ")
        print(df_actual_t) 
        return df_actual_t
  
    def format_ML_actualdata(self, df_actual_path):
        df_actual_after = pd.read_csv(df_actual_path)
        print(df_actual_after)
        actual_feat = []
        grouped_actual_after = df_actual_after.groupby('ID')
        for name, group in list(grouped_actual_after): #create time series features
            fin_actual = time_stats_actual(group)
            actual_feat.append(feature_grp_actual(fin_actual))
        actual_feature_after = np.array(actual_feat)
        print("actual data loaded")
        print(df_actual_after.head)
        return df_actual_after, actual_feature_after

    def format_MLdata(self, meter_file_path, sim_job_file_path, date_str, sq_ft, J_conversion):
        if os.path.isfile(meter_file_path):
            print("meter file inputed")
            start = pd.to_datetime(date_str)
            if start.is_leap_year:
                hourly_periods = 8784 
            else: 
                hourly_periods = 8760
            drange = pd.date_range(start, periods=hourly_periods, freq='H')
            df = pd.read_csv(meter_file_path)
            unique_ids = df['Job_ID'].unique()
            df_sim = []
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

            df_sim = df
            print("DF Sim: ")
            print(df_sim)
            #create time series features for each Job ID 
            grouped_id = df_sim.groupby('Job_ID')
            feature_list = []
            job_id = []
            for name, group in list(grouped_id):
                final_group = time_stats_2(group)
                feature_list.append(feature_grp_meter_file(final_group))
                job_id.append(name)
                feature_vector = np.array(feature_list)
            print("simulation data:")
            print(df_sim.head) 
        if os.path.isdir(meter_file_path):
            meter_files = glob.glob(os.path.join(meter_file_path, "*.csv"))
            #load simulation data, transform to "wide" format where each row is a simulation and columns are each hour of the year
            start = pd.to_datetime(date_str)
            if start.is_leap_year:
                hourly_periods = 8784 
            else: 
                hourly_periods = 8760
            drange = pd.date_range(start, periods=hourly_periods, freq='H')
            df_sim = []
            i=0
            names = []
            for f in meter_files:
                # handle the name of the file input 
                name = os.path.basename(f)
                name = os.path.splitext(name)[0] 
                names.append(name)
                df = pd.read_csv(f)
                df['Job_ID'] = name
                df['Date/Time']=drange
                # names.append(name)
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
                df_sim.append(df)
            print("DF SIM: ")
            print(type(df_sim))
            # # assign df_sim to be each of the file names that contains the Job_ID 
            df_sim=pd.concat(df_sim)
            print("DF Sim: ")
            print(df_sim)
            #create time series features for each Job ID 
            grouped_id = df_sim.groupby('Job_ID')
            feature_list = []
            job_id = []
            for name, group in list(grouped_id):
                final_group = time_stats(group)
                feature_list.append(feature_grp_meter_dir(final_group))
                job_id.append(name)
                feature_vector = np.array(feature_list)
            print("simulation data:")
            print(df_sim.head)
        
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

        return df_sim, building_params, feature_vector, job_id
    
    def KNN(self, building_params, output_files_path, feature_vector, job_id):
        # create a new output files path if needed 
        Path(output_files_path).mkdir(parents=True, exist_ok=True)
        # create a new index for merging purposes 
        building_params.index = np.arange(1, len(building_params)+1)
        building_params = building_params.reset_index()
        # kNN classifying
        le = preprocessing.LabelEncoder()
        label=le.fit_transform(job_id)
        # split the data - 80/20 training/testing
        X_train, X_test, y_train, y_test = train_test_split(feature_vector, label,random_state=135,test_size=0.2,shuffle=True)
        # train the model
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X_train,y_train)
        # test on the subset 
        y_predicted = model.predict(X_test)
        preds = pd.DataFrame(y_predicted.T, columns = ['Job_ID'])
        truth = pd.DataFrame(y_test.T, columns = ['Job_ID'])
        preds['Job_ID']=preds['Job_ID'].astype(str)
        building_params['index']=building_params['index'].astype(str)
        truth['Job_ID']=truth['Job_ID'].astype(str)
        #altered the merging code to be on the index 
        preds = pd.merge(preds, building_params, left_on="Job_ID", right_on="index") #merge with actual building parameters to check how close the match was
        truth = pd.merge(truth, building_params, left_on="Job_ID", right_on="index")
        output_test_path = "/".join([output_files_path, "kNN_test_preds.csv"])
        filepath = Path(output_test_path)  
        filepath.parent.mkdir(parents=True, exist_ok=True)  
        preds.to_csv(filepath, index=False)
        print("kNN_test_preds saved")
        test_truth = "/".join([output_files_path, "kNN_test_true.csv"])
        truth.to_csv(test_truth, index=False) 

    def DecisionTrees(self, buildingparams, output_files_path, feature_vector, job_id): 
        # create output file path if needed 
        Path(output_files_path).mkdir(parents=True, exist_ok=True)

        # multiple decision trees
        # split the data - 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(feature_vector, buildingparams,random_state=203,test_size=0.2,shuffle=True)
        print("Building params:")
        print(buildingparams)
        y_test = y_test.reset_index(inplace=False)
        list_features = list(buildingparams.columns)
        list_features.remove('Job_ID') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
        list_features.remove('index') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
        list_features.remove('#') #tree was overfitting to Job ID - each leaf is one ID, making it take too long
        print("List Features:")
        print(list_features) #check that it's only the features we want
        
        #create empty dataframes before for loop
        multi_class_correct = pd.DataFrame(columns=list_features)
        multi_class_test_preds = pd.DataFrame(columns=list_features)
        multi_class_test_preds["ID"] = y_test.Job_ID.unique()
        #create column with actual building IDs
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

            # create separate folder to contain all the features 
            test_preds_path = f"{output_files_path}/multiple_trees_test_preds_features"
            Path(test_preds_path).mkdir(parents=True, exist_ok=True)
            multi_class_test_preds_path = f"{test_preds_path}/test_preds_{feature}.csv"
            multi_class_test_preds[[feature]].to_csv(multi_class_test_preds_path, index=False)

            # create separate folder to contain all the features 
            test_true_path = f"{output_files_path}/multiple_trees_test_true_features"
            Path(test_true_path).mkdir(parents=True, exist_ok=True)
            y_test_path = f"{test_true_path}/test_true_{feature}.csv"
            y_test[[feature]].to_csv(y_test_path, index=False)

            #multiple_tree_preds_after_path = f"{output_files_path}/multiple_trees_validation_preds_after_{feature}.csv"
            #mult_tree_preds_after[[feature]].to_csv(multiple_tree_preds_after_path, index=False)

            #save all of the results (for each feature tree) in one file
            print("saving results...(trees)")

            multi_class_test_preds_path = "/".join([output_files_path, "multiple_trees_test_preds.csv"])
            multi_class_test_preds.to_csv(multi_class_test_preds_path, index=False)
            y_test_preds_path = "/".join([output_files_path, "multiple_trees_test_true.csv"])
            y_test.to_csv(y_test_preds_path, index=False)
            print("trees test results saved!")

            # mult_tree_preds_after["ID"] = df_actual_t.ID.unique()
            # mult_tree_preds_after_path = "/".join([output_files_path, "multiple_trees_validation_preds.csv"])
            # mult_tree_preds_after.to_csv(path_or_buf = mult_tree_preds_after_path, index=False)
            # print("trees validation results saved!")

# #EUC testing 
# meter_files_dir = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example_IndividualFiles"
# # meter_file = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example.csv"
# sim_job = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/SimJobIndex_Example.csv"
# output_files_path = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Euc_Results_Class" 
# actual_data = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Sample Building Electricity Data.csv"

# date_str = "01/01/2014"
# sq_ft = 210887
# J_conversion = 1 
# bf = BuildingFeatures('Euclidean')
# bf.process_alg(meter_files_dir, sim_job, date_str, output_files_path, actual_data, sq_ft, J_conversion)

#KNN + Decision Tree Testing
meter_files_dir = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example_IndividualFiles"
# meter_file = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Meters_Example.csv"
sim_job = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/SimJobIndex_Example.csv"
output_files_path = "/Users/dipashreyasur/Desktop/DT_Meter_Dir" 
actual_data = "/Users/dipashreyasur/Desktop/Autumn 2023/Classifying code/Sample Building Electricity Data.csv"

date_str = "01/01/2014"
sq_ft = 210887
J_conversion = 1 
bf = BuildingFeatures('Decision Tree')
# bf = BuildingFeatures('KNN')
bf.process_alg(meter_files_dir, sim_job, date_str, output_files_path, actual_data, sq_ft, J_conversion)