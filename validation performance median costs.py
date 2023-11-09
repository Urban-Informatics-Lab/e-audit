import pandas as pd
import numpy as np
import scikitplot as skplt
#these strings are used to read and save files for each set of real buildings and methods
schools = ["Secondary","Primary"]
meths = ["kNN","multiple_trees","euc_dist"]
scenarios = ["All","In Agreement","Not In Agreement","Profitable","Non-profitable"]

all_recs = pd.DataFrame(data=None)
all_installs = pd.DataFrame(data=None)
for school in schools:
    if school == "Secondary":
        sch = "S3C"
    else:
        sch = "P3C"
    for mlmeth in meths:
        for cost_est in scenarios:
            preds_before = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_validation_preds_before.csv".format(sch,mlmeth), dtype={"char_prem_id" : "int64"}) 
            preds_after = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_validation_preds_after.csv".format(sch,mlmeth), dtype={"char_prem_id" : "int64"}) 
            if mlmeth == "euc_dist":
                truth_before = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_truth_before.csv".format(sch,mlmeth), dtype={"char_prem_id" : "int64"})
                truth_after = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_truth_after.csv".format(sch,mlmeth), dtype={"char_prem_id" : "int64"})  
                preds_before = preds_before.join(truth_before)
                preds_after = preds_after.join(truth_after)
                preds_before["Job_ID"] = preds_before["Job_ID"].str.slice(5,13)
                preds_before["Job_ID"] = preds_before["Job_ID"].astype('int64')
                preds_after["Job_ID"] = preds_after["Job_ID"].str.slice(5,13)
                preds_after["Job_ID"] = preds_after["Job_ID"].astype('int64')
            seda = pd.read_csv("seda_CA.csv") #these are the school metadata that tell us whether they are elementary, middle, or high schools
            seda_primary = seda[seda['level'] == 'Elementary']
            seda_secondary = seda[(seda['level'] == 'Middle')|(seda['level'] =='High')]
            #subset the predictions, since the models were trained on either a primary or secondary school simulation and we want to make sure we're comparing apples to apples
            if school == 'Primary':
                preds_before = preds_before[preds_before['char_prem_id'].isin(seda_primary['char_prem_id'])]
                preds_after = preds_after[preds_after['char_prem_id'].isin(seda_primary['char_prem_id'])]
            if school == 'Secondary':
                preds_before = preds_before[preds_before['char_prem_id'].isin(seda_secondary['char_prem_id'])]
                preds_after = preds_after[preds_after['char_prem_id'].isin(seda_secondary['char_prem_id'])]

            ## join with building parameters for kNN and Euclidean ONLY, since the building parameters were explicitly predicted for multiple trees
            if mlmeth == "kNN":
                building_params = pd.read_csv("SimJobIndex{}.csv".format(school))
                building_params["Job_ID"] = building_params["Job_ID"].str.slice(5,13)
                building_params["Job_ID"] = building_params["Job_ID"].astype('int64')
                preds_before = pd.merge(preds_before, building_params, how="left",on="Job_ID")
                preds_after = pd.merge(preds_after, building_params, how="left",on="Job_ID")
            if mlmeth == "euc_dist":
                building_params = pd.read_csv("SimJobIndex{}.csv".format(school))
                building_params["Job_ID"] = building_params["Job_ID"].str.slice(5,13)
                building_params["Job_ID"] = building_params["Job_ID"].astype('int64')
                preds_before = pd.merge(preds_before, building_params, how="left",on="Job_ID")
                preds_after = pd.merge(preds_after, building_params, how="left",on="Job_ID")

            #define which retrofits we think were installed based on the buildign parameters that were predicted - these correspond to the actual buildings' retrofit categories
            #columns will be binary - 1 = retrofit, 0 = no retrofit
            preds_before['EEM HVAC'] = np.array(preds_before['@@fuel@@']=="NATURALGAS", dtype=int)
            preds_before['EEM HVAC Controls'] = np.array(preds_before['@@hvac@@']=="HVACOn", dtype=int)
            preds_before['EEM Boiler'] = np.array((preds_before['@@fuel@@']=="NATURALGAS")|(preds_before["@@boiler@@"] == 0.6), dtype=int)
            preds_before['EEM Lighting'] = np.array((preds_before['@@light@@']==21)|(preds_before['@@lightsmall@@']==16)|(preds_before['@@lightlarge@@']==23), dtype=int) 
            preds_before['EEM Lighting Controls'] = np.array(preds_before['@@lightsched@@']==0.9, dtype = int)
            preds_before['EEM Window'] = np.array((preds_before['@@ufac@@']==2)|(preds_before['@@shgc@@']==0.6), dtype=int)
            preds_before['EEM Envelope'] = np.array((preds_before['@@roof@@']==0.07)|(preds_before['@@infil@@']==0.0015), dtype = int)
            preds_before['EEM Plug loads'] = np.array(preds_before['@@tech@@']==20, dtype = int)
            preds_before['EEM Equipment Schedule'] = np.array(preds_before['@@equip@@'] == 'BLDG_EQUIP_SCH_ALL', dtype = int)
            preds_before['EEM HVAC Setpoints'] = np.array((preds_before['@@cooling1@@']==22)|(preds_before['@@heating2@@']==22), dtype =int)

            preds_after['EEM HVAC'] = np.array(preds_after['@@fuel@@']=="NATURALGAS", dtype=int)
            preds_after['EEM HVAC Controls'] = np.array(preds_after['@@hvac@@']=="HVACOn", dtype=int)
            preds_after['EEM Boiler'] = np.array((preds_after['@@fuel@@']=="NATURALGAS")|(preds_after["@@boiler@@"] == 0.6), dtype=int)
            preds_after['EEM Lighting'] = np.array((preds_after['@@light@@']==21)|(preds_after['@@lightsmall@@']==16)|(preds_after['@@lightlarge@@']==23), dtype=int) 
            preds_after['EEM Lighting Controls'] = np.array(preds_after['@@lightsched@@']==0.9, dtype = int)
            preds_after['EEM Window'] = np.array((preds_after['@@ufac@@']==2)|(preds_after['@@shgc@@']==0.6), dtype=int)
            preds_after['EEM Envelope'] = np.array((preds_after['@@roof@@']==0.07)|(preds_after['@@infil@@']==0.0015), dtype = int)
            preds_after['EEM Plug loads'] = np.array(preds_after['@@tech@@']==20, dtype = int)
            preds_after['EEM Equipment Schedule'] = np.array(preds_after['@@equip@@'] == 'BLDG_EQUIP_SCH_ALL', dtype = int)
            preds_after['EEM HVAC Setpoints'] = np.array((preds_after['@@cooling1@@']==22)|(preds_after['@@heating2@@']==22), dtype =int)

            # need a Site SIR greater than 1.01
            # Site SIR = sum(EEM NPV)/sum(EEM Measure costs)
            #multiply binary columns by the NPV and costs to calculate the SIR of our predictions
            # Do we see how many cheaper EEMs we can add while keeping the SIR > 1.01 or do we only recommend our predictions if the Site SIR > 1.01; second option is only eliminating predictions that arent economical, not adding predictions that are economical but unnecesary 
            # meaning even though the lighting is efficient, if we can add it and the SIR > 1.01 then we recommend anyways? 
            # all of the average costs result in SIR > 1.01, so there will never be a case where our predictions have SIR < 1 and adding retrofits will always help
            # using the median SIRs makes it more realistic

            # Calculate Estimated SIR of our predictions
            EEM_costs = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/EEM_median_measure_cost.csv")
            EEM_NPV = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/EEM_median_NPV.csv")
            list_of_eems = ['EEM HVAC', 'EEM HVAC Controls', 'EEM Boiler', 'EEM Lighting', 'EEM Lighting Controls', 'EEM Window', 'EEM Envelope', 'EEM Plug loads', 'EEM Equipment Schedule', 'EEM HVAC Setpoints']
            preds_before['Site NPV'] = 0
            preds_before['Site Cost'] = 0
            for eem in list_of_eems:
                preds_before['Site NPV'] += preds_before[eem] * EEM_NPV[eem][0]
                preds_before['Site Cost'] += preds_before[eem] * EEM_costs[eem][0]
            preds_before['Site SIR'] = preds_before['Site NPV'] / preds_before['Site Cost']
            preds_before['Addtl_EEMs'] = ""
            for i in preds_before.index:
                if preds_before['Site SIR'][i] < 1.01: #change lowest cost EEM predictions to 1 
                    preds_before.loc[i, 'Addtl_EEMs'] = "Recommended"
                    for eem in list_of_eems:
                        if preds_before[eem][i] == 0 and EEM_costs[eem][0] < 10000: #some projects still have SIR < 1.01
                            preds_before.loc[i, 'Addtl_EEMs'] += ", " + eem     
                else:
                    preds_before.loc[i, 'Addtl_EEMs'] = "None"


            # Compare with actual SIR of projects
            peps = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/PEPS_approved.csv", low_memory=False)
            peps = peps[["Site CDS Code","Site Savings to Investment Ratio"]]
            IDkey = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/AgreementID.csv")
            IDkey = IDkey[["Site CDS Code", "char_prem_id"]]
            peps = pd.merge(peps, IDkey, how="left", on="Site CDS Code")
            preds_before = pd.merge(preds_before, peps, how = "left", on= "char_prem_id")
            preds_before['Cost Estimate'] = ""
            for i in preds_before.index:
                if preds_before['Site SIR'][i] >= 1.01 and preds_before['Site Savings to Investment Ratio'][i] >= 1.01:
                    preds_before.loc[i, "Cost Estimate"] = "In Agreement"
                else:
                    if preds_before['Site SIR'][i] < 1.01 and preds_before['Site Savings to Investment Ratio'][i] < 1.01:
                        preds_before.loc[i,"Cost Estimate"] = "In Agreement"
                    else:
                        preds_before.loc[i,"Cost Estimate"] = "Not In Agreement"
            preds_before.to_csv(path_or_buf="/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/Costs/predicted_costs_{}_{}.csv".format(sch, mlmeth)) 

            if cost_est == "Profitable":
                preds_before = preds_before.loc[preds_before['Site SIR'] >= 1.01] #subset projects based on whether the predicted SIR is profitable
            else:
                if cost_est == "All":
                    preds_before = preds_before
                else: 
                    if cost_est == "Non-profitable":
                        preds_before = preds_before.loc[preds_before['Site SIR'] < 1.01]
                    else:
                        preds_before = preds_before.loc[preds_before['Cost Estimate'] == "{}".format(cost_est)] #subset projects based on whether our predicted SIR and actual SIR are in agreement

            #using the before data, compare our recommended EEM to what was actually approved for the project
            actual_installed = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/EEM_actual_simple.csv", dtype={"char_prem_id" : "int64"}) #read in a csv of what retrofits were installed in each school (binary for each category listed above)
            correct_EEM_recs = pd.merge(preds_before, actual_installed, how="left", on="char_prem_id")
            #take the difference between the predictions and the actual installations 
            correct_EEM_recs_rate = pd.DataFrame(columns=list_of_eems)
            for eem in list_of_eems:
                correct_EEM_recs["{} rate".format(eem)] = correct_EEM_recs["{}_x".format(eem)] - correct_EEM_recs["{}_y".format(eem)] #predicted minus actual
                correct_EEM_recs_rate.loc[1,eem] = correct_EEM_recs["{} rate".format(eem)].mean() #closer to 0 means we did better, positive means we recommended more than reality
            correct_EEM_recs_rate = correct_EEM_recs_rate.T
            correct_EEM_recs.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_{}_{}_correct_EEM_recs.csv".format(sch,sch,mlmeth,cost_est)) #if 1, we thought it did but it didn't, if -1 then we missed it. if 0 then we agree
            correct_EEM_recs_rate.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Visualized results/Validation/rates/{}_{}_{}_correct_EEM_recs_rate.csv".format(sch,mlmeth,cost_est))
            correct_EEM_recs_rate.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/Classification rates/{}_{}_{}_correct_EEM_recs_rate.csv".format(sch,mlmeth,cost_est))
            correct_EEM_recs_rate = correct_EEM_recs_rate.T
            correct_EEM_recs_rate['School'] = school
            correct_EEM_recs_rate['Method'] = mlmeth
            correct_EEM_recs_rate['Cost Scenario'] = cost_est
            all_recs = pd.concat([all_recs, correct_EEM_recs_rate])
            #using both the before and after data, compare our predicted EEM installation to what was actually installed
            correct_EEM_installs = pd.merge(preds_before, preds_after, how="left", on="char_prem_id")
            #take the difference between before and after to see what we predict was installed 
            for eem in list_of_eems:
                correct_EEM_installs["{} install".format(eem)] = correct_EEM_installs["{}_x".format(eem)] - correct_EEM_installs["{}_y".format(eem)] #before minus after
                # if 1, then EEM likely installed. if 0, EEM not installed. if -1, that feature became less efficient
                for i in correct_EEM_installs.index: #change -1 case to 0 (no eem installed), but note that performance worsened
                    if correct_EEM_installs.loc[i,"{} install".format(eem)] == -1:
                        correct_EEM_installs.loc[i,"{} performance worsened".format(eem)] = 1
                        correct_EEM_installs.loc[i,"{} install".format(eem)] = 0
                    else:
                        correct_EEM_installs.loc[i,"{} performance worsened".format(eem)] = 0
            #take the difference between actual installations and predicted installations
            correct_EEM_installs = pd.merge(correct_EEM_installs, actual_installed, how="left", on="char_prem_id")
            correct_EEM_installs_rate = pd.DataFrame(columns=list_of_eems)
            for eem in list_of_eems:
                correct_EEM_installs["{} rate".format(eem)] = correct_EEM_installs["{} install".format(eem)] - correct_EEM_installs[eem] #predicted minus actual: 
                for i in correct_EEM_installs.index: #if 1, we overpredicted (thought they would but didnt). if 0, we got it correct. if -1, we underpredicted (we missed it)
                    if correct_EEM_installs.loc[i, eem] == 1 and correct_EEM_installs.loc[i,"{} performance worsened".format(eem)] == 1:
                        correct_EEM_installs.loc[i, 'Operational issue {}'.format(eem)] = 1
                    else:
                        correct_EEM_installs.loc[i, 'Operational issue {}'.format(eem)] = 0
                    if correct_EEM_installs.loc[i, "{} rate".format(eem)] == -1 and correct_EEM_installs.loc[i,"{} performance worsened".format(eem)] == 1:
                        correct_EEM_installs.loc[i, 'Unnecessary, Operational issue {}'.format(eem)] = 1
                    else:
                        correct_EEM_installs.loc[i, 'Unnecessary, Operational issue {}'.format(eem)] = 0
                correct_EEM_installs_rate.loc[1,eem] = correct_EEM_installs["{} rate".format(eem)].mean() 
                correct_EEM_installs_rate.loc[1,"{} operational issues (%)".format(eem)] = correct_EEM_installs["Operational issue {}".format(eem)].sum()/len(correct_EEM_installs)*100
                correct_EEM_installs_rate.loc[1,"{} unnecessary w operational issues (%)".format(eem)] = correct_EEM_installs["Unnecessary, Operational issue {}".format(eem)].sum()/len(correct_EEM_installs)*100
            correct_EEM_installs = correct_EEM_installs.T
            correct_EEM_installs_rate = correct_EEM_installs_rate.T
            correct_EEM_installs.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_{}_{}_correct_EEM_installs.csv".format(sch,sch,mlmeth,cost_est))
            correct_EEM_installs_rate.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Visualized results/Validation/rates/{}_{}_{}_correct_EEM_installs_rate.csv".format(sch,mlmeth,cost_est))
            correct_EEM_installs_rate.to_csv(path_or_buf = "/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/Classification rates/{}_{}_{}_correct_EEM_installs_rate.csv".format(sch,mlmeth,cost_est))
            correct_EEM_installs_rate = correct_EEM_installs_rate.T
            correct_EEM_installs_rate['School'] = school
            correct_EEM_installs_rate['Method'] = mlmeth
            correct_EEM_installs_rate['Cost Scenario'] = cost_est
            all_installs = pd.concat([all_installs, correct_EEM_installs_rate])

            # #graph confusion matrix - we don't end up using these for visualizations, but they might be interesting to have
            # # for EEM in list_EEMs:
            # #     cm = skplt.metrics.plot_confusion_matrix(actual_installed[EEM], pred_installed[EEM], normalize=False, title = 'Confusion Matrix for {} School {} Installations: {}'.format(school,EEM,fullmeth))
            # #     cm.figure.savefig("/Users/laurenexcell/Documents/grad/UIL/jEPlus files/Visualized results/Validation/confusion matrices/{}_installs_cm_{}_{}.png".format(school,EEM, mlmeth),dpi=300)
            # #graph confusion matrix - we don't end up using these for visualizations, but they might be interesting to have
            # # list_EEMs = ['EEM HVAC','EEM HVAC Controls','EEM Boiler','EEM Lighting','EEM Lighting Controls','EEM Window','EEM Envelope','EEM Plug loads','EEM Equipment Schedule','EEM HVAC Setpoints']
            # # for EEM in list_EEMs:
            # #     cm = skplt.metrics.plot_confusion_matrix(actual_installed[EEM], preds_before[EEM], normalize=False, title = 'Confusion Matrix for {} School {} Recommendations: {}'.format(school,EEM,fullmeth))
            # #     cm.figure.savefig("/Users/laurenexcell/Documents/grad/UIL/jEPlus files/Visualized results/Validation/confusion matrices/{}_recs_cm_{}_{}.png".format(school,EEM, mlmeth),dpi=300)

all_recs.to_csv(path_or_buf="/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/Classification rates/all_recs.csv")
all_installs.to_csv(path_or_buf="/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/Classification rates/all_installs.csv")