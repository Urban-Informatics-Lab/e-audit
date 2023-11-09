import pandas as pd
import numpy as np
import scikitplot as skplt
#these strings are used to read and save files for each set of real buildings and methods
school = "Secondary" #Secondary or Primary
sch = "S3C" #P3C
mlmeth = "euc_dist" #or kNN, euc_dist, multiple_trees
fullmeth = "Euclidean Distance"#"Multiple Decision Trees" #Euclidean Distance, k-Nearest Neighbors
preds = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_test_preds.csv".format(sch,mlmeth), sep=",") 
truth = pd.read_csv("/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_test_true.csv".format(sch,mlmeth), sep=",") 

#need to convert features to factors or binary
#it also doesn't like decimals
preds['light'] = preds['@@light@@'].astype('category')
preds['lightsmall'] = preds['@@lightsmall@@'].astype('category')
preds['lightlarge'] = preds['@@lightlarge@@'].astype('category')
preds['lightsched'] = preds['@@lightsched@@'].astype('str')
preds['lightsched'] = preds['lightsched'].astype('category')
preds['roof'] = preds['@@roof@@'].astype('str')
preds['roof'] = preds['roof'].astype('category')
preds['ufac'] = preds['@@ufac@@'].astype('category')
preds['shgc'] = preds['@@shgc@@'].astype('str')
preds['shgc'] = preds['shgc'].astype('category')
preds['infil'] = preds['@@infil@@'].astype('str')
preds['infil'] = preds['infil'].astype('category')
preds['boiler'] = preds['@@boiler@@'].astype('str')
preds['boiler'] = preds['boiler'].astype('category')
preds['tech'] = preds['@@tech@@'].astype('category')
preds['fuel'] = preds['@@fuel@@'].astype('category')
preds['hvac'] = preds['@@hvac@@'].astype('category')
preds['cooling1'] = preds['@@cooling1@@'].astype('category')
preds['heating2'] = preds['@@heating2@@'].astype('category')
preds['equip'] = preds['@@equip@@'].astype('category')

truth['light'] = truth['@@light@@'].astype('category')
truth['lightsmall'] = truth['@@lightsmall@@'].astype('category')
truth['lightlarge'] = truth['@@lightlarge@@'].astype('category')
truth['lightsched'] = truth['@@lightsched@@'].astype('str')
truth['lightsched'] = truth['lightsched'].astype('category')
truth['roof'] = truth['@@roof@@'].astype('str')
truth['roof'] = truth['roof'].astype('category')
truth['ufac'] = truth['@@ufac@@'].astype('category')
truth['shgc'] = truth['@@shgc@@'].astype('str')
truth['shgc'] = truth['shgc'].astype('category')
truth['infil'] = truth['@@infil@@'].astype('str')
truth['infil'] = truth['infil'].astype('category')
truth['boiler'] = truth['@@boiler@@'].astype('str')
truth['boiler'] = truth['boiler'].astype('category')
truth['tech'] = truth['@@tech@@'].astype('category')
truth['fuel'] = truth['@@fuel@@'].astype('category')
truth['hvac'] = truth['@@hvac@@'].astype('category')
truth['cooling1'] = truth['@@cooling1@@'].astype('category')
truth['heating2'] = truth['@@heating2@@'].astype('category')
truth['equip'] = truth['@@equip@@'].astype('category')

#calculate correct classification rate (if it wasn't done in the ML classifiers code) and graph confusion matrix (not used in paper, but still interesting) 
list_features = ['light','lightsmall','lightlarge','lightsched','roof','ufac','shgc','infil','boiler','tech','fuel','hvac','cooling1','heating2','equip'] #had to remove hvac and equip because they have 3 options
class_correct = pd.DataFrame(columns=list_features)
for feature in list_features:
    class_correct[feature] =  np.array(preds[feature] == truth[feature], dtype=int)
    cm = skplt.metrics.plot_confusion_matrix(truth[feature], preds[feature], normalize=False, title = 'Confusion Matrix for {} School {}: {}'.format(school,feature,fullmeth))
    cm.figure.savefig("/Users/laurenexcell/Documents/grad/UIL/Load matching/Visualized results/Test/confusion matrices/{}_test_cm_{}_{}.png".format(school,mlmeth,feature),dpi=300)
correct_rate = class_correct.mean()
correct_rate.to_csv(path_or_buf="/Users/laurenexcell/Documents/grad/UIL/Load matching/Sherlock results/{}/{}_test_rate.csv".format(sch,mlmeth))
