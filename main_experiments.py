import l1k as l 
import l1k_experiments as le
import pandas as pd
import itertools as it
import numpy as np
import random
from sklearn.datasets import make_classification
from sklearn.datasets import load_svmlight_file
df,sig,genes = l.load_data()

#dd = [{'model': ['vw'],'all_cells':[True]}, {'model':['liblinear'],'C':[0.01,0.1,1,10],'n_threads':[32]}]
#%%
dict_params = {
    'C':1,
    'n_threads': 8,
    's':1,
    'train_cell':'aloi',
    'test_cell':'PC3',
    'loss_function':'squared',
    'passes':3,
    'iter_pd':50,
    'learning_rate':0.3,
    'model': 'vw',
    'holdout_off': True,
    'lambda':0.1,
    'current_fold':0,
    'all_cells':False
    }
#%%
#dict_experiments = [{'train_cell':['aloi'],'model':['vw'],'passes':[1,2,5,10],'learning_rate':[0.01,0.1,0.5,1,2]},{'train_cell':['aloi'],'model':['liblinear'],'C':[0.01,0.1,1,10,100]},{'train_cell':['aloi'],'model':['pdsparse'],'C':[0.01,0.1,1,10,100],'lambda':[0.01,0.1,1,10]}]
dict_experiments = [{'train_cell':['aloi'],'model':['pdsparse'],'C':[0.01,0.1,1,10,100],'lambda':[0.01,0.1,1,10]}]
def aloi():
    features,labels = load_svmlight_file('data_experiments/aloi')
    features = features.todense()
    cv_experiments = []
    exp = le.crossval(features,labels,dict_params,n_folds=10)
    cv_experiments.append({**dict_params,**exp})  
    return features,labels

#%%
def cv_artificial(n_samples=1000,n_classes=111,n_features=978,n_informative=900):
    cv_experiments = []
    features,labels = make_classification(n_samples=n_samples,n_classes=n_classes,n_features=n_features,n_informative=n_informative)
    exp = le.crossval(features,labels,dict_params,n_folds=10)
    cv_experiments.append({**dict_params,**exp})    
    return features,labels
#%%
def main_cv():
    cv_experiments = []
    if dict_params['train_cell'] == 'artificial':
        features,labels = make_classification(n_samples=1000,n_classes=111,n_features=978,n_informative=100)
    else:
        features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=False, dmso=True)
    exp = le.crossval(features,labels,dict_params,n_folds=10)
    cv_experiments.append({**dict_params,**exp})

#%%
def main_test():
	features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=False, dmso=True)
	test_features,test_labels = l.features_labels(df,sig,dict_params['test_cell'],all_cells=False, dmso=True)
	stats,cm = le.test_model(features,labels,test_features,test_labels,dict_params)

#%%
def experiments_cv(dict_experiments,randomized=0):
    cv_experiments = []
    ctrl = True
    list_dict_params = le.grid_search(dict_experiments)
    if randomized:
        list_dict_params = random.sample(list_dict_params,randomized)
    for dict_params in list_dict_params:
        print(pd.DataFrame(dict_params,index=[0])) 
        features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=dict_params['all_cells'], dmso=True)
        exp = le.crossval(features,labels,dict_params,n_folds=10)
        with open('libsvm_experiments.csv','a') as f:
            results = {**dict_params,**exp}
            if ctrl is True:
                pd.DataFrame(results,index=[0]).to_csv(f, header=True)
                ctrl = False
            pd.DataFrame(results,index=[0]).to_csv(f, header=False)
        cv_experiments.append({**dict_params,**exp})
    return pd.DataFrame(cv_experiments)

#%%
def test_best_estimators(df_max,test_cells):
    test_experiments = []
    list_dict_params = list(df_max.T.to_dict())
    for dict_params in df_max.T.to_dict().values():
        dict_params['holdout_off']=True
        for test_cell in test_cells:
            dict_params['test_cell']=test_cell
            features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=False, dmso=True)
            test_features,test_labels = l.features_labels(df,sig,dict_params['test_cell'],all_cells=False, dmso=True)
            if len(set(labels)) >= len(set(test_labels)):
                 ts,cm = le.test_model(features,labels,test_features,test_labels,dict_params)
                 test_experiments.append({**dict_params,**ts})
    return test_experiments
                 
            





