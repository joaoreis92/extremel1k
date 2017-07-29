import l1k as l 
import l1k_experiments as le
import pandas as pd
import itertools as it
import numpy as np
import random
df,sig,genes = l.load_data()

#dd = [{'model': ['vw'],'all_cells':[True]}, {'model':['liblinear'],'C':[0.01,0.1,1,10],'n_threads':[32]}]
#%%
dict_params = {
        'C':1,
        'n_threads': 8,
        's':1,
        'train_cell':'A549',
        'test_cell':'SKBR3',
        'loss_function':'squared',
        'passes':2,
        'learning_rate':0.1,
        'model': 'vw',
        'holdout_off': False,
		'all_cells':False
        }
#%%
list_c=[0.01,0.1,1,10,100]
list_cell = ['BT20','A549','A375']

#%%
def main_cv():
	features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=False, dmso=True)
	exp = le.crossval(features,labels,dict_params,n_folds=10)
	cv_experiments.append({**dict_params,**exp})

#%%
def main_test():
	features,labels = l.features_labels(df,sig,dict_params['train_cell'][0],all_cells=False, dmso=True)
	test_features,test_labels = l.features_labels(df,sig,dict_params['test_cell'][0],all_cells=False, dmso=True)
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

dd = [{'model':['liblinear'],'C':[0.001,0.1,1,10,100],'train_cell':['A375']},{'model':['vw'],'learning_rate':np.linspace(0.001,3).tolist(),'passes':[int(x) for x in np.linspace(1,20,20).tolist()],'train_cell':['A375']}]    

complete_liblinear = [{'model':['liblinear'],'C':[0.001,0.1,1,10,100],'train_cell':['BT20','A549','A375']}]
liblinear_all = complete_liblinear = [{'model':['liblinear'],'C':[0.001,0.1,1,10,100],'all_cells':[True]}]   
complete_liblinear_all =  complete_liblinear  + liblinear_all

complete_vw = [{'model':['vw'],'learning_rate':np.linspace(0.001,3).tolist(),'passes':[int(x) for x in np.linspace(1,20,20).tolist()],'train_cell':['BT20','A549','A375']}]
vw_all = complete_liblinear = [{'model':['vw'],'learning_rate':np.linspace(0.001,3).tolist(),'passes':[int(x) for x in np.linspace(1,20,20).tolist()],'all_cells':[True]}]   
complete_vwr_all =  complete_vw  + vw_all
a549_vw = [{'model':['vw'],'learning_rate':np.linspace(0.001,3).tolist(),'passes':[int(x) for x in np.linspace(1,20,20).tolist()],'train_cell':['A549']}]
    
a375_pdsparse=[{'model':['pdsparse'],'train_cells':['A375']}]
all_pdsparse=[{'model':['pdsparse'],'all_cells':[True]}]
