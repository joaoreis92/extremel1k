import l1k as l 
import l1k_experiments as le
import pandas as pd
import itertools as it
df,sig,genes = l.load_data()


#%%
dict_params = {
        'C':1,
        'n_threads': 8,
        's':1,
        'train_cell':'A375',
        'test_cell':['SKBR3'],
        'loss_function':'squared',
        'passes':3,
        'learning_rate':0.3,
        'model': 'vw',
        'holdout_off': False,
		'all_cels':False
        }
#%%
list_c=[0.01,0.1,1,10,100]
list_cell = ['BT20','A549','A375']

#%%
def main_cv():
	features,labels = l.features_labels(df,sig,dict_params['train_cell'][0],all_cells=False, dmso=True)
	exp = le.crossval(features,labels,dict_params,n_folds=5)
	cv_experiments.append({**dict_params,**exp})

#%%
def main_test():
	features,labels = l.features_labels(df,sig,dict_params['train_cell'][0],all_cells=False, dmso=True)
	test_features,test_labels = l.features_labels(df,sig,dict_params['test_cell'][0],all_cells=False, dmso=True)
	stats,cm = le.test_model(features,labels,test_features,test_labels,dict_params)

#%%
def experiments_cv(dict_experiments):
    for dict_params in le.grid_search(dict_experiments):
        print(pd.DataFrame(dict_params,index=[0])) 
        features,labels = l.features_labels(df,sig,dict_params['train_cell'],all_cells=dict_params['all_cells'], dmso=True)
        exp = le.crossval(features,labels,dict_params,n_folds=3)
        with open('libsvm_experiments.csv','a') as f:
            results = {**dict_params,**exp}
            pd.DataFrame(results,index=[0]).to_csv(f, header=False)
        cv_experiments.append({**dict_params,**exp})
    return pd.DataFrame(cv_experiments)



    
    

