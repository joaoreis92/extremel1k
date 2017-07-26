import l1k as l 
import l1k_experiments as le
df,sig,genes = l.load_data()
#%%
cv_experiments=[]


#%%
def experiments_cv(list_c,list_cellm_threads):
    cv_experiments=[]
    for cell in list_cell:
        for C in list_c:
            dict_params = {
                    'C':C,
                    'n_threads': n_threads,
                    's':2,
                    'train_cell':[cell],
                    'model': 'liblinear'
                    }
            features,labels = l.features_labels(df,sig,dict_params['train_cell'][0],all_cells=False, dmso=True)
            exp = le.crossval(features,labels,dict_params,n_folds=2)
            cv_experiments.append({**dict_params,**exp})
    return cv_experiments
            

