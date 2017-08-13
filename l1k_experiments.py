import os
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support as metrics
import time
sys.path.insert(0,os.path.join(os.path.dirname(__file__), "liblinear-multicore-2.11-2/python/")) # Path to LIBLINEAR
import liblinearutil 
import pandas as pd
import l1k as l
import os
import subprocess
from sklearn.datasets import dump_svmlight_file
import numpy as np
import itertools as it
import re
import random
#%%DIRS
data_dir = 'data_experiments/'
vowpal_dir = 'vowpal_wabbit/vowpalwabbit/'
pdsparse_dir = 'ExtremeMulticlass/'
model_dir = 'model_experiments/'

#%% UTILS
def grid_search(list_dict):
    list_dict_params=[]

    for comb_dict in list_dict:
        params_list = create_permutations(comb_dict)
        for params in params_list:
            params = list(params)
            candidate_dict = {
            'C':1,
            'n_threads': 8,
            's':1,
            'train_cell':'A375',
            'test_cell':'SKBR3',
            'loss_function':'squared',
            'passes':3,
            'iter_pd':50,
            'learning_rate':0.3,
            'model': 'vw',
            'holdout_off': True,
            'lambda':0.1,
            'all_cells':False
            }
            for param in params:
                candidate_dict[param[0]]=param[1]
            list_dict_params.append(candidate_dict)
    return list_dict_params
        
    
    
    
#%%

def create_tuples(param,list_values):
    return [(param, x) for x in list_values]

def create_dict_combs(dict_combs):
    new_dict={}
    i = 1    
    for key in dict_combs:
        new_dict[i]= create_tuples(key,dict_combs[key])
        i = i + 1
    return new_dict
#%%
def create_permutations(my_dict):
    my_dict = create_dict_combs(my_dict)
    allNames = sorted(my_dict)
    combinations = it.product(*(my_dict[Name] for Name in allNames))
    return list(combinations)#
	
def get_data_filename(dict_params,file_type):
    return data_dir+'{0}_{1}_CV{3}_allcells_{4}.{2}'.format('data',dict_params['train_cell'],file_type,dict_params['current_fold'],dict_params['all_cells'])

def get_model_filename(dict_params):
    return model_dir+'{0}_{1}_model'.format(dict_params['model'],dict_params['train_cell'])

def write_files(features,labels,dict_params,data_dir,file_type='train'):
    print('Writing file...')
    start = time.time()
    
    filename = get_data_filename(dict_params,file_type)
    if os.path.isfile(filename) is False:
        if dict_params['model'] in ['vw','pdsparse']:
            dump_svmlight_file(features,labels,filename,zero_based=False)
        
    end = time.time()
    print('File written in ' + str(end - start) + ' seconds')    
    return filename

def run_subproc(run_command,break_word):
    cp = subprocess.run([run_command],
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True)
    print(cp.stdout)
    return cp

        
    
    

#%% Running
def train_model(features,labels,dict_params):
    print('Training model...')
    start = time.time()
    if dict_params['model'] == 'liblinear':
        model = train_liblinear(features,labels,dict_params)
    if dict_params['model'] == 'vw':
        model = train_vw(dict_params)
    if dict_params['model'] == 'pdsparse':
        model = train_pdsparse(dict_params)
        
    end = time.time()
    run_time = end - start
    print('Model trained in ' + str(run_time) + ' seconds')
    return model,run_time

def predict_model(features,model,dict_params):
    print('Making predictions...')
    start = time.time()
    
    if dict_params['model'] == 'liblinear':
        preds = predict_liblinear(features,model)
    if dict_params['model'] == 'vw':
        preds = predict_vw(model,dict_params)
    if dict_params['model'] == 'pdsparse':
        preds = predict_pdsparse(model,dict_params)    
    
    end = time.time()
    run_time = end - start
    print('Predictions made in ' + str(run_time) + ' seconds')
    
    return preds, run_time

def test_model(features,labels,test_features,test_labels,dict_params):
    lenc = l.labelenc()
    #lenc_test = l.labelenc()
    labels = lenc.fit_transform(labels)
    test_labels = lenc.fit_transform(test_labels)
    dict_params['nr_classes']=max(labels)
    dict_params['current_fold']='test'
    write_files(features,labels,dict_params,data_dir,file_type='train') # Write training files
    write_files(test_features,test_labels,dict_params,data_dir,file_type='test')
    
    model, train_time = train_model(features,labels,dict_params)
    preds, pred_time = predict_model(test_features,model,dict_params)
    print(preds)
    cm = confusion_matrix(test_labels,preds)
    acc = cm.trace()/cm.sum()
    stats = {'acc':acc,'train_time':train_time,'pred_time':pred_time}
    return stats,cm


def crossval(features,labels,dict_params,n_folds=10, cv_type = 'stratified'):
    if cv_type == 'stratified':
        skf = StratifiedKFold(n_folds, shuffle = True,random_state=123)
    else:
        skf = KFold(n_folds,shuffle = True)
    lenc = l.labelenc()
    labels = lenc.fit_transform(labels)
    dict_params['nr_classes']=max(labels)
    folds = skf.split(features,labels)
    dic_stats = {}
    i = 0
    sum_cv = 0
    total_training_time = 0
    total_prediction_time = 0
    for train_index, test_index in folds:
        print('Fold {0} out of {1}'.format(i,n_folds))
        
        dict_params['current_fold']=i
        write_files(features[train_index],labels[train_index],dict_params,data_dir,file_type='train') # Write training files
        write_files(features[test_index],labels[test_index],dict_params,data_dir,file_type='test') # Write training files
        
        model, training_time = train_model(features[train_index],labels[train_index],dict_params)
        preds, prediction_time = predict_model(features[test_index],model,dict_params)
        cm = confusion_matrix(labels[test_index],preds)
        
        sum_cv = sum_cv + cm.trace()/cm.sum()
        total_training_time = total_training_time + training_time
        total_prediction_time = total_prediction_time + prediction_time
        
        dic_stats['cv_score_{}'.format(i)] = cm.trace()/cm.sum()
        dic_stats['training_time_{}'.format(i)] = training_time
        dic_stats['prediction_time_{}'.format(i)] = prediction_time
        print(cm.trace()/cm.sum())
        i = i +1
    del dict_params['current_fold']    
    dic_stats['total_training_time'] = total_training_time
    dic_stats['total_prediction_time'] = total_prediction_time
    dic_stats['cv_score'] = sum_cv/n_folds   
    return dic_stats

#%% LIBLINEAR
def train_liblinear(features,labels,dict_params, cv = False):
    if cv is False:
        model = liblinearutil.train(labels, features, '-c {0} -s {1} -n {2} -q'.format(dict_params['C'],dict_params['s'],dict_params['n_threads']))
    else:
        model = liblinearutil.train(labels, features, '-c {0} -s {1} -v {2} -n {3} -q'.format(dict_params['C'],dict_params['s'],dict_params['folds'],dict_params['n_threads']))
    
    if dict_params['s'] == 4:
        
        model = liblinearutil.train(labels, features, '-c {0} -s {1}'.format(dict_params['C'],dict_params['s']))
    return model

def predict_liblinear(features,model,labels=[]):   
    preds, p_acc, p_val = liblinearutil.predict(labels, features, model)
    preds = list(map(int, preds))
    return preds


#%% VOWPAL
def train_vw(dict_params):
    filename = get_data_filename(dict_params,'train')
    model = get_model_filename(dict_params)
    
    if dict_params['holdout_off'] is True:
        comm = 'perl -pe \'s/\s/ | /\' {0} | {1}vw --log_multi {2} -f {3} -c -k --quiet --loss_function={4} --passes {5} --learning_rate {6} --holdout_off '.format(filename,vowpal_dir,dict_params['nr_classes'],model,dict_params['loss_function'],dict_params['passes'],dict_params['learning_rate'])
    else:
        comm = 'perl -pe \'s/\s/ | /\' {0} | {1}vw --log_multi {2} -f {3} -c -k --quiet --loss_function={4} --passes {5} --learning_rate {6} '.format(filename,vowpal_dir,dict_params['nr_classes'],model,dict_params['loss_function'],dict_params['passes'],dict_params['learning_rate'])
    #print(comm)
    run_subproc(comm,'ola')
    os.remove(filename)
    return model

def predict_vw(model,dict_params):
    model = get_model_filename(dict_params)
    preds_file = model_dir+'preds'
    filename = get_data_filename(dict_params,'test')
    comm = 'perl -pe \'s/\s/ | /\' {0} | {1}vw -t -i {2} -p {3} --quiet'.format(filename,vowpal_dir,model,preds_file)
    run_subproc(comm,'ola')
    with open(preds_file,'r') as f:
        preds=f.readlines()
    #print(comm)
    preds = [int(x) for x in preds]    
    #os.remove(preds_file)
    os.remove(model)
    os.remove(filename)
    
    return preds

#%% PD-SPARSE
def train_pdsparse(dict_params):
    filename = get_data_filename(dict_params,'train')
    model = get_model_filename(dict_params)

    comm = '{dir_pd}multiTrain -m {iter_pd} -c {C} -l {l_lambda} {data} {model}'.format(data=filename,dir_pd=pdsparse_dir,model=model,iter_pd=dict_params['iter_pd'],C=dict_params['C'],l_lambda=dict_params['lambda'])
    run_subproc(comm,'ola')
    os.remove(filename)
    return model

def predict_pdsparse(model,dict_params):
    model = get_model_filename(dict_params)
    preds_file = model_dir+'preds'
    filename = get_data_filename(dict_params,'test')
    preds = []
    
    comm = '{dir}multiPred {data} {model} -p 1 {preds_file}'.format(data=filename,dir=pdsparse_dir,model=model,preds_file=preds_file)
    run_subproc(comm,'ola')
    with open(preds_file,'r') as f:
        content = f.readlines()
        for line in content:
            results = re.findall(r'(\d+):\d+.\d+', line)
            preds.append(int(results[0]))
    #os.remove(preds_file)
    os.remove(model)
    os.remove(filename)
    
    return preds
    









