from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder
import sys
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support as metrics
import time
sys.path.insert(0,'~/project/liblinear-multicore-2.11-2/python/') # Path to LIBLINEAR
import liblinearutil 
import pandas as pd

#%%

def train_model(features,labels,dict_params):
    print('Training model...')
    start = time.time()
    if dict_params['model'] == 'liblinear':
        model = train_liblinear(features,labels,dict_params)
        
    end = time.time()
    run_time = end - start
    print('Model trained in ' + str(run_time) + ' seconds')
    return model,run_time

def predict_model(features,model,dict_params):
    print('Making predictions...')
    start = time.time()
    
    if dict_params['model'] == 'liblinear':
        preds = predict_liblinear(features,model)
    
    end = time.time()
    run_time = end - start
    print('Predictions made in ' + str(run_time) + ' seconds')
    
    return preds, run_time

def test_model(features,labels,test_features,test_labels,dict_params):
    lenc = LabelEncoder()
    labels_b = lenc.fit_transform(labels)
    test_labels_b = lenc.fit_transform(test_labels)
    
    model, train_time = train_model(features,labels_b,dict_params)
    preds, pred_time = predict_model(test_features,model,dict_params)
    
    cm = confusion_matrix(test_labels_b,preds)
    acc = cm.trace()/cm.sum()
    stats = {'acc':acc,'train_time':train_time,'pred_time':pred_time}
    return stats,cm
    
    

def crossval(features,labels,dict_params,n_folds=10, cv_type = 'stratified'):
    if cv_type == 'stratified':
        skf = StratifiedKFold(n_folds)
    else:
        skf = KFold(n_folds,shuffle = True)
    lenc = LabelEncoder()
    labels = lenc.fit_transform(labels)
    folds = skf.split(features,labels)
    dic_stats = {}
    i = 0
    sum_cv = 0
    total_training_time = 0
    total_prediction_time = 0
    for train_index, test_index in folds:
        print('Fold {0} out of {1}'.format(i,n_folds))
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
        
    dic_stats['total_training_time'] = total_training_time
    dic_stats['total_prediction_time'] = total_prediction_time
    dic_stats['cv_score'] = sum_cv/n_folds   
    return dic_stats

#%% LIBLINEAR
def train_liblinear(features,labels,dict_params, cv = False):
    if cv is False:
        model = liblinearutil.train(labels, features, '-c {0} -s {1} -n {2}'.format(dict_params['C'],dict_params['s'],dict_params['n_threads']))
    else:
        model = liblinearutil.train(labels, features, '-c {0} -s {1} -v {2} -n {3}'.format(dict_params['C'],dict_params['s'],dict_params['folds'],dict_params['n_threads']))
    
    return model

def predict_liblinear(features,model,labels=[]):   
    preds, p_acc, p_val = liblinearutil.predict(labels, features, model)
    preds = list(map(int, preds))
    return preds








