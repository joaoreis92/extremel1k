import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier as gpc
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support as metrics
import sys
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.svm import LinearSVC as lsvc
from sklearn.svm import SVC as svc
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier as rf
#from vowpalwabbit.sklearn_vw import VWClassifier as vw
from sklearn.linear_model import SGDClassifier as sgdc
import time
import pickle
from sklearn.datasets import dump_svmlight_file
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from multiclass_svm import MulticlassSVM
from sklearn.preprocessing import LabelEncoder
import re
sys.path.insert(0,'/media/sf_project/Linux/liblinear-multicore-2.11-2/python/') # Path to LIBLINEAR
import liblinearutil 
from operator import itemgetter
import subprocess
# Read data in LIBSVM format

#%%
data_dir = '/media/sf_project/Linux/data/'
model_dir = '/media/sf_project/Linux/model/'
fastXML_dir = '/media/sf_project/Linux/FastXML_PfastreXML/FastXML/' 

#df = pd.read_csv('df_978_l5.csv')
def load_data(df_file='../df_978_l5.p',sig_file='../GSE70138_Broad_LINCS_sig_info.csv',genes_file='../GSE92742_Broad_LINCS_gene_info.csv'):
    print('Loading data...')
    start = time.time()    
    df = pickle.load(open(df_file,'rb'))
    sig = pd.read_csv(sig_file)
    genes = pd.read_csv(genes_file)
    end = time.time()    
    print('Data succefully loaded in ' + str(end - start) + 'seconds')
    return df,sig,genes


#%%
def get_model(choice='lr',class_weight=None):
    if choice=='svc':
        model = svc(verbose=1,class_weight=class_weight,n_jobs=-1)
        
    elif choice == 'lsvc':
        model = lsvc(class_weight=class_weight,n_jobs=-1)
    elif choice == 'knn':
        model = KNeighborsClassifier()
    elif choice =='msvm':
        model = MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
        
    elif choice == 'gnb':
        model = gnb(class_weight=class_weight)
        
    elif choice == 'gpc':
         model = gpc(class_weight=class_weight)
    elif choice == 'sgdc':
        model = sgdc(class_weight=class_weight)
    
    elif choice == 'rf':
         model = rf(class_weight=class_weight)
#   elif choice == 'vw':
#         model = vw()
    else:
        model = lr(class_weight=class_weight)
    return model
            
    
#%%
def features_labels(df,sig,cell='BT20',all_cells=False, dmso=True):
    if all_cells == False:
        info_cell = sig.loc[sig['cell_id']==cell]
        mat_cell = df[info_cell['sig_id'].tolist()]
    else:
        info_cell = sig
        mat_cell = df
        
    if dmso == False:
        info_cell = info_cell.loc[info_cell['pert_iname'] != 'DMSO']
        mat_cell = df[info_cell['sig_id'].tolist()]
        
    
    labels = info_cell['pert_iname'].values
    features = mat_cell.T.as_matrix()
    
    return features,labels

#%%
def nparray_to_listdicts(arr):
    print('Converting array to liblinear compatible format...')
    start = time.time()    
    list_dicts=[]
    for row in arr:
        row_dict = {}
        for i in range(1,len(row)):
            row_dict[i]=row[i-1]
        list_dicts.append(row_dict)
    end = time.time()
    print('Array converted in ' + str(end - start) + ' seconds')
    return list_dicts

def train_liblinear(features,labels,C=1,s=1,folds=10,threads=4):   
    print('Training model...')
    start = time.time()
    model = liblinearutil.train(labels, features, '-c {0} -s {1} -v {2} -n {3}'.format(C,2,folds,threads))
    end = time.time()
    print('Model trained in ' + str(end - start) + ' seconds')
    return model


def predict_liblinear(features,labels,model):
    print('Loading predictions...')
    start = time.time()    
    preds, p_acc, p_val = liblinearutil.predict(labels, features, model)
    preds = list(map(int, preds))
    end = time.time()
    print('Predictions made in ' + str(end - start) + ' seconds')
    return preds

def run_liblinear(features,labels,indexes):
    le = LabelEncoder()
    labels_b = le.fit_transform(labels)
    train_features = nparray_to_listdicts(features[indexes[0]])
    test_features = nparray_to_listdicts(features[indexes[1]])
    model =  train_liblinear(train_features,labels_b[indexes[0]])
    preds = predict_liblinear(test_features,labels_b[indexes[1]],model)
    preds =le.inverse_transform(preds)
    
    return model,preds
           
def df_to_libsvm(features,labels,file_name='l1k',test=True):
    print('Saving with libsvm format...')
    start = time.time()
    if test is True:
        train_features,train_labels,test_features,test_labels=split_data(features,labels)
        dump_svmlight_file(train_features,train_labels,file_name+'.train',zero_based=False)
        #dump_svmlight_file(test_features,test_labels,file_name+'.test',zero_based=False)
        dump_svmlight_file(test_features,test_labels,file_name+'.heldout',zero_based=False)
    else:
        dump_svmlight_file(features,labels,file_name+'.train',zero_based=False)
    end = time.time()
    print('File saved in ' + str(end - start) + ' seconds')
    
#%%
def dump_fastxml(features,labels,file_name):
    train_features,train_labels,test_features,test_labels=split_data(features,labels)
    convert_fastxml(train_features,train_labels,file_name+'.train')
    convert_fastxml(test_features,test_labels,file_name+'.test')
    return test_labels
    
def convert_fastxml(features,labels,file_name):
    print('Saving with FastXml format...')
    start = time.time()
    with open(file_name+'.X','w') as f:
        f.write('{} {}\n'.format(features.shape[0],features.shape[1]))
        for row in features:
            i=0
            ctrl=0
            for expr in row:
                if ctrl==0:
                    f.write('{}:{}'.format(i,float(expr)))
                    ctrl=1
                else:
                    f.write(' {}:{}'.format(i,float(expr)))
                i=i+1
            f.write('\n')
    with open(file_name+'.Y','w') as f:
        f.write('{} {}\n'.format(labels.shape[0],len(set(labels))))
        for row in labels:            
            f.write('{}:1'.format(row,'1'))
            f.write('\n')
    end = time.time()
    print('File saved in ' + str(end - start) + ' seconds')

def read_labels_fasxml(file_name):
    labels = []
    with open(file_name) as f:
        content = f.readlines()
        #print(content)
        for line in content:
            #print(line)
            results = re.findall(r'(\d+):', line)
            #print(results)
            try:
                label = int(results[0])
                labels.append(label)
            except:
                continue
                    
    return np.array(labels)

def read_score_fastxml(file_name):
    labels = []
    with open(file_name) as f:
        content = f.readlines()
        for line in content:
            results = re.findall(r'(\d+):(\d+.\d+)', line)
            try:
                label = int(max(results,key=itemgetter(1))[0])
                labels.append(label)
            except Exception as e:
                print(e)
                continue
                    
    return np.array(labels)
#%%
def train_fastXML(features_file,labels_file):
    print('Training model...')
    start = time.time()
    subprocess.call(['./fastXML_train',features_file,labels_file,model_dir,'-T','4'],cwd=fastXML_dir)
    end = time.time()
    print('Model trained in ' + str(end - start) + ' seconds')

def test_fastXML(features_file,score_file):
    print('Loading predictions...')
    start = time.time()
    subprocess.call(['./fastXML_test',features_file,score_file,model_dir],cwd=fastXML_dir)
    end = time.time()
    print('Predictions loaded in ' + str(end - start) + ' seconds')
    
def run_fastXML(features,labels,file_name=0,n_splits=5,create_files=False,score_filename = 'score'): 
    if create_files is True:
        test_labels=dump_fastxml(features,labels,data_dir+file_name)
    else:
        test_labels = read_labels_fasxml(data_dir+file_name+'.test.Y')
    train_fastXML(data_dir+file_name+'.train.X',data_dir+file_name+'.train.Y')
    test_fastXML(data_dir+file_name+'.test.X',model_dir+score_filename)
    preds = read_score_fastxml(model_dir+score_filename)
    cm = confusion_matrix(test_labels,preds)
    met = metrics(test_labels,preds)
    print(cm.trace()/cm.sum())
    return cm,met,preds
#%%
def split_data(features,labels,n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    indexes = next(skf.split(features,labels))
    train_features = features[indexes[0]]
    train_labels = labels[indexes[0]]
    test_features = features[indexes[1]]
    test_labels = labels[indexes[1]]
    return train_features,train_labels,test_features,test_labels


#%%   Compound vs NC
#sig['is_cp'] = (sig['pert_type'] == 'trt_cp')
#labels = a375_info['is_cp']
#features = a375.T.as_matrix()
#model = gpc()
##model = lr()
#model.fit(features,labels)




#%%

def train_model(model_name,features,labels,indexes,class_weight):
    print('Training model...')
    start = time.time()
    model = get_model(model_name,class_weight)
    model.fit(features[indexes[0]],labels[indexes[0]])
    end = time.time()
    print('Model trained in ' + str(end - start) + 'seconds')
    return model

#%%
def predict_model(model,features,indexes):
    print('Loading predictions...')
    start = time.time()
    preds = model.predict(features[indexes[1]])
    end = time.time()
    print('Predictions made in ' + str(end - start) + ' seconds')
    return preds

#%%
def cmatrix_metrics(labels,indexes,preds):
    cm = confusion_matrix(labels[indexes[1]],preds)
    met = metrics(labels[indexes[1]],preds)
    print(cm.trace()/cm.sum())
    #print(precision_score(labels[indexes[1]],preds,average='micro')) Same thing as above
    return cm,met
    
#%%
def run_model(model_name,features,labels,n_splits=5,class_weight=None):
    skf = StratifiedKFold(n_splits=n_splits)
    indexes = next(skf.split(features,labels))
    if model_name == 'liblinear':
        model,preds = run_liblinear(features,labels,indexes)
    else:    
        model = train_model(model_name,features,labels,indexes,class_weight)
        preds = predict_model(model,features,indexes)
    cm,met = cmatrix_metrics(labels,indexes,preds)
    return model,preds,cm,met

#%%
def cluster_kmeans(features,k=8):
    print('Training model...')
    start = time.time()
    kmeans = KMeans(k)
    kmeans.fit(features)
    end = time.time()
    print('Model trained in ' + str(end - start) + 'seconds')
    return kmeans.labels_
    


#%%
def crossval(features,labels,model_name,splits =5):
    skf = StratifiedKFold(splits)
    cv_score = 0
    model = get_model(model_name)
    for train_index, test_index in skf.split(features,labels):
        model.fit(features[train_index],labels[train_index])
        preds = model.predict(features[test_index])
        cm = confusion_matrix(labels[test_index],preds)
        cv_score = cv_score + cm.trace()/cm.sum()
        print(cm.trace()/cm.sum())
    return cv_score/splits
        