import l1k as l 
import seaborn as sns
import matplotlib.pyplot as plt   
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


df,sig,genes = l.load_data()

#%%
features,labels = l.features_labels(df,sig,cell='A375',all_cells='All', dmso=True)
le = LabelEncoder()
labels_b = le.fit_transform(labels)

l.df_to_libsvm(features,labels_b,file_name='../data/l1k_a375')
#%%
l.dump_fastxml(features,labels_b,'../data/l1k_bt20_fml')

#%%
#labels = l.cluster_kmeans(features,100)
#pca = PCA(n_components=2)
#features = pca.fit_transform(features)
#%%
model,preds,cm,met = l.run_model('liblinear',features,labels,n_splits=5,class_weight=None)
#%%
#l.run_fastXML(features,labels,file_name='l1k_bt20_fml')
#%%
#score = l.crossval(features,labels,'sgdc',splits = 5)

#try pca
#%%

sns.set()
ax = sns.heatmap(cm*100,cmap="YlGnBu")
plt.show()
#%%
tsne = TSNE(n_components=3, random_state=0)
ts = tsne.fit_transform(features)
plt.plot(ts[:,0],ts[:,1],'x')
le = LabelEncoder()
ts_labels= le.fit_transform(labels)

#%%
x = ts[:,0]
y = ts[:,1]
Cluster = ts_labels    # Labels of cluster 0 to 3

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x,y,c=Cluster,s=10)

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()
#%%

#%%
preds = l.read_score_fastxml('/media/sf_project/Linux/model/score')
#%%
cm,met,preds = l.run_fastXML(features,labels_b,file_name='l1k_fml_a375',n_splits=5,create_files=True,score_filename = 'score')