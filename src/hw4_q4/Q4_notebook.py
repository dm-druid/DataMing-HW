
# coding: utf-8

# # Import packages and Pre-define func

# In[1]:


import csv
import numpy as np
import pandas as pd
import pip
import matplotlib.pyplot as plt
import numpy as np

def pip_install(module):
    pip.main(['install', module])

# pip_install('scipy')
# pip_install('matplotlib')


# In[2]:


def prepare_data():
    raw_data = pd.read_csv('../../data/trade.csv')
    # count the number of vipno and pluno
    vip_set = set(raw_data.vipno)
    n_vip = len(vip_set)
    plu_set = set(raw_data.pluno)
    n_plu = len(plu_set)
    
    # construct the data matrix of the trade
    vipno = list(vip_set)
    pluno = list(plu_set)
    trade_mat = pd.DataFrame(np.zeros([n_plu, n_vip]), index=pluno, columns=vipno)
    l = len(raw_data)
    for i in range(l):
        p = raw_data.loc[i, 'pluno']
        v = raw_data.loc[i, 'vipno']
        a = raw_data.loc[i, 'amt']
        trade_mat.at[p, v] += a
        
    # apply round
    trade_mat.apply(np.round)
    trade_mat = trade_mat.astype('int64') # it depends
    return trade_mat, vipno, pluno, n_vip, n_plu


# # Clustering

# In[4]:


# get the data for clustering
from sklearn.preprocessing import StandardScaler

trade_mat, vipno, pluno, n_vip, n_plu = prepare_data()
X = trade_mat.values
X = X.transpose()
# standarize
scaler = StandardScaler()
X = scaler.fit_transform(X)
# print(len(X))


# # Silhouette coefficient - k cluster plot

# In[5]:


from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

def cluster_eva(X, k):
    estimator = GaussianMixture(n_components=k, covariance_type='tied', max_iter=20, random_state=0)
    estimator.fit(X)
    cluster_labels = estimator.predict(X)
    cluster_num = len(set(cluster_labels))
    
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_components =", k,
          "The average silhouette_score is :", silhouette_avg)
    print("cluster number =", cluster_num)
    return silhouette_avg, cluster_labels


# In[6]:


ks = []
y = []
labels = []
for k in range(2, 12):
    ks.append(k)
    sil_avg, cluster_labels = cluster_eva(X, k)
    y.append(sil_avg)
    labels.append(cluster_labels)


# In[16]:


# print(len(labels), len(ks), len(y))
# ks = []
# for k in range(2,13):
#     ks.append(k)
# print(len(ks))


# In[7]:


f = interp1d(ks, y)

xnew = ks.copy()
plt.plot(ks, y, 'o', xnew, f(xnew), '-')
plt.legend(['data', 'linear'], loc='best')
plt.xlabel("n_components")
plt.ylabel("The average silhouette_score")
plt.title("The Silhouette-n plot")
plt.show()


# In[29]:


# print(labels[8])


# # (a) Compare with KMeans

# In[8]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=10)
y_kmeans = kmeans.fit_predict(X)
y_GMM = labels[ks.index(2)]


# In[17]:


# since there are only 2 clusters, I just make a simple comparison
corr = 0
for i, y1 in enumerate(y_kmeans):
    if y1 == y_GMM[i]:
        corr += 1
print("The accuracy for KMeans = ", corr / float(len(y_kmeans)))


# # (b) Compare with DBSCAN

# In[18]:


from sklearn.cluster import DBSCAN

y_DBSCAN = DBSCAN(eps=130, min_samples=20).fit_predict(X)
cluster_num = len(set(y_DBSCAN))

y_GMM = labels[ks.index(cluster_num)]
# print(y_GMM)


# In[19]:


y_DBSCAN[y_DBSCAN == -1] = 1 # change the cluster labels
# since there are only 2 clusters, I just make a simple comparison

corr = 0
for i, y1 in enumerate(y_DBSCAN):
    if y1 == y_GMM[i]:
        corr += 1
print("The accuracy for DBSCAN = ", corr / float(len(y_DBSCAN)))


# # (c) Test with LSH

# In[20]:


from lshash.lshash import LSHash

e = 0.01
lsh = LSHash(round(n_vip * e), n_plu)
for v in vipno:
    feature = list(trade_mat[v])
    lsh.index(feature, extra_data=v)


# In[25]:


import random

def lsh_test():
    correct = 0
    # pick up a random vipno
    pick_vip = random.randint(1, n_vip)
    cluster_label = labels[ks.index(7)] 
    pick_vipno = vipno[pick_vip]
    cluster1 = cluster_label[pick_vip]
    print("Pick up a vip: {0}, cluster = {1}".format(pick_vipno, cluster1))

    # lsh query and write the results
    candi = lsh.query(list(trade_mat[pick_vipno]))
    l = len(candi)
#     print(l)
    for i, item in enumerate(candi[1:]):
        dist = item[1]
        feature = list(item[0][0])
        v = item[0][1]
        lsh_pair_no = vipno.index(v)
        cluster2 = cluster_label[lsh_pair_no]
#         print("for vip {0}: distance = {1}, cluster = {2}".format(v, dist, cluster2))
        if cluster2 == cluster1:
            correct += 1
    print("accuracy =", correct, '/', l)
    return correct/float(l)


# In[26]:


correctness = 0
times = 50
for i in range(times):
    print("Time #", i+1)
    try:
        correctness += lsh_test()
    except Exception as e:
        continue
print("The total average accuracy is =", correctness / times)

