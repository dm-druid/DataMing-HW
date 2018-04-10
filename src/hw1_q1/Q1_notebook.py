
# coding: utf-8

# # Package import

# In[2]:


import csv
import numpy as np
import pandas as pd
import pip

def pip_install(module):
    pip.main(['install', module])


# # 1.  Read data 

# In[3]:


raw_data = pd.read_csv('../../data/trade.csv')
print("Read data...")


# In[4]:


# count the number of vipno and pluno
vip_set = set(raw_data.vipno)
n_vip = len(vip_set)
print("vip number =", n_vip)

plu_set = set(raw_data.pluno)
n_plu = len(plu_set)
print("plu number =", n_plu)


# In[6]:


# construct the data matrix of the trade
vipno = list(vip_set)
pluno = list(plu_set)
trade_mat = pd.DataFrame(np.zeros([n_plu, n_vip]), index=pluno, columns=vipno)


# In[7]:


l = len(raw_data)
for i in range(l):
    p = raw_data.loc[i, 'pluno']
    v = raw_data.loc[i, 'vipno']
    a = raw_data.loc[i, 'amt']
    trade_mat.at[p, v] += a


# In[8]:


# apply round
trade_mat.apply(np.round)
trade_mat = trade_mat.astype('int64') # it depends
trade_mat # show the table


# # 2. LSH

# In[9]:


from lshash.lshash import LSHash
import random


# In[10]:


o = open('lsh_output.txt', 'w') # create a file to write the results

# loop with different hash size
for e in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
    lsh = LSHash(round(n_vip * e), n_plu)
    for v in vipno:
        feature = list(trade_mat[v])
        lsh.index(feature, extra_data=v)
        
    # pick up a random vipno
    pick_vip = random.randint(1, n_vip)
    pick_vip = vipno[pick_vip]
    o.write("Hash_size = {} * n_plu \n".format(e))
    o.write("Pick up a vip: {}\n".format(pick_vip))
    
    # lsh query and write the results
    candi = lsh.query(list(trade_mat[pick_vip]), 6, distance_func='hamming')
#     print(len(candi))
    for i, item in enumerate(candi[1:]):
        dist = item[1]
        feature = list(item[0][0])
        v = item[0][1]
        o.write("Top {0} : vipno = {1}, distance = {2}\n".format(i+1, v, dist))
    o.write("\n")

o.close()
print("The lshash results have been saved in file 'lsh_output.txt'.")

