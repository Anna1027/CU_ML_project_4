#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplg
import scipy.sparse.linalg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#read files
scores_df = pd.read_csv("hw4-data/CFB2019_scores.csv", delimiter = ",",header = None, names = ["A_index", "A_points", "B_index", "B_points"])
scores_mat = np.genfromtxt("hw4-data/CFB2019_scores.csv", delimiter = ",")
teams_df = pd.read_table("hw4-data/TeamNames.txt", header = None, names = ["Team_Name"])


# In[22]:


#Initialize M matrix
M = np.zeros((teams_df.shape[0], teams_df.shape[0]))

#create M 
for i in range(scores_mat.shape[0]):
    M[int(scores_mat[i][0])-1][int(scores_mat[i][0])-1] = M[int(scores_mat[i][0])-1][int(scores_mat[i][0])-1] + int(scores_mat[i][1] > scores_mat[i][3]) + (1.0*scores_mat[i][1]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][2])-1][int(scores_mat[i][2])-1] = M[int(scores_mat[i][2])-1][int(scores_mat[i][2])-1] + int(scores_mat[i][1] < scores_mat[i][3]) + (1.0*scores_mat[i][3]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][0])-1][int(scores_mat[i][2])-1] = M[int(scores_mat[i][0])-1][int(scores_mat[i][2])-1] + int(scores_mat[i][1] < scores_mat[i][3]) + (1.0*scores_mat[i][3]/(scores_mat[i][1] + scores_mat[i][3]))
    M[int(scores_mat[i][2])-1][int(scores_mat[i][0])-1] = M[int(scores_mat[i][2])-1][int(scores_mat[i][0])-1] + int(scores_mat[i][1] > scores_mat[i][3]) + (1.0*scores_mat[i][1]/(scores_mat[i][1] + scores_mat[i][3]))

#Normalize M 
for i in range(M.shape[0]):
    M[i] = M[i] / np.sum(M, axis = 1)[i]

#find states 
w_all = []
w_10000 = []

for t in [10, 100,1000,10000]:
    w = (1.0/769)*np.ones(769)
    
    if t == 10000:
        for i in range(t):
            w = w.dot(M)
            w_10000.append(w)
        w_all.append(w)
    else:
        for i in range(t):
            w = w.dot(M)
        w_all.append(w)
        


# In[23]:


#get top team names 
results_df = pd.DataFrame()
for i in range(4):
    results_df = pd.concat([results_df, teams_df.iloc[np.argsort(w_all[i])[::-1][:25], :].reset_index().Team_Name], axis = 1)
    results_df = pd.concat([results_df, pd.Series(w_all[i][np.argsort(w_all[i])[::-1][:25]])],axis = 1)
    
results_df.columns = ['t = 10', 'weight', 't = 100', 'weight', 't = 1000', 'weight', 't = 10000', 'weight']
results_df.index+=1
results_df


# In[24]:


#Computing Eigen Vectors 

w_inf = scipy.sparse.linalg.eigs(M.T,k=1,sigma=1.0)[1]
w_inf = w_inf/np.sum(w_inf)
w_inf_rep = np.tile(w_inf.reshape(769),(10000,1))
norms = nplg.norm(w_10000 - w_inf_rep,1,axis = 1)

_=plt.figure(figsize = (10,8))
_=plt.plot(range(10000),norms)
_=plt.xlim([-100,10100])
_=plt.title("Variation of $|w_\infty - w_t |_1$ with t", fontsize = 18)
_=plt.xlabel("t",fontsize = 14)
_=plt.ylabel("$|w_\infty - w_t |_1$",fontsize = 14)


# In[ ]:




