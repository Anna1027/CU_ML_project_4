#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nplg
import scipy.sparse.linalg
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Parse file

file = open("hw4-data/nyt_data.txt","r", encoding = 'utf-8-sig') 
X = np.zeros((3012,8447))
c1 =0

for line in file.readlines():
    curr_line = line.split(",")
    
    for ele in curr_line:
        X[int(ele.split(":")[0])-1][c1] = ele.split(":")[1]
    c1 += 1


# In[3]:


#function to calculate objective function 
def objective():
    err = (10**(-16))
    WH = W.dot(H)
    I = np.ones(WH.shape)
    obj = X*np.log(I/(WH + err)) + WH
    return np.sum(obj)


# In[4]:


#Initialize W and H 
W = []
H = []

for i in range(X.shape[0]):
    W.append(np.random.uniform(1,2,25))
W = np.array(W)

for i in range(X.shape[1]):
    H.append(np.random.uniform(1,2,25))
H = (np.array(H)).T

obj_all = []
err = (10**(-16))
for iter in range(100):
    H = H*((W.T/np.sum(W.T, axis = 1)[:, np.newaxis]).dot(X/(W.dot(H)+err)))
    W = W *((X/(W.dot(H)+err)).dot(H.T/np.sum(H.T, axis = 0)[np.newaxis, :]))
    obj_all.append(objective())


# In[5]:


_=plt.figure(figsize = (10, 8))
_=plt.title("Variation_of Divergence Penalty with iterations", fontsize = 1)
_=plt.xlabel('iter', fontsize = 14)
_=plt.ylabel("Divergence Penalty", fontsize = 14)
_=plt.plot(range(100), obj_all)


# In[6]:


W_norm = (W/np.sum(W, axis = 0)[np.newaxis, :])

#Get Top Weights
weights_ind = np.around(np.sort(W_norm, axis = 0)[-10:, :].T, decimals = 6)

#Get Top weighted Words' index 
words_ind = np.argsort(W_norm, axis = 0)[-10:, :].T

#read vocab 
vocab = pd.read_table("hw4-data/nyt_vocab.dat", delimiter = ",",header = None)


# In[7]:


pd.Series(weights_ind[0][::-1])


# In[11]:


Results_df = pd.DataFrame()
for i in range(25):
    Results_df = pd.concat([Results_df, vocab.iloc[words_ind[i][::-1],:].reset_index(),pd.Series(weights_ind[i][::-1])], axis = 1)
    
    


# In[12]:


Results_df.to_csv("Topics.csv")


# In[18]:


# Create 5 x 5 matrix of topics

mat_topics = [] 
temp = []

for i in range(25):
    
    b = list(np.fliplr(weights_ind)[i])
    c = list((np.char.array(vocab.iloc[np.fliplr(words_ind)[i],:])).reshape(10,))
    topic = [str(m)+'-'+str(n) for m,n in zip(c,b)]
    temp.append(','.join(topic))
    
    if len(temp)==5:
        mat_topics.append(temp)
        temp = []

(pd.DataFrame(np.array(mat_topics)))



# In[ ]:





# In[ ]:




