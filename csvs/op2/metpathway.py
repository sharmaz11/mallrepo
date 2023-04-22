#!/usr/bin/env python
# coding: utf-8

# In[1]:

# mport nonlincausality as nlc

# In[240]:

import sys

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense,LSTM,Dropout
import matplotlib.pyplot as plt
from statsmodels.tsa.tsatools import lagmat2ds
from scipy.stats import wilcoxon,uniform,randint

from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error,r2_score


# In[2]:

# In[48]:


import time
import multiprocessing as mp



# In[3]:

# In[49]:


import os



# In[8]:

# In[9]:

# In[50]:


#os.chdir("C:/Users/malli/OneDrive/Documents/hackbio/mouse time series data")


# In[10]:

# In[51]:


def createdata(x,lag,tag):
    data_X=lagmat2ds(x[:,0],lag-1,trim="both")[:-1,:]
    data_X=data_X.reshape(data_X.shape[0],data_X.shape[1],1)
    data_Y=lagmat2ds(x[:,1],lag-1,trim="both")[:-1,:]
    data_Y=data_Y.reshape(data_Y.shape[0],data_Y.shape[1],1)
    #data_XY=np.concatenate([data_X,data_Y],axis=2)
    if(tag=="YX"):
        data_YX=np.concatenate([data_X,data_Y],axis=2)
        print("created data")
        return data_YX,data_X
    if(tag=="XY"):
        data_XY=np.concatenate([data_Y,data_X],axis=2)
        print("created data")
        return data_XY,data_Y
    


# In[ ]:





# In[11]:

# STM model where Y->X

# In[52]:


def createmodel(data,lstm,dense,dropout_rate,numlayers):
    np.random.seed(42)
    tf.random.set_seed(42)
    model=Sequential()
    model.add(LSTM(lstm,activation='tanh',recurrent_activation='tanh',use_bias=True,return_sequences=False,dropout=dropout_rate,
                   recurrent_dropout=dropout_rate,
                   input_shape=(data.shape[1],data.shape[2])))
    model.add(Dropout(dropout_rate))
    for i in range(numlayers):
        model.add(Dense(dense, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='linear'))
    opt=tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt,loss='mse',metrics=['mean_squared_error'])
    #print(model.summary())
    return(model)


# In[12]:

# In[53]:


def train_model(data_X,data_X_test,data_YX,data_YX_test,X,X_test,
                data_XY,data_XY_test,data_Y,data_Y_test,Y,Y_test,
                lstm,dense,dropout_rate,numlayers):
    tf.keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model_X=createmodel(data_X,lstm,dense,dropout_rate,numlayers)
    model_YX=createmodel(data_YX,lstm,dense,dropout_rate,numlayers)
    model_Y=createmodel(data_Y,lstm,dense,dropout_rate,numlayers)
    model_XY=createmodel(data_XY,lstm,dense,dropout_rate,numlayers)
    
    
    
    
    history_X=model_X.fit(data_X,X,epochs=100,verbose=0,batch_size=64)
    history_YX=model_YX.fit(data_YX,X,epochs=100,verbose=0,batch_size=64)
    history_Y=model_Y.fit(data_Y,Y,epochs=100,verbose=0,batch_size=64)
    history_XY=model_XY.fit(data_XY,Y,epochs=100,verbose=0,batch_size=64)
    
    XpredX=model_X.predict(data_X_test,verbose=0)
    XpredX=XpredX.reshape(XpredX.size)
    error_X=X_test-XpredX
    r2_1=r2_score(X_test,XpredX)
    YXpredX=model_YX.predict(data_YX_test,verbose=0)
    YXpredX=YXpredX.reshape(YXpredX.size)
    error_YX=X_test-YXpredX
    r2_2=r2_score(X_test,YXpredX)
    S_YX,pval_YX=wilcoxon(np.abs(error_X),np.abs(error_YX),alternative="greater")
    
    
    
    YpredY=model_Y.predict(data_Y_test,verbose=0)
    YpredY=YpredY.reshape(YpredY.size)
    error_Y=Y_test-YpredY
    r2_1a=r2_score(Y_test,YpredY)
    XYpredY=model_XY.predict(data_XY_test,verbose=0)
    XYpredY=XYpredY.reshape(XYpredY.size)
    error_XY=Y_test-XYpredY
    r2_2a=r2_score(Y_test,XYpredY)
    S_XY,pval_XY=wilcoxon(np.abs(error_Y),np.abs(error_XY),alternative="greater")
        
    
    del model_X
    del model_YX
    del model_Y
    del model_XY
    print("Done with training ...")
    return(history_X,history_YX,history_Y,history_XY,error_X,error_YX,error_Y,error_XY,r2_1,r2_2,r2_1a,r2_2a,S_YX,pval_YX,S_XY,pval_XY)


# In[13]:

# In[79]:


def causality(data_1,data_2,lags,lstm,dense,dropout_rate,numlayers):
    start=time.time()
    gene1,gene2=data_1.columns
    data_train=data_1.to_numpy()
    data_test=data_2.to_numpy()
    history_X=[]
    history_YX=[]
    history_Y=[]
    history_XY=[]
    results_YX=pd.DataFrame(columns=["lag","tag","RSS_1","RSS_2","r2_1","r2_2","S","pval"])
    results_XY=pd.DataFrame(columns=["lag","tag","RSS_1","RSS_2","r2_1","r2_2","S","pval"])
    
    def process(i,lstm,dense,dropout_rate):
        tf.keras.backend.clear_session()
        data_YX,data_X=createdata(data_train,i,"YX")
        data_YX_test,data_X_test=createdata(data_test,i,"YX")
        data_XY,data_Y=createdata(data_train,i,"XY")
        data_XY_test,data_Y_test=createdata(data_test,i,"XY")
        X=data_train[i:,0]
        X_test=data_test[i:,0]
        Y=data_train[i:,1]
        Y_test=data_test[i:,1]
        history_x,history_yx,history_y,history_xy,error_x,error_yx,error_y,error_xy,r2_1,r2_2,r2_1a,r2_2a,S_yx,pval_yx,S_xy,pval_xy=train_model(data_X,data_X_test,data_YX,data_YX_test,X,X_test,
                data_XY,data_XY_test,data_Y,data_Y_test,Y,Y_test,
                lstm,dense,dropout_rate,numlayers)
        print("done with lag")
        return(i,history_x,history_yx,history_y,history_xy,error_x,error_yx,error_y,error_xy,
               r2_1,r2_2,r2_1a,r2_2a,S_yx,pval_yx,S_xy,pval_xy)
    n_steps=np.arange(1,lags+1)
    result=Parallel(n_jobs=-1,verbose=0)(delayed(process)(i,lstm,dense,dropout_rate) for i in n_steps)
    for r in result:
        history_X.append(r[1])
        history_YX.append(r[2])
        history_XY.append(r[3])
        history_Y.append(r[4])
        RSS_1=sum(r[5]**2)
        RSS_2=sum(r[6]**2)
        RSS_1a=sum(r[7]**2)
        RSS_2a=sum(r[8]**2)
        r2_1=r[9]
        r2_2=r[10]
        r2_1a=r[11]
        r2_2a=r[12]
        result_1=pd.DataFrame([[r[0],"YX",RSS_1,RSS_2,r2_1,r2_2,r[13],r[14]]],columns=["lag","tag","RSS_1","RSS_2","r2_1","r2_2","S","pval"])
        result_2=pd.DataFrame([[r[0],"XY",RSS_1a,RSS_2a,r2_1a,r2_2a,r[15],r[16]]],columns=["lag","tag","RSS_1","RSS_2","r2_1","r2_2","S","pval"])
        results_XY=pd.concat([results_XY,result_2],axis=0)
        results_YX=pd.concat([results_YX,result_1],axis=0)
    end=time.time()
    
    print("the time taken for"+str(lags)+"lags:",int((end-start)/60),"minutes and",int((end-start)%60),"seconds","for:",gene1+" and "+gene2)
    return(history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2)


# # Making the dataset

# In[14]:

# ellcycledata=pd.read_csv("C:/Users/malli/OneDrive/Documents/hackbio/mouse time series data/processed_timeseries.csv")

# In[129]:


cc=pd.read_csv(sys.argv[1])


# In[15]:

# ata1=cellcycledata.T

# In[16]:

# In[130]:


ccdata=cc.T


# In[17]:

# ata1.columns=data1.iloc[0,:]

# In[131]:


ccdata.columns=ccdata.iloc[0,:]


# In[18]:

# ata1=data1.drop(data1.index[0],axis=0)

# In[132]:


ccdata=ccdata.drop(ccdata.index[0],axis=0)


# In[19]:

# In[133]:


ccdata=ccdata.astype("float32")# this is the cell cycle regulatory gene dataset.
phase1=ccdata.iloc[0:28,:]# the G1 phase till G1-S checkpoint.
phase1_2=ccdata.iloc[0:48,:]#G1 and S phase
phase1_3=ccdata.iloc[0:56,:]#G2 phase


# In[20]:

# In[136]:


count=0
for i in range(0,phase1.shape[1]-1):
    for j in range((i+1),phase1.shape[1]):
        count+=1


# In[21]:

# In[137]:


print(count)


# # This code is to create a class that returns the lag( with least error),pval and RSS and the direction of causality<br>
# 

# In[22]:

# In[62]:


class results:
    tag=0
    
    
    def __init__(self, history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2):
        self.gene1=gene1
        self.gene2=gene2
        self.history_YX=history_YX
        self.history_X= history_X
        self.history_XY=history_XY
        self.history_Y=history_Y
        self.results_YX=results_YX
        self.results_XY=results_XY
        self.lag=0
        self.pval=0
        
    def get_causality(self):
        self.results_XY=self.results_XY[self.results_XY["pval"]<=0.05]
        self.results_YX=self.results_YX[self.results_YX["pval"]<=0.05]
        if self.results_XY.empty and self.results_YX.empty:
            return(self.gene1,self.gene2,self.tag,self.lag,self.pval)
        else:
            results_all=pd.concat([self.results_XY,self.results_YX],ignore_index=True)
            results_all=results_all[results_all["r2_2"]>=0.9]
            if results_all.empty:
                return(self.gene1,self.gene2,self.tag,self.lag,self.pval)
            else:
                print(results_all)
                a=results_all.loc[results_all['S'].idxmax()]
                if(a["tag"]=="XY"):
                    self.tag="XY"
                    self.lag=a["lag"]
                    self.pval=a["pval"]
                else:
                    self.tag="YX"
                    self.lag=a["lag"]
                    self.pval=a["pval"]
                    
                return(self.gene1,self.gene2,self.tag,self.lag,self.pval)
            
            
            
            
        
    
        
    
        
        
        
            
                
            
        
        
    


# In[25]:

# In[ ]:





# In[219]:


count=0
#list1=[]
for i in range(0,phase1.shape[1]-1):
    for j in range((i+1),phase1.shape[1]):
        #tf.keras.backend.clear_session()
        df=phase1.iloc[:,[i,j]]
        print(df.columns)
        history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2=causality(df,df,8,110,70,0.1,2)
        print(count)
        a=results(history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2)
        a.get_causality()
        #print(a.results_YX,"\n",a.results_XY)
        dat=pd.DataFrame([[a.gene1,a.gene2,a.tag,a.lag,a.pval]],columns=["gene1","gene2","tag","lag","pval"])
        #print(dat)
        #list1.append(a)
        dat.to_csv(sys.argv[2],mode="a",header=False,index=False)
        count+=1
        
 

count=0
list1=[]
for i in range(0,phase1_2.shape[1]):
    for j in range((i+1),phase1_2.shape[1]):
        tf.keras.backend.clear_session()
        df=phase1_2.iloc[:,[i,j]]
        print(df.columns)
        history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2=causality(df,df,8,110,80,0.13,2)
        print(count)
        a=results(history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2)
        a.get_causality()
        #print(a.results_YX,"\n",a.results_XY)
        dat=pd.DataFrame([[a.gene1,a.gene2,a.tag,a.lag,a.pval]],columns=["gene1","gene2","tag","lag","pval"])
        #print(dat)
        #list1.append(a)
        dat.to_csv(sys.argv[3],mode="a",header=False,index=False)
        count+=1
#count=0 
#for i in range(0,phase1_3.shape[1]-1):
    #for j in range((i + 1), phase1_3.shape[1]):
     #   tf.keras.backend.clear_session()
      #  df=phase1_3.iloc[:,[i,j]]
      #  print(df.columns)
      #  history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2=causality(df,df,8,110,80,0.16,2)
      #  #print(count)
      #  a=results(history_YX,history_X,history_XY,history_Y,results_YX,results_XY,gene1,gene2)
      #  a.get_causality()
        ##print(a.results_YX,"\n",a.results_XY)
       # dat=pd.DataFrame([[a.gene1,a.gene2,a.tag,a.lag,a.pval]],columns=["gene1","gene2","tag","lag","pval"])
       # print(dat)
        ##list1.append(a)
       # dat.to_csv(sys.argv[4],mode="a",header=False,index=False)
        #count+=1



