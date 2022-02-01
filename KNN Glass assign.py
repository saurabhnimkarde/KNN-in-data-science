#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# In[2]:


glass=pd.read_csv("glass.csv")


# In[3]:


glass.head()


# In[4]:


glass.shape


# In[5]:


glass.info()


# In[6]:


X=glass.values[:,0:9]
Y=glass.values[:,9]


# In[7]:


X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)


# In[8]:


knn.score(X_train,y_train)


# In[9]:


knn.score(X_test,y_test)


# In[13]:


#We Obtained 66 % accuracy by using KNN


# In[10]:


preds=knn.predict(X_test)


# In[11]:


preds


# In[12]:


from sklearn.metrics import classification_report


# In[16]:


print(classification_report(preds,y_test))


# In[ ]:




