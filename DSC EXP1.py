#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import warnings
warnings.simplefilter("ignore")


# In[3]:


df=pd.read_csv("D:\\New folder\\iris (1).csv")
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


x=df.iloc[:,:-1]
x


# In[8]:


y=df.iloc[:,-1]
y


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)


# In[10]:


x_train.shape


# In[11]:


x_test.shape


# In[12]:


y_train.shape


# In[13]:


y_test.shape


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)


# In[15]:


model.fit(x_train,y_train)


# In[16]:


y_pred=model.predict(x_test)
y_pred


# In[17]:


from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test,y_pred)


# In[18]:


accuracy=accuracy_score(y_test,y_pred)*100
print("Accuracy of the model is {:.2f}".format(accuracy))


# In[19]:


from sklearn.metrics import classification_report
class_report = classification_report(y_test, y_pred)
print(f"\nClassification Report:\n{class_report}")


# In[20]:


new_flower = [[5.1, 3.5, 1.4, 0.2]]  
predicted_class = model.predict(new_flower)
predicted_class


# In[ ]:




