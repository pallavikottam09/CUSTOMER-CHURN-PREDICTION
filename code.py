#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
#from google.colab import drive
import matplotlib.pyplot as plt
import keras
import re
import os
from keras.models import load_model
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[3]:


#dataset_path= '/content/drive/MyDrive/churn/Churn_Modelling.csv'

df= pd.read_csv("Churn_Modelling.csv")

df .head()


# In[4]:


df.isnull().sum()


# In[5]:


df.dtypes


# In[6]:


df.duplicated().sum()


# In[7]:


count = df['Exited']
count.value_counts().plot(kind="bar",figsize=(15,10),color='red')
count.value_counts()


# In[8]:


df['Geography'].value_counts().plot(kind="bar",figsize=(15,10),color='skyblue')
df['Geography'].value_counts()


# In[9]:


df['Gender'].value_counts().plot(kind="bar",figsize=(15,10),color='skyblue')
df['Gender'].value_counts()


# In[10]:


df['Age'].value_counts().plot(kind="bar",figsize=(15,10),color='blue')
df['Age'].value_counts()


# In[11]:


df['IsActiveMember'].value_counts().plot(kind="bar",figsize=(15,10),color='pink')
df['IsActiveMember'].value_counts()


# In[12]:


df.drop(columns=['RowNumber','CustomerId','Surname'], inplace= True)
df


# In[13]:


df.dtypes


# In[14]:


label_encoder = LabelEncoder()
columns_to_convert_in_string_to_int = ['Geography','Gender']
for col in columns_to_convert_in_string_to_int:
    df[col] = label_encoder.fit_transform(df[col])
columns_to_convert_in_int_to_float = ['CreditScore','Geography','Gender', 'Age','Tenure','NumOfProducts','HasCrCard','IsActiveMember']
for col in columns_to_convert_in_int_to_float:
    df[col] = df[col].astype('float64')


# In[15]:


df.dtypes


# In[16]:


df.head()


# In[17]:


X = df.drop(columns=['Exited'])
Y = df['Exited']


# In[18]:


X


# In[19]:


Y


# In[20]:


scaler = StandardScaler()
X= scaler.fit_transform(X)


# In[22]:


X


# In[23]:


X_train, X_test , Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[24]:


log_reg = LogisticRegression()
log_reg.fit(X_train,Y_train)


# In[26]:


y_pred_log_reg = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred_log_reg)}")
print(classification_report(Y_test, y_pred_log_reg))


# In[25]:


random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train,Y_train)


# In[27]:


y_pred_forest = random_forest.predict(X_test)
print(f"Accuracy: {accuracy_score(Y_test, y_pred_forest)}")
print(classification_report(Y_test, y_pred_forest))


# In[28]:


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Exited', 'Exited'], yticklabels=['Not Exited', 'Exited'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


# In[29]:


plot_confusion_matrix(Y_test, y_pred_log_reg)


# In[30]:


plot_confusion_matrix(Y_test, y_pred_forest)


# In[ ]:




