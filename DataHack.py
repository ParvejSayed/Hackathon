#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np


# In[69]:


training_data=pd.read_csv('training_set_features.csv')
test_data=pd.read_csv('test_set_features.csv')
training_target=pd.read_csv('training_set_labels.csv')


# In[70]:


training_data.drop(columns=["respondent_id"],inplace=True)
training_target.drop(columns=["respondent_id"],inplace=True)
test_data.drop(columns=["respondent_id"],inplace=True)


# In[71]:


training_data.head(5)


# In[72]:


training_target.sample(5)


# In[73]:


training_data.drop(columns=["employment_occupation",'rent_or_own',"health_insurance","employment_industry"], inplace=True)
test_data.drop(columns=["employment_occupation",'rent_or_own',"health_insurance","employment_industry"], inplace=True)


# In[74]:


mean_childs_1=training_data["household_children"].mean()
mean_childs_2=test_data["household_children"].mean()
training_data["household_children"].fillna(mean_childs_1,inplace=True)
test_data["household_children"].fillna(mean_childs_2,inplace=True)


# In[75]:


mean_adults_1=training_data["household_children"].mean()
mean_adults_2=test_data["household_children"].mean()
training_data["household_children"].fillna(mean_adults_1,inplace=True)
test_data["household_children"].fillna(mean_adults_2,inplace=True)


# In[76]:


for col in training_data.columns:
  if training_data[col].isnull().sum()>0:
      mode_1=training_data[col].mode()[0]
      training_data[col].fillna(mode_1,inplace=True)


for col in test_data.columns:
  if test_data[col].isnull().sum()>0:
      mode_2=test_data[col].mode()[0]
      test_data[col].fillna(mode_2,inplace=True)


# In[77]:


for col in training_data.columns:
  print(f"Column name is :{col}")
  print(training_data[col].isnull().mean())


for col in test_data.columns:
  print(f"Column name is :{col}")
  print(test_data[col].isnull().mean())


# In[78]:


df=pd.concat([training_data,training_target],axis=1)


# In[79]:


X=df.drop(columns=["xyz_vaccine","seasonal_vaccine"])
X.columns


# In[80]:


Y_1=df["xyz_vaccine"]
Y_2=df["seasonal_vaccine"]


# In[81]:


#Now we are doing one hot encoding for categorical columns of the dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# for XYZ Vacine
X_train, X_test, y_train, y_test = train_test_split(X, Y_1, test_size=0.2, random_state=42)


# In[82]:


ohe=OneHotEncoder(drop="first")


# In[83]:


X_train_new=ohe.fit_transform(X_train[['age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
        'employment_status', 'hhs_geo_region', 'census_msa']]).toarray()

X_test_new=ohe.transform(X_test[['age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
        'employment_status', 'hhs_geo_region', 'census_msa']]).toarray()


# In[84]:


from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(X_train_new,y_train)


# In[85]:


y_pred_proba_1=GNB.predict_proba(X_test_new)[:,1]
y_pred_1=GNB.predict(X_test_new)


# In[86]:


# For seasonal_vaccine
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X, Y_2, test_size=0.2, random_state=42)


# In[87]:


X_train_new_1=ohe.fit_transform(X_train_1[['age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
        'employment_status', 'hhs_geo_region', 'census_msa']]).toarray()

X_test_new_1=ohe.transform(X_test_1[['age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
        'employment_status', 'hhs_geo_region', 'census_msa']]).toarray()


# In[88]:


GNB1=GaussianNB()

GNB1.fit(X_train_new_1,y_train_1)
y_pred_proba_2=GNB1.predict_proba(X_test_new_1)[:,1]
y_pred_2=GNB1.predict(X_test_new_1)


# In[89]:


from sklearn.metrics import accuracy_score
print(X_test_new.shape)
print(y_test.shape)
print(y_pred_1.shape)


# In[90]:


y_test_1


# In[91]:


y_pred_1


# In[92]:


y_pred_2


# In[93]:


accuracy_score(y_test,y_pred_1)


# In[94]:


accuracy_score(y_test_1,y_pred_2)


# In[95]:


test_data=ohe.transform(test_data[['age_group',
       'education', 'race', 'sex', 'income_poverty', 'marital_status',
       'employment_status', 'hhs_geo_region', 'census_msa']]).toarray()


xyz_prob=GNB.predict_proba(test_data)[:,1]
seasonal_prob=GNB1.predict_proba(test_data)[:,1]


# In[96]:


new=pd.read_csv("test_set_features.csv")


# In[97]:


submission=pd.DataFrame({"respondent_id":new["respondent_id"],"xyz_vaccine":xyz_prob,"seasonal_vaccine":seasonal_prob})


# In[98]:


submission


# In[101]:


submission.to_csv("submission_parvej.csv",index=False)


# In[102]:


new


# In[ ]:





# In[ ]:




