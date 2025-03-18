#!/usr/bin/env python
# coding: utf-8

#  # Titanic Survival Prediction Using Machine Learning

# Importing Libraries

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


data_titanic = pd.read_csv(r"C:\Users\Shadab Momin\Downloads\Titanic-Dataset.csv")
print(data_titanic.head())


# In[6]:


data_titanic.info()


# In[7]:


data_titanic.isnull().sum()


# In[8]:


del data_titanic['Cabin']


# In[11]:


data_titanic


# In[12]:


data_titanic.describe()


# ## now replacing missing values in "age" with mean value 

# In[13]:


data_titanic["Age"].fillna(data_titanic["Age"].mean(), inplace=True)
data_titanic.describe()


# In[ ]:


##finding the mode value of embarked column


# In[14]:


print(data_titanic["Embarked"].mode())


# In[15]:


print(data_titanic["Embarked"].mode()[0])


# In[16]:


data_titanic["Embarked"].fillna(data_titanic["Embarked"].mode()[0], inplace=True)


# In[17]:


data_titanic.isnull().sum()


# 
# Exploratory Data Analysis

# In[18]:


data_titanic["Survived"].value_counts()


# In[20]:


sns.set()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Survived', data=data_titanic)
plt.show()


# In[26]:


sns.countplot(x='Sex', data=data_titanic)  # Correct syntax
plt.show()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Pclass', data=data_titanic)  # Correct syntax
plt.show()


# In[31]:


sns.countplot('Embarked', data=data_titanic)


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Embarked', data=data_titanic)  # Correct syntax
plt.show()


# Checking numerical attributes

# In[33]:


sns.distplot(data_titanic['Age'])


# In[34]:


#checking for Fare column
sns.distplot(data_titanic['Fare'])


# In[37]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_titanic.fillna(0, inplace=True)  

# Correlation heatmap
corr = data_titanic.corr(numeric_only=True) 
plt.figure(figsize=(15, 9))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()



# In[38]:


data_titanic.head()


# In[44]:


## drop unnecessary columns
data_titanic = data_titanic.drop(columns=['Name', 'Ticket'], axis=1)
data_titanic.head()


# Encoding Label

# In[45]:


#Categorical to Numerical for further modelling
data_titanic["Sex"].value_counts()


# In[46]:


data_titanic['Embarked'].value_counts()


# In[47]:


from sklearn.preprocessing import LabelEncoder
cols = ['Sex', 'Embarked']
le = LabelEncoder()

for col in cols:
    data_titanic[col] = le.fit_transform(data_titanic[col])
data_titanic.head()


# Train_Test_Split

# In[48]:


X = data_titanic.drop(columns = ['PassengerId','Survived'],axis=1)
Y = data_titanic['Survived']


# In[49]:


print(X)


# In[50]:


print(Y)


# In[51]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[52]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# In[53]:


#Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[54]:


model = LogisticRegression()


# In[55]:


data_titanic.info()


# In[56]:


data_titanic.astype({'Age':'int','Fare':'int'}).dtypes


# In[57]:


#training the Logistic Regression model with training data
model.fit(X_train, Y_train)


# In[58]:


#accuracy on training data
X_train_prediction = model.predict(X_train)


# In[59]:


print(X_train_prediction)


# In[60]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy_score_of_training_data : ', training_data_accuracy)


# In[61]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[62]:


print(X_test_prediction)


# In[63]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy_score_of_test data : ', test_data_accuracy)


# In[ ]:





# # Accuracy_score_of_test data :  0.7877094972067039

# In[ ]:





# In[ ]:





# The end Project______________________________________

# In[ ]:




