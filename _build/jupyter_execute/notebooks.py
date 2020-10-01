#!/usr/bin/env python
# coding: utf-8

# # Visualization in our everyday workbench

# In[1]:


import pandas as pd


# In[2]:


from typing import List


# In[3]:


column_names: List[str] = ['age', 'workclass', 'fnlwgt' ,'education', 'education-num', 'marital-status',
                           'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                          'hours-per-week', 'native-country', 'income']


# In[4]:


training_data: pd.DataFrame = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", header=None, names=column_names)


# In[5]:


test_data: pd.DataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', header=None, names=column_names, skiprows=[0])


# In[6]:


X_train: pd.DataFrame = training_data.iloc[:,:-1]


# In[7]:


X_test: pd.DataFrame = test_data.iloc[:, :-1]


# ![Data Panel](Data.gif)

# ### AI presented as the harbinger of progress and hence data is for the taking
# 
# Arcade mechanical picker game is depicted, where the toys are replaced by people’s faces. 
# There’s a stark disparity between the abundance of ‘western’ faces and those with darker skin tones. 
# Ironically there is a sign on top of the game saying “diverse, ethical, fair” with the faces of prominent scholars. 
# A Mark- Zuckerburg-type is seen operating the joystick and playing the game. 
# The mechanical hand has the words Efficiency, Advancement, Progress and Innovation on it.

# In[8]:


y_train: pd.Series = training_data.iloc[:, -1]


# In[9]:


y_test: pd.Series = test_data.iloc[:,-1]


# ![Labels](Labels.gif)

# ### People at the margins that do not conform to societal archetypes are the ones most affected by imposed class labels
# 
# A powerful, white male shoots an arrow into a large terrain. 
# The arrow reads “Objectivity”. 
# The ground starts splitting into 4-5 different land masses and people at the boundary start to fall into the abyss below.
