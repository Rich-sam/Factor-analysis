#!/usr/bin/env python
# coding: utf-8

# # Name:Kimaita bundi 19/04156
#     Msc Data analytics 
#     MDA:5404 Data analytics and Knowledge Engineering

# source of data:
#     https://www.kaggle.com/ajrwhite/covid-19-risk-factors

# Factor Analysis is a Data Reduction Technique which helps in reducing the features by identifying variables that are highly correlated to each other. In a way, factor analysis is kind of a classification technique as it groups the features by extracting the maximum common variance from all the variables. 

# # Step1:.  Evaluate factorability!

# In[23]:


#import libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt


# In[24]:


#step1(b)Load data
data=pd.read_csv("D:\Data visualization\Dr Mwendia_Class\Covid_factor Analysis.csv")
#a way of listing attributes
attributes=data.columns


# In[25]:


#Printing attributes helps in indicating variables of interest
print(attributes)


# In[4]:


print(data)


# In[34]:


#Check the correlation of the variables
data.corr()


# In[5]:


attributes=data.columns


# In[6]:


#Determine Factorability
#use Barlett_sphericity Test
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(data)
chi_square_value,p_value


# our P_value=0 is less than 0.05. Therefore the test was statistically significant, indicating that the observed correlation matrix is not an identity matrix and Factor analysis can be conducted.
# 

# In[7]:


#Determine Factorability
#perform KMD test
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(data)
print('\nKMO Model',kmo_model)


# The overall KMO for our data is 0.73, which is meritorious 
# This value indicates that we can proceed with your planned factor analysis.
# 

# # Step 2: Extract  Factors!

# Principal  Component Analysis (PCA)  involves  compressing  and reducing all the relationships in the data into  newly created features called components.  Each of these components represents  a factor

# In[17]:


#Extract the factors
factor=FactorAnalysis().fit(data)
Factors=pd.DataFrame(factor.components_,columns=attributes)
print(Factors)


# At the intersection of each factor and feature, a positive number indicates that a positive relationship  exists between the two;A negative number, instead, points out that they  are negatively related .The relationship of each variable to the underlying factor is expressed by the so-called factor loading. 

# # Step 3: Choose Factors

# In[38]:


#choose Factors
#Using Kaiser criterion_Get eigen values

fa=FactorAnalyzer(rotation=None)
fa.fit(data)


#check Eigenvalues
ev,v=fa.get_eigenvalues()
ev


# In[39]:


print(v)


# In[46]:


pd.DataFrame.from_records(fa.loadings_)


# Kaiser  criterion involves choosing   eigen value   > 1.
# factors with a variance less than 1  are discarded because it is an indicator that  are no better than a single variable
# 
# in our example we can use 3 factors 

# In[41]:


plt.scatter(range(1,data.shape[1]+1),ev)
plt.plot(range(1,data.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# Factors are extracted from the point the curve  first begin to straighten out. in our case is at 5.
# 

# # 4rd Step: Rotate Factors
# 

# In[19]:


#rotate factors
#generate totated factors factors matrix
fa=FactorAnalyzer(5,rotation='varimax')
fa.fit(data)
loads=fa.loadings_
print(loads)


# In[47]:


#change the array to list
pd.DataFrame.from_records(loads)


# 
