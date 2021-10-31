#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('world Happiness report  2019.csv')
df


# In[4]:


(df[0:10])


# In[5]:


df.tail()


# In[6]:


(df[10:51])


# In[7]:


(df[51:91])


# In[8]:


(df[91:126])


# In[9]:


(df[137:145])


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


from sklearn import linear_model
X = df['Overall rank'].values
Y = df['Score'].values


# In[13]:


#mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)

#total number of values 
m = len(X)

#using the formula to clacualte b1 and b0
numer = 0
denom = 0
for i in range(m):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
b1 = numer / denom
b0 = mean_y - (b1 * mean_x)

#print coefficients
print(b1,b0)


# In[14]:


#plotting  values and regression line

max_x = np.max(X) + 100
min_x = np.min(X) - 100

#calulating line values x and y
x = np.linspace(min_x,max_x, 1000)
y = b0+b1 * x

#ploting line
plt.plot(x,y,color="#58b970", label='Regression Line')
#ploting Scatter points
plt.scatter(X,Y, c ='#ef5423', label ='Scatter Plot')

plt.xlabel('Overall rank')
plt.ylabel('Score')
plt.legend()
plt.show()


# In[15]:


df.shape


# In[16]:


###accuracy###
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_y) **2
    ss_r += (Y[i] - y_pred) **2
    
r2 = 1 - (ss_r/ss_t)
print(r2)


# In[17]:


df.drop(['Generosity','Healthy life expectancy','Social support','Perceptions of corruption'],axis=1,inplace=True)


# In[18]:


df.head()

