#!/usr/bin/env python
# coding: utf-8

# # ``BERCHMANS KEVIN S``
# 

# ## `Data Visualization in Seaborn`

# ### Import necessary modules

# In[1]:


import pandas as pd 
import csv
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import random as randrange


# # `1. Visualizing Statistical Relationships`

# ### Import train_upvote_mini.csv file

# In[2]:


df = pd. read_csv ("train_upvote_mini.csv") 
df.head() 


# ### What is its size?

# In[3]:


df.shape


# ### Show the types of each feature

# In[4]:


df.dtypes 


# ### How many unique "tag" available?

# In[5]:


df. Tag.nunique()


# ### Visualize with Scatterplot

# ### Plot replot between "Views" and "Upvotes"

# In[6]:


sns.relplot(x="Views",y="Upvotes", data = df) ;


# ### Next, we want to see the tag associated with data.
# ### Plot relplot between "Views" and "Upvotes" with hue as "Tag"

# In[7]:


sns. relplot(x="Views", y="Upvotes", hue = "Tag", data = df) ;


# ### Hue Plot

# ### Plot relplot between "Views" and "Upvotes" with hue as "Answers"

# In[8]:


sns. relplot(x="Views", y="Upvotes", hue = "Answers", data = df);


# ### Plot relplot between "Views" and "Upvotes" with size as "Tag"

# In[9]:


sns.relplot(x="Views", y="Upvotes", size = "Tag", data = df);


# ### Does no of times question answered impact the no. of upvotes?
# ### Plot line chart using relplot between "Answers" and "Upvotes"

# In[10]:


sns. relplot (data=df, x='Answers', y='Upvotes', kind='line') ;
plt.show() 


# ### Does Reputation score of question author impact no of upvotes?. Draw replot.

# In[11]:


sns.relplot(data=df, x='Reputation', y='Upvotes') 
plt.show() 


# # `2. Visualizating Categorial Data`
# 
# ### Various Categorial Plots in Seaborn

# # Jitter Plot

# In[12]:


df2 = pd. read_csv ("train_hr_mini.csv") 
df2.head() 


# In[13]:


df2.shape 


# # Show Jitter plot between education and avg_training_score

# In[14]:


sns.catplot(x="education", y="avg_training_score", data=df2) 


# # Sworm Plot

# In[15]:


sns.catplot(x="education", y="avg_training_score", jitter = False, data=df2) 


# # Hue Plot

# In[16]:


import warnings 
warnings.filterwarnings('ignore') 


# In[17]:


sns.catplot(x="education", y="avg_training_score", kind = "swarm", data=df2)


# In[18]:


sns.catplot(x="education", y="avg_training_score", hue = "gender", data=df2) 


# # Box Plot
# ## Draw box plot between education and avg_traing_score

# In[19]:


sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind ="swarm", data=df2)


# # Box Plot with Hue Dimension
# ## Who are promoted and not promoted considering education and avg_training_score? Draw Box Plot.

# In[20]:


sns.catplot(x="education", y="avg_training_score", kind = "box", data=df2) 


# # Violin Plot

# ## Show violin plot between education categories and avg training score with hue as "is_promoted", target variable

# In[21]:


sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind ="box", data=df2) 


# # Draw violin plot with only 2 hue levels, use split attribute

# In[22]:


sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind ="violin", data=df2)


# # Using Catplot()
# ### Draw a Bar Chart between " education " and " avg_training_score ", with hue as " is_promopted "

# In[23]:


sns.catplot(x="education", y="avg_training_score", hue = "is_promoted", kind ="bar", data=df2) 


# # Point Plot
# ### Show point plot between education and avg training score with hue promotion category

# In[24]:


sns.catplot (x="education", y="avg_training_score", hue = "is_promoted", kind ="point", data=df2) 


# # Multiple Dimension in Seaborn
# ### Draw swarm plot for education, avg training score, hue as is_promoted for male and female category

# In[25]:


sns.catplot (x="education", y="avg_training_score", hue="is_promoted", 
             col="gender", aspect=.9,  
             kind="swarm", data=df2); 


# # `3. Visualizing the Distribution of Data`

# # Plot Univariable Distriobutions
# ### Plot Histogram with kernal density estimate value for age attribute

# In[26]:


sns.distplot(df2.age)


# ## Show only Histogram for avg variable, without KDE

# In[27]:


sns. distplot(df2.age, kde=False, rug = True) 


# # Plot Bivariate Distributions

# # Joint Plot
# ### Draw a joint plot between avg_training_score and age

# In[28]:


sns. jointplot(x="avg_training_score", y="age", data=df2); 


# # Hex Plot
# ### Draw a hexplot for depicting the relationship between avg training score and age

# In[29]:


sns.jointplot(x=df2.age, y=df2.avg_training_score, kind="hex", data = df2) 


# ## KDE Plot
# ### show KDE Plot visualize age vs avg training score

# In[30]:


sns. jointplot(x="age", y="avg_training_score", data=df2, kind="kde");


# ### Heat Map
# ### Draw heatmap for the dataset

# In[31]:


corrmat = df2.corr() 
f, ax = plt.subplots (figsize=(9, 6))
sns.heatmap(corrmat, vmax=.8, square=True) 


# ## Boxen Plot

# ### Draw Boxen Plot between "age" and "avg_training_score", with hue"is_promoted"

# In[32]:


sns.catplot(x="age", y="avg_training_score", data=df2, kind="boxen",height=4,aspect=2.7, hue = "is_promoted") 


# ### Draw a Pair Plot for the dataset

# In[33]:


sns.pairplot(df2) 

