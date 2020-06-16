#!/usr/bin/env python
# coding: utf-8

# # Build A Simple Recommender System 

# a recommender system, or a recommendation system (sometimes replacing 'system' with a synonym such as platform or engine), is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications

# # DATA Preprocessing

# In[1]:


import numpy as np 
import pandas as pd
ratings_data=pd.read_csv(r'C:\Users\TTos\Desktop\dataset\ml-latest-small\ml-latest-small\ratings.csv')
ratings_data.head()


# # Movies.csv

# In[2]:


movies_names=pd.read_csv(r'C:\Users\TTos\Desktop\dataset\ml-latest-small\ml-latest-small\movies.csv')
movies_names.head()


# # Merging Data from Rtings.csv & Movies.csv

# In[3]:


movies_data=pd.merge(ratings_data,movies_names,on="movieId")
movies_data.head()


# # Groub by Rating & Title

# In[4]:


movies_data.groupby('title')['rating'].mean().head()


# # Sort the Average Ratings

# In[5]:


movies_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# # Average Ratings with number of Ratings

# In[6]:


ratings_mean_count=pd.DataFrame(movies_data.groupby('title')['rating'].mean())
ratings_mean_count['rating_count']=pd.DataFrame(movies_data.groupby('title')['rating'].count())
ratings_mean_count.head()


# # Correlation System

# Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. For example, height and weight are related; taller people tend to be heavier than shorter people. The relationship isn't perfect

# # Extract Rating_count and Rating 

# In[7]:


x=[ float(_) for _ in  ratings_mean_count['rating_count']]
y=[ _ for _ in  ratings_mean_count['rating']]
x=pd.Series(x)
y=pd.Series(y)


# <html><head>
# 
# 
# <!-- Load require.js. Delete this if your page already loads require.js -->
# <script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" crossorigin="anonymous"></script>
# <script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>
# <script type="application/vnd.jupyter.widget-state+json">
# {
#     "version_major": 2,
#     "version_minor": 0,
#     "state": {}
# }
# </script>
# </head>
#     
# <body>
# 
# <table width="40%" style="margin-left:-50px" >
# <thead>
# <tr>
# <th>Pearson’s r Value</th>
# <th>Correlation Between <strong>x</strong> and <strong>y</strong></th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>equal to 1</td>
# <td>perfect positive linear relationship</td>
# </tr>
# <tr>
# <td>greater than 0</td>
# <td>positive correlation</td>
# </tr>
# <tr>
# <td>equal to 0</td>
# <td>independent</td>
# </tr>
# <tr>
# <td>less than 0</td>
# <td>negative correlation</td>
# </tr>
# <tr>
# <td>equal to -1</td>
# <td>perfect negative linear relationship</td>
# </tr>
# </tbody>
# </table>
# </body>
# </html>
# 

# In[8]:


r=x.corr(y)
r2=x.corr(y, method='spearman')
r3=x.corr(y, method='kendall')
from pprint import pprint
pprint(
    {
       "Pearson's r":r,
       "Spearman's rho":r2,
       "Kendall's tau":r3
    })


# # Recommander function

# In[9]:


corrChek=lambda r:False if r <0 else True 
from random import randint
def Recommander():
    if any([corrChek(r),corrChek(r2),corrChek(r3)]) is True:
        return ratings_mean_count['rating'].head()
    else:
        return  ratings_mean_count['rating'][::-randint(0,len(ratings_mean_count['rating']))]


# # Seek  Recommanded Moives

# In[11]:


Recommander()

