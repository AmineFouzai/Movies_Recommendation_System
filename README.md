# Movies_Recommendation_System

# Build A Simple Recommender System 

a recommender system, or a recommendation system (sometimes replacing 'system' with a synonym such as platform or engine), is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications

# DATA Preprocessing


```python
import numpy as np 
import pandas as pd
ratings_data=pd.read_csv(r'C:\Users\TTos\Desktop\dataset\ml-latest-small\ml-latest-small\ratings.csv')
ratings_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
      <td>964981247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
      <td>964982224</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
      <td>964983815</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
      <td>964982931</td>
    </tr>
  </tbody>
</table>
</div>



# Movies.csv


```python
movies_names=pd.read_csv(r'C:\Users\TTos\Desktop\dataset\ml-latest-small\ml-latest-small\movies.csv')
movies_names.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>



# Merging Data from Rtings.csv & Movies.csv


```python
movies_data=pd.merge(ratings_data,movies_names,on="movieId")
movies_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
      <td>964982703</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>4.0</td>
      <td>847434962</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7</td>
      <td>1</td>
      <td>4.5</td>
      <td>1106635946</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>1</td>
      <td>2.5</td>
      <td>1510577970</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>1</td>
      <td>4.5</td>
      <td>1305696483</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
  </tbody>
</table>
</div>



# Groub by Rating & Title


```python
movies_data.groupby('title')['rating'].mean().head()
```




    title
    '71 (2014)                                 4.0
    'Hellboy': The Seeds of Creation (2004)    4.0
    'Round Midnight (1986)                     3.5
    'Salem's Lot (2004)                        5.0
    'Til There Was You (1997)                  4.0
    Name: rating, dtype: float64



# Sort the Average Ratings


```python
movies_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
```




    title
    Karlson Returns (1970)                           5.0
    Winter in Prostokvashino (1984)                  5.0
    My Love (2006)                                   5.0
    Sorority House Massacre II (1990)                5.0
    Winnie the Pooh and the Day of Concern (1972)    5.0
    Name: rating, dtype: float64



# Average Ratings with number of Ratings


```python
ratings_mean_count=pd.DataFrame(movies_data.groupby('title')['rating'].mean())
ratings_mean_count['rating_count']=pd.DataFrame(movies_data.groupby('title')['rating'].count())
ratings_mean_count.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>rating_count</th>
    </tr>
    <tr>
      <th>title</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>'71 (2014)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Hellboy': The Seeds of Creation (2004)</th>
      <td>4.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Round Midnight (1986)</th>
      <td>3.5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>'Salem's Lot (2004)</th>
      <td>5.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>'Til There Was You (1997)</th>
      <td>4.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



# Correlation System

Correlation is a statistical technique that can show whether and how strongly pairs of variables are related. For example, height and weight are related; taller people tend to be heavier than shorter people. The relationship isn't perfect

# Extract Rating_count and Rating 


```python
x=[ float(_) for _ in  ratings_mean_count['rating_count']]
y=[ _ for _ in  ratings_mean_count['rating']]
x=pd.Series(x)
y=pd.Series(y)
```

<html><head>


<!-- Load require.js. Delete this if your page already loads require.js -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" crossorigin="anonymous"></script>
<script src="https://unpkg.com/@jupyter-widgets/html-manager@*/dist/embed-amd.js" crossorigin="anonymous"></script>
<script type="application/vnd.jupyter.widget-state+json">
{
    "version_major": 2,
    "version_minor": 0,
    "state": {}
}
</script>
</head>
    
<body>

<table width="40%" style="margin-left:-50px" >
<thead>
<tr>
<th>Pearson’s r Value</th>
<th>Correlation Between <strong>x</strong> and <strong>y</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>equal to 1</td>
<td>perfect positive linear relationship</td>
</tr>
<tr>
<td>greater than 0</td>
<td>positive correlation</td>
</tr>
<tr>
<td>equal to 0</td>
<td>independent</td>
</tr>
<tr>
<td>less than 0</td>
<td>negative correlation</td>
</tr>
<tr>
<td>equal to -1</td>
<td>perfect negative linear relationship</td>
</tr>
</tbody>
</table>
</body>
</html>



```python
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
```

    {"Kendall's tau": 0.037132866375530676,
     "Pearson's r": 0.12730726667013137,
     "Spearman's rho": 0.0397780088264808}
    

# Recommander function


```python
corrChek=lambda r:False if r <0 else True 
from random import randint
def Recommander():
    if any([corrChek(r),corrChek(r2),corrChek(r3)]) is True:
        return ratings_mean_count['rating'].head()
    else:
        return  ratings_mean_count['rating'][::-randint(0,len(ratings_mean_count['rating']))]
```

# Seek  Recommanded Moives


```python
Recommander()
```




    title
    '71 (2014)                                 4.0
    'Hellboy': The Seeds of Creation (2004)    4.0
    'Round Midnight (1986)                     3.5
    'Salem's Lot (2004)                        5.0
    'Til There Was You (1997)                  4.0
    Name: rating, dtype: float64


