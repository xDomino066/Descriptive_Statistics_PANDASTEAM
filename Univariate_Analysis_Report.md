# Univariate Analysis Report - Airbnb dataset

In this notebook we continue the Airbnb case study and prepare a simple univariate analysis report. The structure is inspired by Exercise 8 from the `pandas_exercises` repository.

Univariate analysis means that we look at one variable at a time. We describe its center, spread, shape, outliers, and basic visual distribution.

## 1. Import libraries and read the data

We use pandas for data work, seaborn and matplotlib for plots, and scipy for a few descriptive statistics.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from IPython.display import display

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', 50)
```


```python
airbnb = pd.read_csv('data/airbnb.csv', index_col='Unnamed: 0')

airbnb.head()
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
      <th>listing_id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>neighbourhood_full</th>
      <th>coordinates</th>
      <th>room_type</th>
      <th>price</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
      <th>rating</th>
      <th>number_of_stays</th>
      <th>5_stars</th>
      <th>listing_added</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>13740704</td>
      <td>Cozy,budget friendly, cable inc, private entra...</td>
      <td>20583125</td>
      <td>Michel</td>
      <td>Brooklyn, Flatlands</td>
      <td>(40.63222, -73.93398)</td>
      <td>Private room</td>
      <td>45$</td>
      <td>10</td>
      <td>2018-12-12</td>
      <td>0.70</td>
      <td>85</td>
      <td>4.100954</td>
      <td>12.0</td>
      <td>0.609432</td>
      <td>2018-06-08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22005115</td>
      <td>Two floor apartment near Central Park</td>
      <td>82746113</td>
      <td>Cecilia</td>
      <td>Manhattan, Upper West Side</td>
      <td>(40.78761, -73.96862)</td>
      <td>Entire home/apt</td>
      <td>135$</td>
      <td>1</td>
      <td>2019-06-30</td>
      <td>1.00</td>
      <td>145</td>
      <td>3.367600</td>
      <td>1.2</td>
      <td>0.746135</td>
      <td>2018-12-25</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21667615</td>
      <td>Beautiful 1BR in Brooklyn Heights</td>
      <td>78251</td>
      <td>Leslie</td>
      <td>Brooklyn, Brooklyn Heights</td>
      <td>(40.7007, -73.99517)</td>
      <td>Entire home/apt</td>
      <td>150$</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>65</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2018-08-15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6425850</td>
      <td>Spacious, charming studio</td>
      <td>32715865</td>
      <td>Yelena</td>
      <td>Manhattan, Upper West Side</td>
      <td>(40.79169, -73.97498)</td>
      <td>Entire home/apt</td>
      <td>86$</td>
      <td>5</td>
      <td>2017-09-23</td>
      <td>0.13</td>
      <td>0</td>
      <td>4.763203</td>
      <td>6.0</td>
      <td>0.769947</td>
      <td>2017-03-20</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22986519</td>
      <td>Bedroom on the lively Lower East Side</td>
      <td>154262349</td>
      <td>Brooke</td>
      <td>Manhattan, Lower East Side</td>
      <td>(40.71884, -73.98354)</td>
      <td>Private room</td>
      <td>160$</td>
      <td>23</td>
      <td>2019-06-12</td>
      <td>2.29</td>
      <td>102</td>
      <td>3.822591</td>
      <td>27.6</td>
      <td>0.649383</td>
      <td>2020-10-23</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Short data preparation

Before the analysis, we make a clean working copy of the Airbnb data. This is a lighter version of the cleaning from the previous report, only enough to make the variables usable for univariate analysis.


```python
df_clean = airbnb.copy()

# price to numeric
df_clean['price'] = df_clean['price'].str.strip('$').astype(float)
df_clean.loc[df_clean['price'] <= 0, 'price'] = np.nan

# dates
df_clean['listing_added'] = pd.to_datetime(df_clean['listing_added'], format='%Y-%m-%d')
df_clean['last_review'] = pd.to_datetime(df_clean['last_review'], format='%Y-%m-%d')

# room type categories
df_clean['room_type'] = df_clean['room_type'].str.lower().str.strip()
room_type_mapping = {
    'private room': 'Private Room',
    'private': 'Private Room',
    'entire home/apt': 'Entire place',
    'home': 'Entire place',
    'shared room': 'Shared Room'
}
df_clean['room_type'] = df_clean['room_type'].replace(room_type_mapping)

# borough and neighbourhood
neighbourhood_split = df_clean['neighbourhood_full'].str.split(',', expand=True)
df_clean['borough'] = neighbourhood_split[0].str.strip()
df_clean['neighbourhood'] = neighbourhood_split[1].str.strip()
df_clean = df_clean.drop(columns='neighbourhood_full')

# rating range
df_clean.loc[df_clean['rating'] > 5, 'rating'] = 5

# missing values connected with no reviews
df_clean = df_clean.fillna({
    'reviews_per_month': 0,
    'number_of_stays': 0,
    '5_stars': 0
})
df_clean['is_rated'] = np.where(df_clean['rating'].isna(), 0, 1)

# missing price by room type median
median_price_by_room_type = df_clean.groupby('room_type')['price'].median()
df_clean['price'] = df_clean['price'].fillna(df_clean['room_type'].map(median_price_by_room_type))

# small text gaps
df_clean['name'] = df_clean['name'].fillna('Unknown listing')
df_clean['host_name'] = df_clean['host_name'].fillna('Unknown host')

# inconsistent dates and duplicates
inconsistent_dates = df_clean[df_clean['listing_added'].dt.date > df_clean['last_review'].dt.date]
df_clean = df_clean.drop(inconsistent_dates.index)
df_clean = df_clean.drop_duplicates()

aggregation_rules = {column: 'first' for column in df_clean.columns if column != 'listing_id'}
aggregation_rules.update({
    'price': 'mean',
    'rating': 'mean',
    'listing_added': 'max',
    'last_review': 'max',
    'number_of_reviews': 'max',
    'reviews_per_month': 'mean',
    'number_of_stays': 'mean',
    '5_stars': 'mean',
    'availability_365': 'max',
    'is_rated': 'max'
})
df_clean = df_clean.groupby('listing_id', as_index=False).agg(aggregation_rules)

print('Original shape:', airbnb.shape)
print('Prepared shape:', df_clean.shape)
df_clean.head()
```

    Original shape: (10019, 16)
    Prepared shape: (9993, 18)





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
      <th>listing_id</th>
      <th>name</th>
      <th>host_id</th>
      <th>host_name</th>
      <th>coordinates</th>
      <th>room_type</th>
      <th>price</th>
      <th>number_of_reviews</th>
      <th>last_review</th>
      <th>reviews_per_month</th>
      <th>availability_365</th>
      <th>rating</th>
      <th>number_of_stays</th>
      <th>5_stars</th>
      <th>listing_added</th>
      <th>borough</th>
      <th>neighbourhood</th>
      <th>is_rated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3831</td>
      <td>Cozy Entire Floor of Brownstone</td>
      <td>4869</td>
      <td>LisaRoxanne</td>
      <td>(40.68514, -73.95976)</td>
      <td>Entire place</td>
      <td>89.0</td>
      <td>270</td>
      <td>2019-07-05</td>
      <td>4.64</td>
      <td>194</td>
      <td>3.273935</td>
      <td>324.0</td>
      <td>0.757366</td>
      <td>2018-12-30</td>
      <td>Brooklyn</td>
      <td>Clinton Hill</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6848</td>
      <td>Only 2 stops to Manhattan studio</td>
      <td>15991</td>
      <td>Allen &amp; Irina</td>
      <td>(40.70837, -73.95352)</td>
      <td>Entire place</td>
      <td>140.0</td>
      <td>148</td>
      <td>2019-06-29</td>
      <td>1.20</td>
      <td>46</td>
      <td>3.495760</td>
      <td>177.6</td>
      <td>0.789743</td>
      <td>2018-12-24</td>
      <td>Brooklyn</td>
      <td>Williamsburg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7322</td>
      <td>Chelsea Perfect</td>
      <td>18946</td>
      <td>Doti</td>
      <td>(40.74192, -73.99501)</td>
      <td>Private Room</td>
      <td>140.0</td>
      <td>260</td>
      <td>2019-07-01</td>
      <td>2.12</td>
      <td>12</td>
      <td>4.389051</td>
      <td>312.0</td>
      <td>0.669873</td>
      <td>2018-12-26</td>
      <td>Manhattan</td>
      <td>Chelsea</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7726</td>
      <td>Hip Historic Brownstone Apartment with Backyard</td>
      <td>20950</td>
      <td>Adam And Charity</td>
      <td>(40.67592, -73.94694)</td>
      <td>Entire place</td>
      <td>99.0</td>
      <td>53</td>
      <td>2019-06-22</td>
      <td>4.44</td>
      <td>21</td>
      <td>3.305382</td>
      <td>63.6</td>
      <td>0.640251</td>
      <td>2018-12-17</td>
      <td>Brooklyn</td>
      <td>Crown Heights</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12303</td>
      <td>1bdr w private bath. in lofty apt</td>
      <td>47618</td>
      <td>Yolande</td>
      <td>(40.69673, -73.97584)</td>
      <td>Private Room</td>
      <td>120.0</td>
      <td>25</td>
      <td>2018-09-30</td>
      <td>0.23</td>
      <td>311</td>
      <td>4.568745</td>
      <td>30.0</td>
      <td>0.918593</td>
      <td>2018-03-27</td>
      <td>Brooklyn</td>
      <td>Fort Greene</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## 3. Variables selected for univariate analysis

For the numerical part we focus on variables that are easy to interpret:

- `price` - nightly price,
- `rating` - listing rating from 0 to 5,
- `number_of_reviews` - number of reviews,
- `availability_365` - days available in a year,
- `reviews_per_month` - review frequency.

For categorical variables we check `room_type` and `borough`.


```python
numeric_columns = ['price', 'rating', 'number_of_reviews', 'availability_365', 'reviews_per_month']
categorical_columns = ['room_type', 'borough']

analysis_data = df_clean[numeric_columns + categorical_columns].copy()
analysis_data.head()
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
      <th>price</th>
      <th>rating</th>
      <th>number_of_reviews</th>
      <th>availability_365</th>
      <th>reviews_per_month</th>
      <th>room_type</th>
      <th>borough</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>89.0</td>
      <td>3.273935</td>
      <td>270</td>
      <td>194</td>
      <td>4.64</td>
      <td>Entire place</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>1</th>
      <td>140.0</td>
      <td>3.495760</td>
      <td>148</td>
      <td>46</td>
      <td>1.20</td>
      <td>Entire place</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>2</th>
      <td>140.0</td>
      <td>4.389051</td>
      <td>260</td>
      <td>12</td>
      <td>2.12</td>
      <td>Private Room</td>
      <td>Manhattan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>99.0</td>
      <td>3.305382</td>
      <td>53</td>
      <td>21</td>
      <td>4.44</td>
      <td>Entire place</td>
      <td>Brooklyn</td>
    </tr>
    <tr>
      <th>4</th>
      <td>120.0</td>
      <td>4.568745</td>
      <td>25</td>
      <td>311</td>
      <td>0.23</td>
      <td>Private Room</td>
      <td>Brooklyn</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Summary statistics

The first step in univariate analysis is a summary table. We look at count, mean, standard deviation, minimum, quartiles, median, and maximum.


```python
summary_basic = analysis_data[numeric_columns].describe().T.round(2)
summary_basic
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>9993.0</td>
      <td>149.73</td>
      <td>202.51</td>
      <td>10.0</td>
      <td>70.00</td>
      <td>105.00</td>
      <td>175.00</td>
      <td>8000.00</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>7922.0</td>
      <td>4.01</td>
      <td>0.57</td>
      <td>3.0</td>
      <td>3.52</td>
      <td>4.03</td>
      <td>4.51</td>
      <td>5.00</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>9993.0</td>
      <td>22.47</td>
      <td>43.20</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>22.00</td>
      <td>510.00</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>9993.0</td>
      <td>112.30</td>
      <td>131.65</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>44.00</td>
      <td>226.00</td>
      <td>365.00</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>9993.0</td>
      <td>1.07</td>
      <td>1.54</td>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.37</td>
      <td>1.55</td>
      <td>16.22</td>
    </tr>
  </tbody>
</table>
</div>



We add more statistics from Exercise 8: variance, IQR, skewness, kurtosis, and coefficient of variation.


```python
summary_extra = pd.DataFrame(index=numeric_columns)

summary_extra['variance'] = analysis_data[numeric_columns].var()
summary_extra['iqr'] = analysis_data[numeric_columns].apply(lambda x: stats.iqr(x.dropna()))
summary_extra['skewness'] = analysis_data[numeric_columns].skew()
summary_extra['kurtosis'] = analysis_data[numeric_columns].kurt()
summary_extra['cv_percent'] = analysis_data[numeric_columns].std() / analysis_data[numeric_columns].mean() * 100

summary_extra.round(2)
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
      <th>variance</th>
      <th>iqr</th>
      <th>skewness</th>
      <th>kurtosis</th>
      <th>cv_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>41011.52</td>
      <td>105.00</td>
      <td>14.71</td>
      <td>380.61</td>
      <td>135.26</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>0.33</td>
      <td>1.00</td>
      <td>-0.04</td>
      <td>-1.19</td>
      <td>14.32</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>1866.35</td>
      <td>21.00</td>
      <td>3.63</td>
      <td>17.85</td>
      <td>192.23</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>17331.74</td>
      <td>226.00</td>
      <td>0.77</td>
      <td>-0.98</td>
      <td>117.23</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>2.37</td>
      <td>1.51</td>
      <td>2.26</td>
      <td>6.88</td>
      <td>143.46</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Measures of central tendency

Here we compare mean, median, and mode. For skewed variables, the median is often more useful than the mean.


```python
central_tendency = pd.DataFrame(index=numeric_columns)
central_tendency['mean'] = analysis_data[numeric_columns].mean()
central_tendency['median'] = analysis_data[numeric_columns].median()
central_tendency['mode'] = analysis_data[numeric_columns].mode().iloc[0]

central_tendency.round(2)
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
      <th>mean</th>
      <th>median</th>
      <th>mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>149.73</td>
      <td>105.00</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>4.01</td>
      <td>4.03</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>22.47</td>
      <td>5.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>112.30</td>
      <td>44.00</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>1.07</td>
      <td>0.37</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The mean price is higher than the median price. This suggests that expensive listings pull the average upward.

## 6. Quantiles and IQR

Quantiles divide ordered data into parts. The first quartile (Q1) means that 25% of observations are below this value. The third quartile (Q3) means that 75% of observations are below this value.


```python
quantile_table = pd.DataFrame(index=numeric_columns)
quantile_table['min'] = analysis_data[numeric_columns].min()
quantile_table['q1'] = analysis_data[numeric_columns].quantile(0.25)
quantile_table['median'] = analysis_data[numeric_columns].quantile(0.50)
quantile_table['q3'] = analysis_data[numeric_columns].quantile(0.75)
quantile_table['max'] = analysis_data[numeric_columns].max()
quantile_table['iqr'] = quantile_table['q3'] - quantile_table['q1']

quantile_table.round(2)
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
      <th>min</th>
      <th>q1</th>
      <th>median</th>
      <th>q3</th>
      <th>max</th>
      <th>iqr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>10.0</td>
      <td>70.00</td>
      <td>105.00</td>
      <td>175.00</td>
      <td>8000.00</td>
      <td>105.00</td>
    </tr>
    <tr>
      <th>rating</th>
      <td>3.0</td>
      <td>3.52</td>
      <td>4.03</td>
      <td>4.51</td>
      <td>5.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>number_of_reviews</th>
      <td>0.0</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>22.00</td>
      <td>510.00</td>
      <td>21.00</td>
    </tr>
    <tr>
      <th>availability_365</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>44.00</td>
      <td>226.00</td>
      <td>365.00</td>
      <td>226.00</td>
    </tr>
    <tr>
      <th>reviews_per_month</th>
      <td>0.0</td>
      <td>0.04</td>
      <td>0.37</td>
      <td>1.55</td>
      <td>16.22</td>
      <td>1.51</td>
    </tr>
  </tbody>
</table>
</div>



For `price`, Q1 and Q3 show the middle half of the market. This is useful because prices have many high outliers.

## 7. Price distribution

Price is the most important continuous variable for the Airbnb dataset. We start with the full distribution and then use a limited x-axis to see the main part of the data more clearly.


```python
plt.figure(figsize=(9, 5))
sns.histplot(df_clean['price'], bins=60, kde=True)
plt.title('Distribution of Airbnb prices')
plt.xlabel('Price')
plt.ylabel('Number of listings')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_19_0.png)
    



```python
plt.figure(figsize=(9, 5))
sns.histplot(df_clean[df_clean['price'] <= 500]['price'], bins=40, kde=True)
plt.title('Distribution of Airbnb prices up to 500')
plt.xlabel('Price')
plt.ylabel('Number of listings')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_20_0.png)
    



```python
plt.figure(figsize=(9, 2.5))
sns.boxplot(x=df_clean['price'])
plt.title('Boxplot of Airbnb prices')
plt.xlabel('Price')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_21_0.png)
    



```python
plt.figure(figsize=(9, 2.5))
sns.stripplot(x=df_clean['price'], size=2, alpha=0.25, jitter=0.25)
plt.xlim(0, 500)
plt.title('Stripplot of Airbnb prices up to 500')
plt.xlabel('Price')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_22_0.png)
    


The price distribution is strongly right-skewed. Most listings are in the lower price range, but a small number of very expensive listings create a long right tail.

## 8. Price outliers

We use the usual IQR rule: values below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR are treated as outliers.


```python
price_q1 = df_clean['price'].quantile(0.25)
price_q3 = df_clean['price'].quantile(0.75)
price_iqr = price_q3 - price_q1
price_lower = price_q1 - 1.5 * price_iqr
price_upper = price_q3 + 1.5 * price_iqr

price_outliers = df_clean[(df_clean['price'] < price_lower) | (df_clean['price'] > price_upper)]

print('Q1:', round(price_q1, 2))
print('Median:', round(df_clean['price'].median(), 2))
print('Q3:', round(price_q3, 2))
print('IQR:', round(price_iqr, 2))
print('Upper outlier boundary:', round(price_upper, 2))
print('Number of price outliers:', len(price_outliers))

price_outliers[['listing_id', 'name', 'room_type', 'borough', 'price']].sort_values('price', ascending=False).head(10)
```

    Q1: 70.0
    Median: 105.0
    Q3: 175.0
    IQR: 105.0
    Upper outlier boundary: 332.5
    Number of price outliers: 569





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
      <th>listing_id</th>
      <th>name</th>
      <th>room_type</th>
      <th>borough</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>852</th>
      <td>2953058</td>
      <td>Film Location</td>
      <td>Entire place</td>
      <td>Brooklyn</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2243699</td>
      <td>SuperBowl Penthouse Loft 3,000 sqft</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>5250.0</td>
    </tr>
    <tr>
      <th>5163</th>
      <td>20654227</td>
      <td>Fulton 2</td>
      <td>Entire place</td>
      <td>Brooklyn</td>
      <td>5000.0</td>
    </tr>
    <tr>
      <th>5845</th>
      <td>22296197</td>
      <td>Chelsea Gallery for events, exhibitions, fashion</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>4160.0</td>
    </tr>
    <tr>
      <th>8702</th>
      <td>33171891</td>
      <td>30 days minimum Time square West Midtown apart...</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>4100.0</td>
    </tr>
    <tr>
      <th>721</th>
      <td>2224896</td>
      <td>NYC SuperBowl Wk 5 Bdrs River View</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>4000.0</td>
    </tr>
    <tr>
      <th>7043</th>
      <td>27629043</td>
      <td>A Night at Anchor Aboard Yacht Ventura</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>3750.0</td>
    </tr>
    <tr>
      <th>5991</th>
      <td>22779746</td>
      <td>East 7th Street III by (Hidden by Airbnb)</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>3518.0</td>
    </tr>
    <tr>
      <th>8508</th>
      <td>32476606</td>
      <td>Recently Renovated &amp; Furnished Apt- Room avail...</td>
      <td>Private Room</td>
      <td>Manhattan</td>
      <td>2850.0</td>
    </tr>
    <tr>
      <th>740</th>
      <td>2274084</td>
      <td>3 Bedroom Apartment</td>
      <td>Entire place</td>
      <td>Manhattan</td>
      <td>2750.0</td>
    </tr>
  </tbody>
</table>
</div>



The boxplot confirms that there are many high-price outliers. This is why the median is a better typical value for price than the mean.

## 9. Rating distribution

Ratings are bounded between 0 and 5, so their distribution is easier to interpret than price.


```python
rated_data = df_clean[df_clean['is_rated'] == 1].copy()

plt.figure(figsize=(8, 5))
sns.histplot(rated_data['rating'], bins=25, kde=True)
plt.title('Distribution of ratings')
plt.xlabel('Rating')
plt.ylabel('Number of listings')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_28_0.png)
    



```python
plt.figure(figsize=(8, 2.5))
sns.boxplot(x=rated_data['rating'])
plt.title('Boxplot of ratings')
plt.xlabel('Rating')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_29_0.png)
    



```python
rated_data['rating'].describe().round(2)
```




    count    7922.00
    mean        4.01
    std         0.57
    min         3.00
    25%         3.52
    50%         4.03
    75%         4.51
    max         5.00
    Name: rating, dtype: float64



Most ratings are relatively high. The distribution is not as extreme as price, but it is still not perfectly symmetric.

## 10. Availability and reviews

Next we look at `availability_365` and `number_of_reviews`. These variables help describe how active listings are.


```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_clean['availability_365'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Availability in days')
axes[0].set_xlabel('availability_365')

sns.histplot(df_clean['number_of_reviews'], bins=40, kde=True, ax=axes[1])
axes[1].set_title('Number of reviews')
axes[1].set_xlabel('number_of_reviews')

plt.tight_layout()
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_33_0.png)
    



```python
df_clean[['availability_365', 'number_of_reviews', 'reviews_per_month']].describe().round(2)
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
      <th>availability_365</th>
      <th>number_of_reviews</th>
      <th>reviews_per_month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9993.00</td>
      <td>9993.00</td>
      <td>9993.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>112.30</td>
      <td>22.47</td>
      <td>1.07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>131.65</td>
      <td>43.20</td>
      <td>1.54</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>44.00</td>
      <td>5.00</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>226.00</td>
      <td>22.00</td>
      <td>1.55</td>
    </tr>
    <tr>
      <th>max</th>
      <td>365.00</td>
      <td>510.00</td>
      <td>16.22</td>
    </tr>
  </tbody>
</table>
</div>



`number_of_reviews` is right-skewed. Most listings have a small number of reviews, while a few listings have a lot of them.

## 11. Categorical variables

For categorical variables, we use frequency tables and count plots.


```python
df_clean['room_type'].value_counts()
```




    room_type
    Entire place    5172
    Private Room    4595
    Shared Room      226
    Name: count, dtype: int64




```python
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='room_type', order=df_clean['room_type'].value_counts().index)
plt.title('Number of listings by room type')
plt.xlabel('Room type')
plt.ylabel('Number of listings')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_38_0.png)
    



```python
df_clean['borough'].value_counts()
```




    borough
    Manhattan        4436
    Brooklyn         4075
    Queens           1180
    Bronx             229
    Staten Island      73
    Name: count, dtype: int64




```python
plt.figure(figsize=(9, 5))
sns.countplot(data=df_clean, y='borough', order=df_clean['borough'].value_counts().index)
plt.title('Number of listings by borough')
plt.xlabel('Number of listings')
plt.ylabel('Borough')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_40_0.png)
    


The two largest room type groups are `Entire place` and `Private Room`. The largest borough groups are Manhattan and Brooklyn.

## 12. Price grouped by room type and borough

Although this is still mostly a univariate report, grouped summaries help us describe the same variable (`price`) across simple categories.


```python
price_by_room_type = df_clean.groupby('room_type')['price'].describe().round(2)
price_by_room_type
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>room_type</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Entire place</th>
      <td>5172.0</td>
      <td>208.93</td>
      <td>249.21</td>
      <td>10.0</td>
      <td>120.0</td>
      <td>163.0</td>
      <td>225.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>Private Room</th>
      <td>4595.0</td>
      <td>87.02</td>
      <td>101.53</td>
      <td>10.0</td>
      <td>53.0</td>
      <td>70.0</td>
      <td>95.0</td>
      <td>2850.0</td>
    </tr>
    <tr>
      <th>Shared Room</th>
      <td>226.0</td>
      <td>69.78</td>
      <td>127.28</td>
      <td>10.0</td>
      <td>35.0</td>
      <td>50.0</td>
      <td>75.0</td>
      <td>1800.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(9, 5))
sns.boxplot(data=df_clean[df_clean['price'] <= 500], x='room_type', y='price')
plt.title('Price by room type, limited to 500')
plt.xlabel('Room type')
plt.ylabel('Price')
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_44_0.png)
    



```python
price_by_borough = df_clean.groupby('borough')['price'].describe().round(2)
price_by_borough
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
    <tr>
      <th>borough</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Bronx</th>
      <td>229.0</td>
      <td>88.63</td>
      <td>98.62</td>
      <td>20.0</td>
      <td>45.0</td>
      <td>65.0</td>
      <td>99.0</td>
      <td>1000.0</td>
    </tr>
    <tr>
      <th>Brooklyn</th>
      <td>4075.0</td>
      <td>123.06</td>
      <td>185.36</td>
      <td>10.0</td>
      <td>60.0</td>
      <td>90.0</td>
      <td>150.0</td>
      <td>8000.0</td>
    </tr>
    <tr>
      <th>Manhattan</th>
      <td>4436.0</td>
      <td>192.36</td>
      <td>232.41</td>
      <td>10.0</td>
      <td>93.0</td>
      <td>150.0</td>
      <td>220.0</td>
      <td>5250.0</td>
    </tr>
    <tr>
      <th>Queens</th>
      <td>1180.0</td>
      <td>96.95</td>
      <td>102.80</td>
      <td>10.0</td>
      <td>50.0</td>
      <td>72.0</td>
      <td>110.0</td>
      <td>2000.0</td>
    </tr>
    <tr>
      <th>Staten Island</th>
      <td>73.0</td>
      <td>91.89</td>
      <td>58.60</td>
      <td>29.0</td>
      <td>50.0</td>
      <td>75.0</td>
      <td>110.0</td>
      <td>300.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_clean[df_clean['price'] <= 500], x='borough', y='price')
plt.title('Price by borough, limited to 500')
plt.xlabel('Borough')
plt.ylabel('Price')
plt.xticks(rotation=20)
plt.show()
```


    
![png](Univariate_Analysis_Report_files/Univariate_Analysis_Report_46_0.png)
    


Entire places are usually more expensive than private or shared rooms. Manhattan also has higher typical prices than most other boroughs.

## 13. Final findings

From the univariate analysis we can summarize the Airbnb data as follows:

- Price is strongly right-skewed.
- The median price is more useful than the mean because of high outliers.
- The boxplot shows many price outliers.
- Ratings are usually high and stay within the 0-5 range after cleaning.
- Most listings are either entire places or private rooms.
- Manhattan and Brooklyn have the largest number of listings.
- Entire places and Manhattan listings tend to have higher prices.

This report gives a first one-variable-at-a-time view of the Airbnb dataset before moving to bivariate or multivariate analysis.
