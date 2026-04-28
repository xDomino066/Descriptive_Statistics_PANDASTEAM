# Airbnb Data Cleaning

## Task: Clean the Airbnb dataset

### Step 1. Import the necessary libraries
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import missingno as msno
import datetime as dt

from IPython.display import display
```

### Step 2. Import the dataset and create a cleaning copy
```python
airbnb = pd.read_csv('../data/airbnb.csv', index_col='Unnamed: 0')
df_clean = airbnb.copy()

airbnb.head()
```

### Step 3. Check data types and DataFrame information
```python
airbnb.dtypes
airbnb.info()
```

### Step 4. Check missing values and descriptive statistics
```python
airbnb.isna().sum()
airbnb.describe()
```

### Step 5. Check `room_type` categories
```python
airbnb['room_type'].unique()
airbnb['room_type'].value_counts()
```

### Step 6. Check example prices
```python
airbnb['price'].head(10)
```

### Step 7. Visualize missing values
```python
msno.matrix(airbnb)
plt.show()

msno.bar(airbnb)
plt.show()
```

### Step 8. Visualize rating distribution
```python
sns.histplot(airbnb['rating'], kde=True, bins=20)
plt.title('Distribution of listing ratings')
plt.show()
```

### Step 9. Start from a fresh copy of the original DataFrame
```python
df_clean = airbnb.copy()
```

### Step 10. Split coordinates into latitude and longitude
```python
coordinates_split = (
    df_clean['coordinates']
    .str.replace('(', '', regex=False)
    .str.replace(')', '', regex=False)
    .str.split(',', expand=True)
)

df_clean['latitude'] = coordinates_split[0].str.strip().astype(float)
df_clean['longitude'] = coordinates_split[1].str.strip().astype(float)

df_clean = df_clean.drop(columns='coordinates')

df_clean[['latitude', 'longitude']].head()
```

### Step 11. Remove `$` from price and convert it to float
```python
missing_price_before = df_clean['price'].isna().sum()
zero_price_before = (df_clean['price'].str.strip('$').astype(float) == 0).sum()

df_clean['price'] = df_clean['price'].str.strip('$')
df_clean['price'] = df_clean['price'].astype(float)

df_clean.loc[df_clean['price'] <= 0, 'price'] = np.nan

df_clean['price'].head()
```

### Step 12. Convert date columns
```python
df_clean['listing_added'] = pd.to_datetime(df_clean['listing_added'], format='%Y-%m-%d')
df_clean['last_review'] = pd.to_datetime(df_clean['last_review'], format='%Y-%m-%d')

df_clean[['listing_added', 'last_review']].head()
```

### Step 13. Check selected column types after conversion
```python
df_clean[['price', 'listing_added', 'last_review', 'latitude', 'longitude']].dtypes
```

### Step 14. Check original `room_type` values
```python
df_clean['room_type'].unique()
```

### Step 15. Standardize spelling, spacing and capitalization
```python
df_clean['room_type'] = df_clean['room_type'].str.lower()
df_clean['room_type'] = df_clean['room_type'].str.strip()

df_clean['room_type'].unique()
```

### Step 16. Replace inconsistent `room_type` values with clean categories
```python
room_type_mapping = {
    'private room': 'Private Room',
    'private': 'Private Room',
    'entire home/apt': 'Entire place',
    'home': 'Entire place',
    'shared room': 'Shared Room'
}

df_clean['room_type'] = df_clean['room_type'].replace(room_type_mapping)

df_clean['room_type'].value_counts()
```

### Step 17. Split borough and neighbourhood
```python
neighbourhood_split = df_clean['neighbourhood_full'].str.split(',', expand=True)

df_clean['borough'] = neighbourhood_split[0].str.strip()
df_clean['neighbourhood'] = neighbourhood_split[1].str.strip()

df_clean = df_clean.drop(columns='neighbourhood_full')

df_clean[['borough', 'neighbourhood']].head()
```

### Step 18. Cap ratings above 5
```python
rating_above_5 = (df_clean['rating'] > 5).sum()

print('Ratings above 5 before cleaning:', rating_above_5)

df_clean.loc[df_clean['rating'] > 5, 'rating'] = 5

print('Maximum rating after cleaning:', df_clean['rating'].max())
```

### Step 19. Check missing ratings and number of reviews
```python
df_clean[df_clean['rating'].isna()]['number_of_reviews'].value_counts()
```

### Step 20. Fill review-related missing values and create `is_rated`
```python
df_clean = df_clean.fillna({
    'reviews_per_month': 0,
    'number_of_stays': 0,
    '5_stars': 0
})

df_clean['is_rated'] = np.where(df_clean['rating'].isna(), 0, 1)

df_clean[['number_of_reviews', 'reviews_per_month', 'number_of_stays', '5_stars', 'rating', 'is_rated']].head()
```

### Step 21. Check price distribution by room type
```python
sns.boxplot(x='room_type', y='price', data=df_clean)
plt.ylim(0, 400)
plt.title('Price by room type')
plt.show()
```

### Step 22. Calculate median price by room type
```python
median_price_by_room_type = df_clean.groupby('room_type')['price'].median()
median_price_by_room_type
```

### Step 23. Fill missing prices using median price by room type
```python
df_clean['price'] = df_clean['price'].fillna(df_clean['room_type'].map(median_price_by_room_type))

df_clean['price'].isna().sum()
```

### Step 24. Fill small text missing values
```python
df_clean['name'] = df_clean['name'].fillna('Unknown listing')
df_clean['host_name'] = df_clean['host_name'].fillna('Unknown host')

df_clean[['name', 'host_name']].isna().sum()
```

### Step 25. Check future listings and future reviews
```python
today = dt.date.today()

future_listings = df_clean[df_clean['listing_added'].dt.date > today]
future_reviews = df_clean[df_clean['last_review'].dt.date > today]

print('Listings added in the future:', len(future_listings))
print('Reviews in the future:', len(future_reviews))
```

### Step 26. Find records where listing date is after last review
```python
inconsistent_dates = df_clean[df_clean['listing_added'].dt.date > df_clean['last_review'].dt.date]

inconsistent_dates[['listing_id', 'name', 'listing_added', 'last_review']]
```

### Step 27. Remove rows with inconsistent dates
```python
df_clean = df_clean.drop(inconsistent_dates.index)

print('Rows after removing inconsistent dates:', len(df_clean))
```

### Step 28. Check and remove fully duplicated rows
```python
exact_duplicates = df_clean.duplicated().sum()
print('Fully duplicated rows:', exact_duplicates)

df_clean = df_clean.drop_duplicates()
```

### Step 29. Check duplicated `listing_id` values
```python
duplicated_listing_ids = df_clean[df_clean.duplicated(subset='listing_id', keep=False)].sort_values('listing_id')

duplicated_listing_ids.head(10)
```

### Step 30. Merge repeated listings with the same `listing_id`
```python
rows_before_listing_id_cleaning = len(df_clean)

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

print('Rows merged by listing_id:', rows_before_listing_id_cleaning - len(df_clean))
print('Is listing_id unique now?', df_clean['listing_id'].is_unique)
```

### Step 31. Check cleaned DataFrame information
```python
df_clean.info()
```

### Step 32. Check missing values after cleaning
```python
df_clean.isna().sum()
```

### Step 33. Compare original and cleaned data shape
```python
print('Original shape:', airbnb.shape)
print('Cleaned shape:', df_clean.shape)
```

### Step 34. Create cleaning summary table
```python
summary = pd.DataFrame({
    'step': [
        'missing prices before cleaning',
        'zero prices treated as missing',
        'ratings above 5 corrected',
        'rows with inconsistent dates removed',
        'fully duplicated rows removed',
        'rows merged by listing_id'
    ],
    'count': [
        missing_price_before,
        zero_price_before,
        rating_above_5,
        len(inconsistent_dates),
        exact_duplicates,
        rows_before_listing_id_cleaning - len(df_clean)
    ]
})

summary
```

### Step 35. Preview the cleaned DataFrame
```python
df_clean.head()
```

### Step 36. Main cleaning operations

In this notebook, the Airbnb dataset was cleaned by:

- splitting `coordinates` into `latitude` and `longitude`,
- converting `price` from text to numeric values,
- converting `listing_added` and `last_review` to date columns,
- cleaning the `room_type` categories,
- splitting `neighbourhood_full` into `borough` and `neighbourhood`,
- capping ratings above 5,
- filling review-related missing values with 0 when listings had no reviews,
- filling missing prices using median price by room type,
- filling missing values in `name` and `host_name`,
- removing rows with inconsistent dates,
- removing fully duplicated rows,
- merging repeated listings with the same `listing_id`.

The original CSV file was not changed. All cleaning was done in the `df_clean` DataFrame.
