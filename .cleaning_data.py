"""Author: Pascale C. Fischbach """

""" EDA Project - King County Housing
King County Housing (KCH) data. Client is Erin Robinson.
She is looking to invest in poor neighborhood, buy & sell.
Costs back plus a little profit. Socially responsible decisions.
"""
### IMPORT NEEDED LIBRARIES
import pandas as pd
import numpy as np
from geopy.distance import geodesic    # calculate distance between two points with given geodata
from numpy import setdiff1d

### IMPORT DATA AND SAVE AS DATAFRAME
kch_df = pd.read_csv('data/King_County_House_prices_dataset.csv')


### FOR OVERVIEW ON DATA REFER TO EDA NOTEBOOK

### START CLEANING PROCESS
# transform date type to_datetime
kch_df['date'] = pd.to_datetime(kch_df['date'], format="%m/%d/%Y")
kch_df['date'] = pd.to_datetime(kch_df['date'].dt.date)
print('Type of column "date" now: ', type(kch_df['date'][0]))

# make month and year columns for later best selling time analysis
kch_df['month'] = kch_df['date'].dt.month
kch_df['year'] = kch_df['date'].dt.year

# Transform dtype of condition to int
kch_df['condition'] = kch_df['condition'].astype('int32')
print('Type of column "condition" now: ', type(kch_df['condition'][0]))

# transform data type of yr_renovated from float to int
# first check, why dtype was made float
print('Unique yr_renovated: ', kch_df.yr_renovated.unique())    # have to handle missing values first

kch_df['yr_renovated'] = np.nan_to_num(kch_df['yr_renovated'])
print('Unique yr_renovated after nan-handling: ', kch_df.yr_renovated.unique())

# change data type now
kch_df['yr_renovated'] = kch_df['yr_renovated'].astype('int')

# Add variable for distance of each property to Seattle Center with coordinates
# -122.3320700 (lon), 47.6062100 (lat) which I took from
# https://dateandtime.info/citycoordinates.php?id=5809844 on 27th Nov, 2023 at 10:27 am.
seattle = (47.60621, -122.33207)
dist_to_seattle = pd.Series([geodesic(seattle, (kch_df.lat[i], kch_df.long[i]))
                                     for i in np.arange(len(kch_df.lat))])
kch_df['dist_to_seattle'] = dist_to_seattle # in km

# check distance
print(f'Minimum distance to Seattle: {kch_df.dist_to_seattle.min()}\n Maximum distance to Seattle: {kch_df.dist_to_seattle.max()}')
print(f'Data type of dist_to_seattle: {type(kch_df.dist_to_seattle[0])}')

#Need to change dtype in order to use distance as measure for graphs:
kch_df['dist_to_seattle'] = kch_df['dist_to_seattle'].astype('str')
kch_df['dist_to_seattle'] = kch_df['dist_to_seattle'].str.strip('km').astype('float')
print(f'Data type distance to Seattle: {type(kch_df.dist_to_seattle[0])}')

# Calculate price per sqft of living space
kch_df['price_sqft_living'] = kch_df.price / kch_df.sqft_living

# Deal with duplicates
print(f'Duplicates? {kch_df.duplicated().value_counts()}') # no duplicates

# Are there any properties more than once?
# Check for duplicates for individual properties
print(f"Check for duplicates for individual properties: {kch_df['id'].duplicated(keep='last').value_counts()}")

# yes, there are.
# check duplicates - difference in price and date?
duplicate_properties = kch_df.query('id.duplicated(keep="last")==True') # keep the first entry of duplicate, i.e. "old" price etc.

# make new variable stating whether property was sold more than once (True/False)
kch_df['multi_sold'] = pd.Series(np.zeros(len(kch_df)).astype('bool'))
kch_df.loc[duplicate_properties.index+1,['multi_sold']] = True
print('Nunique multiple times sold: ', kch_df.query('multi_sold==True').id.nunique())

# drop duplicates, i.e. old price
kch_df.drop(index=duplicate_properties.index, axis=0, inplace=True)

# Append additional columns with information on date sold and price sold for the duplicate properties.
# For those properties not having been sold more than once fill value with 0. Take older selling date and
# price into new variable.
print(f"Indices of multiple times sold properties:\n{kch_df.query('multi_sold==True').index}")

# one property was sold 3 times (id 795000620). Just take the latest date and price for most updated and the middle
# date/price for old price (more recent price evolution)

# Drop index 17588 which has id 795000620
duplicate_properties.drop(axis=0, index=17588, inplace=True)

# fill new columns with old selling date and old price
kch_df.loc[kch_df.query('multi_sold==True').index,'date_sold_old'] = duplicate_properties['date'].values
kch_df.loc[kch_df.query('multi_sold==True').index,'price_sold_old'] = duplicate_properties['price'].values

# copy df and get rid of not used data
kch_robinson = kch_df.copy()
kch_robinson.drop(columns=['bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'grade',
       'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15', 'month', 'year'],
       inplace=True)


### INTERIM DATA EXPLORATION
print(f'Columns: {kch_robinson.columns}')

import matplotlib.pyplot as plt
# Distributions
fig, ax = plt.subplots(1,3, figsize=(20,5)) # create subplots on 1 row and 3 columns
plt.suptitle('Distribution of condition, sqft_living15, yr_renovated', fontsize=20)
fig.tight_layout()

ax[0].hist(kch_robinson['condition']) # boxplot for condition
ax[0].set_title("Distribution of condition count", fontsize = 15); # sets title for subplot

ax[1].boxplot(x = kch_robinson['price_sqft_living'])
ax[1].set_title("Distribution of price per living sqft", fontsize = 15);

ax[2].hist(x = kch_robinson.query('yr_renovated!=0').yr_renovated)
ax[2].set_title("Distribution of year last renovated count (if ever renovated)", fontsize = 15);

fig.show()

###TAKE AWAYS SO FAR
# Mostly medium condition, so not too shabby. Socially responsible renovation for properties
    #of conditions 2 or 3. everything above unnecessary posh
# costs per sqft has a lot of outliers to the top. maybe too expensive - drop those? look closer
    #(total price, need of renovation?)
# most properties never renovated - maybe not that old - or in high need of renovation.
    #most renovations after 1980

### DATA CLEANING 2. ITERATION

# make df with only price per sqft of living area and the according zipcode
price_in_Q2 = kch_robinson.loc[:,['price_sqft_living', 'zipcode']]

# add column with frequency of each zipcode
price_in_Q2['freq_zipcode'] = price_in_Q2.groupby('zipcode')['zipcode'].transform('count')

# make data frame with only zipcodes where price per sqft of living area fall in second quartile
freq_Q2_zip = pd.DataFrame(price_in_Q2.query('price_sqft_living<=price_sqft_living.median()')['zipcode'])

# count frequency of each zipcode and add that information as a column
freq_Q2_zip['freq_price_q2'] = price_in_Q2.query('price_sqft_living<=price_sqft_living.median()').groupby('zipcode')['zipcode'].transform('count')

# only keep unique zips with respective frequency
price_in_Q2 = price_in_Q2.sort_values(by='zipcode')
price_in_Q2.drop_duplicates(subset='zipcode', inplace=True)
freq_Q2_zip = freq_Q2_zip.sort_values(by='zipcode')
freq_Q2_zip.drop_duplicates(subset='zipcode', inplace=True)


freq_Q2_zip.zipcode.sort_values().unique(), price_in_Q2.zipcode.sort_values().unique()
set1 = price_in_Q2.zipcode.sort_values().unique()
set2 = freq_Q2_zip.zipcode.sort_values().unique()
zip_diff = setdiff1d(set1,set2)[0]

# which row is zipcode 98039 in in price_in_Q2? Drop that
price_in_Q2.drop(axis=0, index=price_in_Q2.query('zipcode==@zip_diff').index, inplace=True)

# check
price_in_Q2.zipcode.unique() in freq_Q2_zip.zipcode.unique()

price_in_Q2.reset_index(inplace=True)
price_in_Q2.drop(axis=1, labels='index', inplace=True)
freq_Q2_zip.reset_index(inplace=True)
freq_Q2_zip.drop(axis=1, labels='index', inplace=True)

# add column with percentage of properties in zipcode that have a price per sqft of living up to median price
freq_Q2_zip['share_q2'] = (freq_Q2_zip['freq_price_q2'] / price_in_Q2['freq_zipcode']) * 100

# Make pd.Series with only zipcodes that meet share >= 80 %
share_80 = pd.Series(freq_Q2_zip.query('share_q2 >= 80')['zipcode'])

# only keep those properties that are in zipcode areas where at least 80% have a price per sqft living up to median price
kch_robinson = kch_robinson.query('zipcode in @share_80')
kch_robinson.reset_index(inplace=True)
kch_robinson.drop(labels='index', axis=1, inplace=True)

# add column with years between building and renovating. if never renovated calculate age of house (i.e. (year 2015 - year property was built))
yrs_since_renovation = pd.Series(abs(2023 - kch_robinson.yr_renovated))
yrs_since_renovation.name = 'yrs_since_renovation'
yrs_since_renovation = pd.DataFrame(yrs_since_renovation)
yrs_since_renovation['yr_renovated'] = kch_robinson.yr_renovated.values

yrs_since_renovation.loc[yrs_since_renovation.query('yr_renovated==0').index,['yrs_since_renovation']] = 0

kch_robinson['yrs_since_renovation'] = yrs_since_renovation['yrs_since_renovation'].values


# DISTRIBUTIONS AGAIN
# Check for distributions again with cleaned data set, only poor neighborhoods

fig, ax = plt.subplots(2,2, figsize=(20,10)) # create subplots on 1 row and 3 columns
plt.suptitle('Distribution of condition, sqft_living15, yr_renovated', fontsize=20)
fig.tight_layout()

ax[0][0].hist(kch_robinson['condition']) # boxplot for condition
ax[0][0].set_title("Distribution of condition count", fontsize = 15); # sets title for subplot

ax[0][1].hist(x = kch_robinson.query('yr_renovated!=0').yr_renovated)
ax[0][1].set_title("Distribution of year last renovated count (if ever renovated)", fontsize = 15);

ax[1][0].boxplot(x = kch_robinson['price_sqft_living'])
ax[1][0].set_title("Distribution of price per living sqft", fontsize = 15);

ax[1][1].boxplot(x = kch_robinson['price'])
ax[1][1].set_title("Distribution of total price", fontsize = 15);


### SAVE CLEANED DATA SET FOR EDA
kch_robinson.to_csv('data/kch_poor_neighborhood_clean_data.csv')
kch_df.to_csv('data/kch_clean_data.csv')