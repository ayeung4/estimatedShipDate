# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:01:22 2021
@company: Gooten
@author: allen
"""
# Import Libraries
import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from scipy import stats


# Load in data file
plt.close("all")
file = "Data/Production Days Data 2020.csv"
df = pd.read_csv(file, header=0, low_memory=False)

#  DataFrame pre-processing
# 1. Reduce dataframe
# 2. Convert date columns to datetime
# 3. Remove nan values 
# 4. Sort by In Production Date
df.iloc[:,7:10] = df.iloc[:,7:10].apply(pd.to_datetime, errors='coerce', utc=True)
df = df[['VENDOR_ID', 'CATEGORY_ID','PRODUCT_ID','SKU_ID','IN_PRODUCTION_DATE','SHIP_DATE']]
df.dropna(inplace=True)
df = df.sort_values(by='IN_PRODUCTION_DATE')
df = df.loc[(df['IN_PRODUCTION_DATE']>="2021-01-04") & (df['IN_PRODUCTION_DATE']<="2021-01-20")]

# Calculate Production Days: Ship Date - In Production Date
df['PRODUCTION_DAYS'] = (df['SHIP_DATE'] - df['IN_PRODUCTION_DATE']).astype('timedelta64[D]')

# Group data by Vendor and Product
df['VenProd'] = df.iloc[:,[0,2]].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

# Calculate SMA, EMA and CMA
df['SMA_10'] = df.groupby('VenProd')['PRODUCTION_DAYS'].transform(lambda x: x.rolling(10,1).mean())
df['SMA_20'] = df.groupby('VenProd')['PRODUCTION_DAYS'].transform(lambda x: x.rolling(20,1).mean())
df['CMA_10'] = df.groupby('VenProd')['PRODUCTION_DAYS'].transform(lambda x: x.expanding(min_periods=10).mean())
df['EMA'] = df.groupby('VenProd')['PRODUCTION_DAYS'].transform(lambda x: x.ewm(span=10,adjust=False).mean())

# Filter by particular VenProd
filter_param = "3_43"
vp = df.query('VenProd == "' + filter_param +'"')

# General Analysis of Filter VenProd
print("############ VenProd = " + filter_param + " ################")
print(vp['PRODUCTION_DAYS'].describe())
plt.hist(vp['PRODUCTION_DAYS'])


# Plot Analysis
vp.plot(kind='line',x='IN_PRODUCTION_DATE',y=['PRODUCTION_DAYS','SMA_10','SMA_20','CMA_10','EMA'],figsize=(15,10))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Algo Analysis
algo = "EMA"
print("########## " + algo + " ##############")
RMSE = math.sqrt(np.square(np.subtract(vp['PRODUCTION_DAYS'],vp[algo])).mean())
print("RMSE:",RMSE)
Bias = np.subtract(vp[algo],vp['PRODUCTION_DAYS']).sum()
print("Bias:", Bias)
print(vp[algo].describe())
UpperLim = vp[algo].mean()+2*vp[algo].std()
vp['TEST'] = vp['PRODUCTION_DAYS'].between(0,UpperLim)
print("Orders with Production Days less than ", UpperLim)
print(vp['TEST'].value_counts(normalize=True)*100)
plt.hist(vp[algo])
plt.show()


'''
# Export Dataframe to a CSV file
compression_opts = dict(method='zip',archive_name='out.csv')
df.to_csv("output.zip", index=False, compression=compression_opts)
'''
