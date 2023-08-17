#import necessary libararies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from scipy import stats
from scipy.stats import norm, skew
from sklearn.model_selection import train_test_split, KFold, GroupKFold,GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler, RobustScaler
from sklearn.metrics import *
import sys, os
import random
if not sys.warnoptions:
import warnings
warnings.simplefilter("ignore")

#Read CSV file
df=pd.read_csv("C:/Users/DELL/Downloads/solar_wind (1).csv")
df.head()

#print shape and info 
df.shape
df.info()

#to describe the dataset use
df.describe()

#calculate null values
pd.isnull(df).sum()

#to print the column names and know the datatypes use the following commmands
df.columns
df.dtypes

#handle the null values
numeric_columns = ['bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm','phi_gse' , 'phi_gsm', 'bt']
for column in numeric_columns:
mean_value = df[column].mean()
df[column].fillna(mean_value, inplace=True)
df['source'].fillna(method='ffill', inplace=True)

#print number of null values after mean imputation
print("Number of null values in each column after imputation:")
print(df.isnull().sum())

#now read remaining two datasets
dst = pd.read_csv("C:/Users/DELL/Downloads/labels.csv")
dst.head()
sunspots = pd.read_csv("C:/Users/DELL/Downloads/sunspots.csv")
sunspots.head()

#lets convert the timedelta format to proper date and time format
df.timedelta = pd.to_timedelta(df.timedelta)
dst.timedelta = pd.to_timedelta(dst.timedelta)
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta

# Set the index for each DataFrame using "period" and "timedelta"
df.set_index(["period", "timedelta"], inplace=True)
dst.set_index(["period", "timedelta"], inplace=True)
sunspots.set_index(["period", "timedelta"], inplace=True)

# Print basic information about the data and show the first few rows
print("Solar wind shape: ", df.shape)
print("Sunspot shape: ", sunspots.shape)
print("Label: ", dst.shape)

print("Solar wind data:")
print(df.head())

print("Sunspot data:")
print(sunspots.head())

print("label data:")
print(dst.head())

# Visualize the raw solar wind data for selected features
plt.style.use('fivethirtyeight')
def show_raw_visualization(data):
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15), dpi=80)
for i, key in enumerate(data.columns):
t_data = data[key]
ax = t_data.plot(
ax=axes[i // 2, i % 2],
title=f"{key.capitalize()}",
rot=25, color='teal', lw=1.2
)
fig.subplots_adjust(hspace=0.8)
plt.tight_layout()
cols_to_plot = ["bx_gse", "bx_gsm", "bt", "density", "speed", "temperature"]
show_raw_visualization(df[cols_to_plot].iloc[:1000])

# Check for missing values in solar wind data
print(df.isnull().sum())

# Join the solar wind, sunspot, and DST labels data and fill missing values
joined = df.join(sunspots).join(dst).fillna(method="ffill")

# Visualize the correlation between features using a clustermap
plt.figure(figsize=(20, 15))
sns.clustermap(joined.corr(), annot=True)

# Set random seeds for reproducibility -By setting both NumPy and TensorFlow␣
from numpy.random import seed
from tensorflow.random import set_seed
seed(2020)
set_seed(2021)

# Import StandardScaler for feature scaling
from sklearn.preprocessing import StandardScaler
# Define a subset of solar wind features to use for modeling
SOLAR_WIND_FEATURES = [
"bt",
"temperature",
"bx_gse",
"by_gse",
"bz_gse",
"speed",
"density",
]

# Define all features to use, including sunspot numbers
XCOLS = (
[col + "_mean" for col in SOLAR_WIND_FEATURES]
+ [col + "_std" for col in SOLAR_WIND_FEATURES]
+ ["smoothed_ssn"]
)

# Function to impute missing values using forward fill for sunspot data and␣
def impute_features(feature_df):
feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
feature_df = feature_df.interpolate()
return feature_df

# Function to aggregate features to the floor of each hour using mean and␣
def aggregate_hourly(feature_df, aggs=["mean", "std"]):
agged = feature_df.groupby(
["period", feature_df.index.get_level_values(1).floor("H")]
).agg(aggs)
agged.columns = ["_".join(x) for x in agged.columns]
return agged

 # Function to preprocess features-Overall, this function allows you to␣
def preprocess_features(df, sunspots, scaler=None, subset=None):
if subset:
df = df[subset]
hourly_features = aggregate_hourly(df).join(sunspots)
if scaler is None:
scaler = StandardScaler()
scaler.fit(hourly_features)
normalized = pd.DataFrame(
scaler.transform(hourly_features),
index=hourly_features.index,
columns=hourly_features.columns,
)
imputed = impute_features(normalized)
return imputed, scaler

# Preprocess features using the defined functions
features, scaler = preprocess_features(df, sunspots, subset=SOLAR_WIND_FEATURES)

# Check the shape and ensure there are no missing values in the features␣
↪DataFrame
print(features.shape)
assert (features.isna().sum() == 0).all()

# Define target columns for labels-In summary, this function takes a DataFrame␣
YCOLS = ["t0", "t1"]
# Function to process labels for forecasting
def process_labels(dst):
y = dst.copy()
y["t1"] = y.groupby("period").dst.shift(-1)
y.columns = YCOLS
return y

# Process labels using the defined function
labels = process_labels(dst)

# Combine features and labels into a single DataFrame
data = labels.join(features)

# Function to split data across periods into train, test, and validation␣
def get_train_test_val(data, test_per_period, val_per_period):
test = data.groupby("period").tail(test_per_period)
interim = data[~data.index.isin(test.index)]
val = data.groupby("period").tail(val_per_period)
train = interim[~interim.index.isin(val.index)]
return train, test, val

# Split data into train, test, and validation sets using the defined function
train, test, val = get_train_test_val(data, test_per_period=6_000,val_per_period=3_000)


import tensorflow as tf
from keras import preprocessing
# Configuration for time series data
data_config = {
"timesteps": 32,
"batch_size": 32,
}

 # Function to create time series dataset from DataFrame
def timeseries_dataset_from_df(df, batch_size):
dataset = None
timesteps = data_config["timesteps"]
for _, period_df in df.groupby("period"):
inputs = period_df[XCOLS][:-timesteps]
outputs = period_df[YCOLS][timesteps:]
period_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
inputs,
outputs,
timesteps,
batch_size=batch_size,
)
if dataset is None:
dataset = period_ds
else:
dataset = dataset.concatenate(period_ds)
return dataset

# Create time series datasets for training and validation
train_ds = timeseries_dataset_from_df(train, data_config["batch_size"])
val_ds = timeseries_dataset_from_df(val, data_config["batch_size"])
print(f"Number of train batches: {len(train_ds)}")
print(f"Number of val batches: {len(val_ds)}")

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, LSTM
# Define the LSTM model-This code defines an LSTM (Long Short-Term Memory)␣
model_config = {"n_epochs": 20, "n_neurons": 512, "dropout": 0.4, "stateful":False}
model = Sequential()
model.add(
LSTM(
model_config["n_neurons"],
batch_input_shape=(None, data_config["timesteps"], len(XCOLS)),
stateful=model_config["stateful"],
dropout=model_config["dropout"],
)
)
model.add(Dense(len(YCOLS)))
model.compile(
loss="mean_squared_error",
optimizer="adam",
)
model.summary()

# Train the model and save training history
history = model.fit(
15
train_ds,
batch_size=data_config["batch_size"],
epochs=model_config["n_epochs"],
verbose=1,
shuffle=False,
validation_data=val_ds,
)

# Plot the training and validation loss
for name, values in history.history.items():
plt.plot(values)

# Create the time series dataset for testing
test_ds = timeseries_dataset_from_df(test, data_config["batch_size"])

# Evaluate the model on the test dataset and calculate the RMSE-trained a␣
mse = model.evaluate(test_ds)
print(f"Test RMSE: {mse**.5:.2f}")

# Save the trained model, scaler, and data configuration for future use-In this␣
import pickle
import json
model.save("model")
with open("scaler.pck", "wb") as f:
pickle.dump(scaler, f)
data_config["solar_wind_subset"] = SOLAR_WIND_FEATURES
print(data_config)
with open("config.json", "w") as f:
json.dump(data_config, f)






