import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('assets/DATA/SuperMarketSales.csv',
                      parse_dates=['Date'], dayfirst=True)

# Examine the Dataset (prints first few rows).
dataset.head()

# dictionary To save Feature and its MSE
my_dict = {}

# Extract New Features From Date Column using pandas (creating new features from one column).

# Convert date string to datetime format for manipulation.
dataset['Date'] = pd.to_datetime(dataset['Date'])
# Extract year from the date-time formatted column.
dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
# Extract month from the date-time formatted column.
dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
# Extract day from the date-time formatted column.
dataset['Day'] = pd.DatetimeIndex(dataset['Date']).day
# day_of_year:  (1-365 or 1-366 for leap years).
dataset['day_of_year'] = pd.DatetimeIndex(dataset['Date']).dayofyear


# Delete The Date Column after we create the new feature since it is no longer needed.
dataset.drop(['Date'], axis=1, inplace=True)

# Simple Linear Regression
def plot_and_get_mse(x, y):
    feature = np.array(dataset[x]).reshape(dataset[x].shape[0], 1)
    label = np.array(dataset[y]).reshape(dataset[y].shape[0], 1)
    model = LinearRegression()
    model.fit(feature, label)
    predictions = model.predict(feature)
    MSE = mean_squared_error(label, predictions)
    plt.scatter(dataset[x], dataset[y])
    plt.xlabel(x)
    plt.ylabel('Weekly_Sales')
    plt.plot(dataset[x], predictions, color='red', linewidth=4)
    plt.show()
    return MSE

def plot_and_get_mse(x, y):
    feature = np.array(dataset[x]).reshape(dataset[x].shape[0], 1)
    label = np.array(dataset[y]).reshape(dataset[y].shape[0], 1)
    model = LinearRegression()
    model.fit(feature, label)
    predictions = model.predict(feature)
    MSE = mean_squared_error(label, predictions)
    plt.scatter(dataset[x], dataset[y])
    plt.xlabel(x)
    plt.ylabel('Weekly_Sales')
    plt.plot(dataset[x], predictions, color='red', linewidth=4)
    plt.show()
Features = ['Store', 'Temperature', 'Fuel_Price',
            'CPI', 'Year', 'Month', 'Day', 'day_of_year']

# For loop to calculate MSE to all features
for Feature in Features:
    MSE = plot_and_get_mse(Feature, "Weekly_Sales")
    my_dict[Feature] = MSE
    print(f'{Feature} Mean squared error: {MSE}')


print("____________________________________________________________")
# Print the biggest MSE, least MSE, and the best feature
max_key = max(my_dict, key=my_dict.get)
max_value = max(my_dict.values())
min_key = min(my_dict, key=my_dict.get)
min_value = min(my_dict.values())

print(f'The biggest MSE is "{max_key}" = {max_value}')
print(f'The least MSE is "{min_key}" = {min_value}')
print(f'The best feature is "{min_key}"')
