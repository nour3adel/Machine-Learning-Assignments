import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
dataset = pd.read_csv('assets/DATA/SuperMarketSales.csv',
                      parse_dates=['Date'], dayfirst=True)

# Examine the dataset
dataset.head()

# dictionary To save Feature and its MSE
my_dict = {}

# Extract New Featues From Date Column 
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
dataset['Day'] = pd.DatetimeIndex(dataset['Date']).day
dataset['day_of_year'] = pd.DatetimeIndex(dataset['Date']).dayofyear 

#  day_of_year: The day of the year as an integer (1-365 or 1-366 for leap years).


# Delete The Date Column after we create the new feature.
dataset.drop(['Date'], axis=1, inplace=True)

# Simple Linear Regression
def hypothesis(x, y):
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

# ------------------------------------------------------------------------------------------------
# Store Feature
# MSE_Store = hypothesis("Store", "Weekly_Sales")
# my_dict["Store"] = MSE_Store
# print("Store", "Mean squared error:", MSE_Store)

# ------------------------------------------------------------------------------------------------
# Temperature Feature
MSE_Temperature = hypothesis("Temperature","Weekly_Sales")
my_dict["Temperature"] = MSE_Temperature
print("Temperature", "Mean squared error:", MSE_Temperature)


# ------------------------------------------------------------------------------------------------
# Fuel_Price Feature

MSE_Fuel_Price = hypothesis("Fuel_Price","Weekly_Sales")
my_dict["Fuel_Price"] = MSE_Fuel_Price
print("Fuel_Price", "Mean squared error:", MSE_Fuel_Price)


# ------------------------------------------------------------------------------------------------
# CPI Feature

MSE_CPI = hypothesis("CPI","Weekly_Sales")
my_dict["CPI"] = MSE_CPI
print("CPI", "Mean squared error:", MSE_CPI)


# ------------------------------------------------------------------------------------------------
# Day Feature

MSE_Day= hypothesis("Day","Weekly_Sales")
my_dict["Day"] = MSE_Day
print("Day", "Mean squared error:", MSE_Day)


# ------------------------------------------------------------------------------------------------
# day_of_year Feature

MSE_yearday= hypothesis("day_of_year","Weekly_Sales")
my_dict["day_of_year"] = MSE_yearday
print("day_of_year", "Mean squared error:", MSE_yearday)


# ------------------------------------------------------------------------------------------------


print("____________________________________________________________")
# print the biggest MSE
max_key = max(my_dict, key=my_dict.get)
max_value = max(my_dict.values())
print("The Biggest MSE is ", "\"", max_key, "\"", " = ", max_value)
print("____________________________________________________________")
# print the Least MSE
min_key = min(my_dict, key=my_dict.get)
min_value = min(my_dict.values())
print("The Least MSE is ", "\"", min_key, "\"", " = ", min_value)
print("____________________________________________________________")
# print the Best Feature
print("The BEST Feature is -------> ", "\"", min_key, "\"")
print("____________________________________________________________")


# for name, value in my_dict.items():
#     print(name, ":", value)
