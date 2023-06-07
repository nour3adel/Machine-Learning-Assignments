import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model

# loading the dataset
dataset = pd.read_csv('SuperMarketSales.csv',
                      parse_dates=['Date'], dayfirst=True)

# Extract New Featues From Date Column
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month
# Delete The Date Column after we create the new feature.
dataset.drop(['Date'], axis=1, inplace=True)
X = dataset.drop(['Weekly_Sales'], axis=1)  # Features
Y = dataset['Weekly_Sales']  # Label

print(X)
# Feature Selection
SupermarketCorr = dataset.corr()
# Top 50% Correlation training features with the Value
Common_features = SupermarketCorr.index[abs(
    SupermarketCorr['Weekly_Sales']) > 0.05]
# Correlation plot
plt.subplots(figsize=(12, 8))
top_corr = dataset[Common_features].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
Top = Common_features.delete(-2)
X = X[Top]
print(X)
# create_poly_Combinations


def PolynomailRegression(X, degree):
    # Convert pandas dataframe to numpy array
    if isinstance(X, pd.DataFrame):
        X = X.values
    # Get number of samples and features
    n_samples, n_features = X.shape
    # Generate all possible combinations of features up to the given degree
    Poly_Combinations = []
    for i in range(degree+1):
        for j in range(n_features**i):
            comb = []
            for k in range(i):
                comb.append(j // (n_features**k) % n_features)
            Poly_Combinations.append(comb)
    # Remove duplicate combinations
    Poly_Combinations = list(set([tuple(sorted(c))
                             for c in Poly_Combinations]))
    Poly_Combinations.sort()
    # Calculate total number of features after polynomial expansion
    feature_counting = len(Poly_Combinations)
    # Create a new numpy array to store polynomial-expanded feature matrix
    X_new = np.empty((n_samples, feature_counting))
    # Calculate polynomial-expanded features using the product of feature combinations
    for i, comb in enumerate(Poly_Combinations):
        X_new[:, i] = np.prod(X[:, comb], axis=1)

    return X_new


# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)
# Define the degree of the polynomial regression model
degree = int(input("What's The degree of the polynomial regression ?  "))
# transforms the existing features to higher degree features.
X_train_Polynomial = PolynomailRegression(X_train, degree)
# fit the transformed features to Linear Regression
PolynomialRegression_Model = linear_model.LinearRegression()
PolynomialRegression_Model.fit(X_train_Polynomial, y_train)
X_test_Polynomial = PolynomailRegression(X_test, degree)
# Predict
Y_Prediction = PolynomialRegression_Model.predict(X_test_Polynomial)
# Print Mean Square Error
MeanSquareError = mean_squared_error(y_test, Y_Prediction)
print("Mean Squared Error Degree ", degree, " :", MeanSquareError)
