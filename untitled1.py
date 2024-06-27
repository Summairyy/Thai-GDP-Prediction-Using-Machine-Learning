# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:45:40 2024

@author: fsmai
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load GDP data
df_gdp = pd.read_csv(r"C:\Users\fsmai\Desktop\MyProject_new\EconProject\data_GDP.csv")

# Train-test split
train_data = df_gdp.iloc[:30]
test_data = df_gdp.iloc[30:]

# Define features and target variable
features = ['Years','Export', 'Import', 'CPI', 'Exchange_rate', 'Tourist', 'GDP_fed']
target = 'GDP_thai'

# Prepare train and test data
x_train, y_train = train_data[features], train_data[target]
x_test, y_test = test_data[features], test_data[target]

# Train the linear regression model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predictions
y_pred_train = reg.predict(x_train)
y_pred_test = reg.predict(x_test)
y_pred_test

# Model evaluation
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)
print("Test RMSE:", rmse_test)
print("Test MAPE:", mape_test)

# Model coefficients and intercept
print("Intercept:", reg.intercept_)
print("Coefficients:", reg.coef_)

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(test_data['Years'], y_test, label="Actual GDP")
plt.plot(test_data['Years'], y_pred_test, label="Predicted GDP", linestyle='--')
plt.title("Actual vs Predicted GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(pd.concat([train_data['Years'],test_data['Years']]), pd.concat([y_train,y_test]), label="Actual GDP")
plt.plot(pd.concat([train_data['Years'],test_data['Years']]), np.concatenate([y_pred_train,y_pred_test]), label="Predicted GDP", linestyle='--')
plt.title("Actual vs Predicted GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.show()





















