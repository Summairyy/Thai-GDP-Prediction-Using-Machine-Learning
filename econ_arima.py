import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the GDP data
df_gdp = pd.read_csv(r"C:\Users\fsmai\Desktop\MyProject_new\EconProject\data_GDP.csv")
print(df_gdp)

# Split the data into training and testing sets
train_data_arima = df_gdp.iloc[:30]
test_data_arima = df_gdp.iloc[30:]
print(train_data_arima)
# Extract target variable
y_train_arima = train_data_arima[['GDP_thai']]
y_test_arima = test_data_arima[['GDP_thai']]

# Fit ARIMA model
model = ARIMA(y_train_arima, order=(1, 1, 1))  # Example order, you may need to tune this
model_fit = model.fit()

# Forecast
y_pred_train = model_fit.predict()
train_pred_df = pd.concat([train_data_arima, y_pred_train], axis = 'columns')
train_pred_df.rename(columns={'predicted_mean': 'Predicted'},inplace=True)
train_pred_df.head()
# Evaluate the model
mse_train = mean_squared_error(y_train_arima, y_pred_train)
mape_train = mean_absolute_percentage_error(y_train_arima, y_pred_train)

print("Train MSE:", mse_train)
print("Train MAPE:", mape_train)

# Forecast test data
y_pred_test = model_fit.forecast(steps=len(test_data_arima))
print(y_pred_test)
test_pred_df = pd.concat([test_data_arima.reset_index(), y_pred_test.reset_index()], axis = 'columns')
test_pred_df.rename(columns={'predicted_mean': 'Predicted'},inplace=True)
test_pred_df.drop(['index'], axis='columns', inplace=True)
test_pred_df.head()


# Evaluate the model on test data
mse_test = mean_squared_error(y_test_arima, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test_arima, y_pred_test)

print("Test MSE:", mse_test)
print("Test MAPE:", mape_test)

plt.figure(figsize=(10, 6))
plt.plot(pd.concat([train_data_arima['Years'],test_data_arima['Years']]),pd.concat([y_train_arima,y_test_arima]), label="Actual GDP")
plt.plot(pd.concat([train_data_arima['Years'],test_data_arima['Years']]),pd.concat([train_pred_df['Predicted'],test_pred_df['Predicted']]), label="Predicted GDP", linestyle='--')
plt.title("Actual vs Predicted GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.show()

print(model_fit.summary())