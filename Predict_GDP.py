 # -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 22:24:24 2024

@author: Phitchaya Watcharasathianphan
"""
#!pip install -U scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the GDP data
df_gdp = pd.read_csv(r"C:\Users\fsmai\Desktop\MyProject_new\EconProject\data_GDP.csv")
print(df_gdp)
df_gdp.columns
#Data plot
sns.pairplot(df_gdp, kind='reg');

# Split the data into training and testing sets
train_data = df_gdp[0:30]
test_data = df_gdp[30::]
train_data.head()
test_data.head()

# Training Set
x_train = train_data[['Years','Export', 'Import', 'CPI', 'Exchange_rate',
                      'Tourist','GDP_fed']]
y_train = train_data[['GDP_thai']]
x_train.head()
y_train.head()
# Test Set
x_test = test_data[['Years','Export', 'Import', 'CPI', 'Exchange_rate', 
                    'Tourist','GDP_fed']]
y_test = test_data[['GDP_thai']]

# Fit Linaer Regression : Training Set
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
reg.score(x_train, y_train) #R-squared = 0.9965040611453452
reg.intercept_ #array([-403880.57942091])
reg.coef_ #array([[ 2.05281118e+02, -1.03293417e-04,  1.21254872e-04,
          #        -1.00134065e+01, -3.65000158e+01,  2.56006696e-02,
          #         6.43724244e-02]])
          

# Forecasting Test set
y_pre = reg.predict(x_train)
y_pre

# Dataframe of Train data and Predicted
train = pd.concat([x_train, y_train], axis = 'columns')
train.head()
y_pre = np.array(y_pre).reshape(-1)
y_pre_train = pd.concat([train.reset_index(), pd.Series(y_pre, name='Predicted')], axis='columns')
y_pre_train.head()

# Dataframe of Test data and Predicted
y_pre_test = reg.predict(x_test)
y_pre_test #array([[11045.26652262],[11747.77636892]])
test = pd.concat([x_test, y_test], axis='columns')
test.head()
y_pre_test = np.array(y_pre_test).reshape(-1)
test_pre = pd.concat([test.reset_index(), pd.Series(y_pre_test, name='Predicted')], axis='columns')
test_pre.head()

#indicator
def indicators(y,y_hat):
    mse = mean_squared_error(y,y_hat) 
    rmse = root_mean_squared_error(y,y_hat)
    mape = mean_absolute_percentage_error(y,y_hat)
    print('MSE :', mse)
    print('RMSE :', rmse)
    print('MAPE :', mape)

indicators(y_test, y_pre_test) # MSE : 447180.6304109424
                              # RMSE : 668.7156573693653
                              # MAPE : 0.057312759214320635 = 5%

# Graph
plt.figure(figsize=(10, 6))
plt.plot(pd.concat([train_data['Years'],test_data['Years']]), pd.concat([y_train,y_test]), label="Actual GDP")
plt.plot(pd.concat([train_data['Years'],test_data['Years']]), np.concatenate([y_pre,y_pre_test]), label="Predicted GDP", linestyle='--')
plt.title("Actual vs Predicted GDP (Regression)")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.show()


import statsmodels.api as sm
import statsmodels.formula.api as smf
model = smf.ols(formula='GDP_thai ~ Export + Import + CPI + Exchange_rate + Tourist + GDP_fed',
                data=df_gdp).fit()
print(model.summary())

##############################################################
# =============================================================================
# '''
#     ***ARIMA MODEL***
# '''
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Split the data into training and testing sets
train_data_arima = df_gdp.iloc[:30]
test_data_arima = df_gdp.iloc[30:]
print(train_data_arima)
# Extract target variable
y_train_arima = train_data_arima[['GDP_thai']]
y_test_arima = test_data_arima[['GDP_thai']]

# Fit ARIMA model
model = ARIMA(y_train_arima, order=(1, 1, 1))
model_fit = model.fit()

# Forecast
y_pred_train = model_fit.predict()
train_pred_df = pd.concat([train_data_arima, y_pred_train], axis = 'columns')
train_pred_df.rename(columns={'predicted_mean': 'Predicted'},inplace=True)
train_pred_df.head()

# Evaluate the model
indicators(y_train_arima, y_pred_train)
            ### MSE : 603794.7477878823
            ### RMSE : 777.0423075919884
            ### MAPE : 0.06114663143179196
            
# Forecast test data
y_pred_test = model_fit.forecast(steps=len(test_data_arima))
print(y_pred_test)
test_pred_df = pd.concat([test_data_arima.reset_index(), y_pred_test.reset_index()], axis = 'columns')
test_pred_df.rename(columns={'predicted_mean': 'Predicted'},inplace=True)
test_pred_df.drop(['index'], axis='columns', inplace=True)
test_pred_df.head()


# Evaluate the model on test data
indicators(y_test_arima, y_pred_test)
            ### MSE : 1268.8093511618954
            ### RMSE : 35.62035023918063
            ### MAPE : 0.003249293606766969

plt.figure(figsize=(10, 6))
plt.plot(pd.concat([train_data_arima['Years'],test_data_arima['Years']]),pd.concat([y_train_arima,y_test_arima]), label="Actual GDP")
plt.plot(pd.concat([train_data_arima['Years'],test_data_arima['Years']]),pd.concat([train_pred_df['Predicted'],test_pred_df['Predicted']]), label="Predicted GDP", linestyle='--')
plt.title("Actual vs Predicted GDP (ARIMA)")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.show()

print(model_fit.summary())
