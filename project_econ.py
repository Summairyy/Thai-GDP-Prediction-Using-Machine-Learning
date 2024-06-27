# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 20:55:57 2024

@author: Phitchaya Watcharasathianphan
"""
! pip install -U scikit-learn
pip show scikit-learn 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"C:\Users\fsmai\Desktop\MyProject_new\EconProject\API_NY.GDP.MKTP.CD.csv")
print(data)

y = data[data['Country Name'] =='Thailand']
y = y.drop(['Country Name','Country Code','Indicator Name','Indicator Code'], axis=1)
x = np.arange(1960, 2023).reshape(-1,1)
print(x)
df_t = y.T
print(df_t)
#df_t.head()
y_th = df_t.values.tolist()
plt.scatter(x, y_th)

## Linear Regression
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(x, y_th)
reg.intercept_, reg.coef_ #มุม (array([-1.64486086e+13]), array([[8.33860564e+09]]))

y_pre = reg.predict(x)
plt.plot(x, y_pre, 'r')
plt.scatter(x, y_th)

y_pre1 = reg.predict([[2024]])
y_pre1  #array([[4.28729198e+11]])  

## Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_fea = PolynomialFeatures(degree=2)
X = poly_fea.fit_transform(x)
reg.fit(X, y_th) 

y_pre_poly = reg.predict(X)
plt.plot(x, y_pre_poly, "r")
plt.scatter(x, y_th)

x_test = poly_fea.fit_transform([[2024],[2025]])
y_pre_poly = reg.predict(x_test) 
y_pre_poly #array([[5.86476497e+11],[6.08752336e+11]])
































