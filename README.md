# Forecasting Thailand's GDP with Machine Learning Using Scikit-learn and Statsmodels Library
**This project is designed to help learn and practice the basics of Machine Learning.*

This project aims to forecast the GDP value (in billion baht) of Thailand using data from 1992 to 2023 with Machine Learning. 
The data is divided into a Training Set of 30 observations from the years 1992 to 2021, and a Test Set for the years 2022 and 2023. 
The forecast will be conducted using both Multiple Linear Regression and the ARIMA Model.

Initially, the forecast will be made using Multiple Linear Regression, 
where the dependent variable (Y) is GDP_thai (Thailand's GDP in billion baht). 
The independent variables (X) include:
- Export: Value of Thai exports in million baht
- Import: Value of Thai imports in million baht
- CPI: Consumer Price Index
- Exchange_rate: Exchange rate (baht to US dollar)
- Tourist: Number of tourists entering Thailand (in thousands)
- GDP_fed: GDP of the United States (in billion US dollars)

![image](https://github.com/Summairyy/Thai-GDP-Prediction-Using-Machine-Learning/assets/132217814/92431b19-7c39-4800-8e4b-4ecf94a949a9)

Additionally, I will forecast Thailand's GDP using the ARIMA Model,
which bases its predictions on past values and errors, as shown in the equation.
This will allow us to compare the performance of the Multiple Linear Regression model with the ARIMA model.
![image](https://github.com/Summairyy/Thai-GDP-Prediction-Using-Machine-Learning/assets/132217814/c88107c4-fe15-4193-a107-956c8471db2d)

Below is a summarized step-by-step explanation of the code's operation:

### Data Preparation and Visualization

  1.Import Libraries: The necessary libraries (numpy, pandas, matplotlib, seaborn, sklearn, statsmodels) are imported.
  
  2.Load Data: The GDP data is loaded from a CSV file into a DataFrame df_gdp.
  
  3.Data Visualization: A pair plot of the data is created using seaborn to visualize relationships between variables.
  
### Linear Regression Model

  4.Split Data: The dataset is split into training (first 30 rows) and testing sets (remaining rows).
  
  5.Select Features and Target:
  
  - For training: Features (x_train) and target (y_train) are selected.
    
  - For testing: Features (x_test) and target (y_test) are selected.
    
  6.Train Linear Regression Model:
  
  - A linear regression model is created and fitted using the training data.
    
  - The model's performance is evaluated using the R-squared value, intercept, and coefficients.
    
  7.Prediction:
  
  - Predict GDP for the training set (y_pre) and testing set (y_pre_test).
    
  - Create DataFrames to compare actual and predicted values for both sets.
    
  8.Evaluation Metrics:
  
  - Define a function indicators to calculate and print Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
    
  - Evaluate the predictions for the testing set using the indicators function.
    
  9.Visualization:
  
  - Plot the actual vs. predicted GDP values over the years for both training and testing sets.
    
### ARIMA Model

  10.Split Data for ARIMA:
  
  - The dataset is split similarly into training and testing sets.
  
  - The target variable is extracted for both sets (y_train_arima and y_test_arima).
    
  11.Train ARIMA Model:
  
  - An ARIMA model is fitted on the training set GDP data.
    
  12.Prediction:
  
  - Predict GDP for the training set and evaluate using the indicators function.
    
  - Forecast GDP for the testing set and evaluate using the indicators function.
    
  - Create DataFrames to compare actual and predicted values for both sets.
    
  13.Visualization:
  
  - Plot the actual vs. predicted GDP values over the years for both training and testing sets.
    
  14.Model Summary: Print the summary of the ARIMA model.
  

## Conclusion
  After comparing the Thailand GDP forecasts from the Multiple Linear Regression and ARIMA models, 
  it was found that the MAPE value of the ARIMA Model is 0.0032 or 0.32%, 
  while the MAPE value of the Multiple Linear Regression is 0.0573 or 5.73%. 
  It can be concluded that the ARIMA Model provides more accurate forecasts due to its smaller MAPE value
  ![image](https://github.com/Summairyy/Thai-GDP-Prediction-Using-Machine-Learning/assets/132217814/80654a63-0b05-4ba7-b12a-865d70994aee)
  ![image](https://github.com/Summairyy/Thai-GDP-Prediction-Using-Machine-Learning/assets/132217814/e92b7be5-7711-410a-b82c-ff4d6692584a)

***However, the Multiple Linear Regression Model may not be the best model because it has not addressed potential issues that may arise.

