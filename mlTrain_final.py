import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split



def significant_acf_lag(ts_data):
    # Identify significant lags using ACF
    acf_result = sm.tsa.stattools.acf(ts_data, nlags=30)
    significant_lags_acf = np.where(np.abs(acf_result) > 2 / np.sqrt(len(ts_data)))[0]
    acf_lag = significant_lags_acf[len(significant_lags_acf) - 1]
    return acf_lag


def create_lagged_features(data, lag):
    lagged_data = data.copy()
    for i in range(1, lag + 1):
        lagged_data[f'Lag_{i}'] = data.shift(i)
       # print(lagged_data)
    return lagged_data.dropna()



def train_partial(lagged_data): 
    # Split data into train and test sets
    X = lagged_data.drop('Value', axis=1)
    y = lagged_data['Value']
    #practically, this achieved the same result if we were to used the train_data. becaause of spliting rule of 80-20
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


    # Define the model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.0001, verbosity = 3)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mpe = np.mean((y_pred - y_test) / y_test) * 100
    return mse, mpe


def train_all(ts_data, lagged_data, forecast_periods, freq, lag):
    
    # Split data into train and test sets
    X = lagged_data.drop('Value', axis=1)
    y = lagged_data['Value']


    # Define the model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.0001, verbosity = 3)

    # Train the model
    model.fit(X, y)



    #code here--------------------------- 


    # Out-of-sample forecasting
    #BASICALLY, WE'RE USING STEPPING FORECAST
    last_row = lagged_data.iloc[-1].drop('Value')
    forecasted_values = []

    for _ in range(forecast_periods):
        y_pred = model.predict(last_row.values.reshape(1, -1))[0]
        forecasted_values.append(y_pred)
        # Update last_row to include the latest prediction
        last_row = last_row.shift(1)
        last_row.iloc[0] = y_pred

    # Generate future dates based on the frequency
    frequency = freq[0]  # extract the first character of frequency (D, W, M, etc.)
    if frequency == 'D':
        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq='D')[1:]
    elif frequency == 'W':
        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq='W')[1:]
    elif frequency == 'M':
        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq='M')[1:]
    elif frequency == 'A' or frequency == 'Y':
        future_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_periods + 1, freq='Y')[1:]
    else:
        return None  # Handle unsupported frequency

    # Create DataFrame for forecasting
    forecast_df = pd.DataFrame({'Value': forecasted_values}, index=future_dates)

    return forecast_df

    