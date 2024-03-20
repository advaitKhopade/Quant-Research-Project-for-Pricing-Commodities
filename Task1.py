import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

def load_and_prepare_data(csv_file):
    # Load the data from the CSV file, ensuring dates are parsed correctly
    data = pd.read_csv(csv_file, parse_dates=['Dates'], date_parser=lambda x: pd.to_datetime(x, format='%m/%d/%y'))
    data.set_index('Dates', inplace=True)

    # Create a numerical time column for regression analysis
    data['Time'] = np.arange(len(data.index))

    return data

def train_linear_regression_model(data):
    # Train the Linear Regression Model
    X = data[['Time']]  # Independent variable
    y = data['Prices']  # Dependent variable
    model = LinearRegression().fit(X, y)
    return model

def extrapolate_future_data(data, model, months=12):
    # Extrapolate for future dates
    last_time_point = data['Time'].iloc[-1]
    future_time_points = np.arange(last_time_point + 1, last_time_point + 1 + months)
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=months, freq='M')  # Updated freq from 'ME' to 'M'
    future_prices = model.predict(future_time_points.reshape(-1, 1))

    # Combine future dates and prices into a DataFrame
    future_data = pd.DataFrame({'Dates': future_dates, 'Predicted Price': future_prices}).set_index('Dates')
    return future_data

def estimate_price(data, model, date):
    date = pd.to_datetime(date)
    delta = (date - data.index[0])
    time_point = delta.days / 30.0  # Now you get the total number of days, then divide
    # Ensure input for prediction is consistent with training
    time_point_df = pd.DataFrame([[time_point]], columns=['Time'])
    return model.predict(time_point_df)[0]

# Example usage
csv_file = "Nat_Gas.csv"
data = load_and_prepare_data(csv_file)
model = train_linear_regression_model(data)
future_data = extrapolate_future_data(data, model, months=12)

# Visualizing original and extrapolated data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Prices'], label='Original Data')
plt.plot(future_data.index, future_data['Predicted Price'], label='Extrapolated Data', linestyle='solid')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices: Historical and Extrapolated')
plt.legend()
plt.show()

# Getting user input for date and estimating price
user_date = input("Enter a date (MM/DD/YYYY) to estimate the price of natural gas: ")
print("Estimated price on", user_date, ":", estimate_price(data, model, user_date))
