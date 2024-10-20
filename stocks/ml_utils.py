import os
from django.conf import settings
import pandas as pd
from .models import StockData
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# Fetch stock data from the database
def get_stock_data(symbol):
    stock_records = StockData.objects.filter(symbol=symbol).order_by('date')
    
    # Create a DataFrame with the data
    data = {
        'date': [record.date for record in stock_records],
        'close_price': [float(record.close_price) for record in stock_records]
    }
    df = pd.DataFrame(data)
    df.sort_values('date', inplace=True)
    
    return df

# Train linear regression model
def train_linear_regression_model(symbol='AAPL'):
    df = get_stock_data(symbol)
    
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a new column 'days' which counts days from the start
    df['days'] = (df['date'] - df['date'].min()).dt.days
    
    X = df[['days']].values  # Independent variable (days)
    y = df['close_price'].values  # Dependent variable (closing price)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Save the trained model in the 'models/' directory
    model_dir = os.path.join(settings.BASE_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{symbol}_linear_model.pkl')

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Linear regression model trained and saved as '{model_path}'")

# Predict stock prices for the next 'days_ahead' days
def predict_stock_prices(symbol='AAPL', days_ahead=30):
    # Define the path where the model is saved
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{symbol}_linear_model.pkl')
    
    # Load the model from the correct path
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    df = get_stock_data(symbol)
    # Get the number of existing records (days)
    num_days = len(df)
    
    # Create future days starting from the next day after the last one in your dataset
    future_days = np.array(range(num_days, num_days + days_ahead)).reshape(-1, 1)
    
    predicted_prices = model.predict(future_days)
    
    predictions = [{'day': int(day[0]), 'predicted_price': price} for day, price in zip(future_days, predicted_prices)]
    
    return predictions