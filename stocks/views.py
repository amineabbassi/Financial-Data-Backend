import os
import pandas as pd
import numpy as np
import requests
from django.shortcuts import render
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from django.conf import settings
from .models import StockData, StockPrediction
from datetime import datetime, timedelta
from django.http import JsonResponse
from .ml_utils import train_linear_regression_model, predict_stock_prices


def fetch_stock_data(symbol):
    api_key = 'QSVJLR22VTK7NNWQ'  
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full'
    
    response = requests.get(url)
    json_response = response.json()
    print(json_response) 

    # Check if data is present
    data = json_response.get('Time Series (Daily)', {})
    
    if not data:
        print("No data returned for symbol:", symbol)
        return

    # Get the date 2 years ago
    two_years_ago = datetime.now().date() - timedelta(days=2*365)

    # Save only the data for the last 2 years
    for date_str, values in data.items():
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        if date >= two_years_ago:  # Only save data from the last 2 years
            obj, created = StockData.objects.update_or_create(
                symbol=symbol,
                date=date,
                defaults={
                    'open_price': values['1. open'],
                    'close_price': values['4. close'],
                    'high_price': values['2. high'],
                    'low_price': values['3. low'],
                    'volume': values['5. volume']
                }
            )
            print(f"Saved data for {date}: {obj}, Created: {created}")

def fetch_and_save_data(request):
    symbol = request.GET.get('symbol', 'AAPL')  # Default to 'AAPL'
    fetch_stock_data(symbol)
    return JsonResponse({'status': 'Data fetched and saved successfully'})


def calculate_sma(prices, window):
    """Calculate the Simple Moving Average (SMA) over a given window."""
    return prices.rolling(window=window).mean()

def backtest_strategy(symbol, initial_investment=10000, short_window=50, long_window=200):
    """Simulate a buy/sell strategy based on moving averages."""
    
    # Get stock data from the database
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    
    # Convert the stock data to a pandas DataFrame
    data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    data.set_index('date', inplace=True)
    
    # Calculate moving averages
    data['50_day_sma'] = calculate_sma(data['close_price'], short_window)
    data['200_day_sma'] = calculate_sma(data['close_price'], long_window)

    # Initialize variables for the backtest
    cash = initial_investment  # Start with the initial investment
    shares = 0  # No shares held initially
    portfolio_value = initial_investment  # Value of cash + held stocks
    number_of_trades = 0
    max_drawdown = 0
    peak_value = initial_investment
    
    # Loop through the data to simulate buy/sell strategy
    for index, row in data.iterrows():
        if pd.isna(row['50_day_sma']) or pd.isna(row['200_day_sma']):
            continue  # Skip rows without enough data for moving averages
        
        # Buy when the price is below the 50-day moving average and we have no shares
        if row['close_price'] < row['50_day_sma'] and shares == 0:
            shares = cash // float(row['close_price'])  # Convert to float
            cash -= shares * float(row['close_price'])  # Convert to float
            number_of_trades += 1
            print(f"Buying {shares} shares at {row['close_price']} on {index}")

        # Sell when the price is above the 200-day moving average and we have shares
        elif row['close_price'] > row['200_day_sma'] and shares > 0:
            cash += shares * float(row['close_price'])  # Convert to float
            shares = 0  # Reset shares to zero
            number_of_trades += 1
            print(f"Selling all shares at {row['close_price']} on {index}")

        # Calculate the current portfolio value
        current_value = cash + shares * float(row['close_price'])  # Convert to float
        peak_value = max(peak_value, current_value)  # Track peak portfolio value
        drawdown = (peak_value - current_value) / peak_value  # Calculate drawdown
        max_drawdown = max(max_drawdown, drawdown)  # Track max drawdown

        # Update portfolio value
        portfolio_value = current_value

    # Calculate total return
    total_return = ((portfolio_value - initial_investment) / initial_investment) * 100

    # Prepare backtest results
    result = {
        'initial_investment': initial_investment,
        'final_portfolio_value': portfolio_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,  # Convert to percentage
        'number_of_trades': number_of_trades
    }

    return result

    """Simulate a buy/sell strategy based on moving averages."""
    
    # Get stock data from the database
    stock_data = StockData.objects.filter(symbol=symbol).order_by('date')
    
    # Convert the stock data to a pandas DataFrame
    data = pd.DataFrame(list(stock_data.values('date', 'close_price')))
    data.set_index('date', inplace=True)
    
    # Calculate moving averages
    data['50_day_sma'] = calculate_sma(data['close_price'], short_window)
    data['200_day_sma'] = calculate_sma(data['close_price'], long_window)

    # Initialize variables for the backtest
    cash = initial_investment  # Start with the initial investment
    shares = 0  # No shares held initially
    portfolio_value = initial_investment  # Value of cash + held stocks
    number_of_trades = 0
    max_drawdown = 0
    peak_value = initial_investment
    
    # Loop through the data to simulate buy/sell strategy
    for index, row in data.iterrows():
        if pd.isna(row['50_day_sma']) or pd.isna(row['200_day_sma']):
            continue  # Skip rows without enough data for moving averages
        
        # Buy when the price is below the 50-day moving average and we have no shares
        if row['close_price'] < row['50_day_sma'] and shares == 0:
            shares = cash // row['close_price']  # Buy as many shares as possible
            cash -= shares * row['close_price']  # Deduct the spent cash
            number_of_trades += 1
            print(f"Buying {shares} shares at {row['close_price']} on {index}")

        # Sell when the price is above the 200-day moving average and we have shares
        elif row['close_price'] > row['200_day_sma'] and shares > 0:
            cash += shares * row['close_price']  # Sell all shares
            shares = 0  # Reset shares to zero
            number_of_trades += 1
            print(f"Selling all shares at {row['close_price']} on {index}")

        # Calculate the current portfolio value
        current_value = cash + shares * row['close_price']
        peak_value = max(peak_value, current_value)  # Track peak portfolio value
        drawdown = (peak_value - current_value) / peak_value  # Calculate drawdown
        max_drawdown = max(max_drawdown, drawdown)  # Track max drawdown

        # Update portfolio value
        portfolio_value = current_value

    # Calculate total return
    total_return = ((portfolio_value - initial_investment) / initial_investment) * 100

    # Prepare backtest results
    result = {
        'initial_investment': initial_investment,
        'final_portfolio_value': portfolio_value,
        'total_return': total_return,
        'max_drawdown': max_drawdown * 100,  # Convert to percentage
        'number_of_trades': number_of_trades
    }

    return result

def backtest_view(request):
    symbol = request.GET.get('symbol', 'AAPL')  # Default to AAPL
    initial_investment = float(request.GET.get('initial_investment', 10000))
    
    result = backtest_strategy(symbol, initial_investment)
    
    return JsonResponse(result)


def predict_stock_prices_view(request):
    symbol = request.GET.get('symbol', 'AAPL')
    days_ahead = int(request.GET.get('days_ahead', 30))

    # Check if the model already exists
    model_path = os.path.join(settings.BASE_DIR, 'models', f'{symbol}_linear_model.pkl')
    if not os.path.exists(model_path):
        train_linear_regression_model(symbol)

    # Call your model prediction function
    predictions = predict_stock_prices(symbol, days_ahead)
    
    #  save predictions to the database
    for prediction in predictions:

        StockPrediction.objects.create(
        symbol=symbol,
        predicted_price=prediction['predicted_price'],
        day=prediction['day']
        )

    # Return the predictions as a JSON response
    return JsonResponse({'predictions': predictions})


def generate_report_view(request):
    symbol = request.GET.get('symbol', 'AAPL')  # Get stock symbol from request

    # Fetch actual and predicted prices
    actual_data = StockData.objects.filter(symbol=symbol).order_by('date')
    predictions = StockPrediction.objects.filter(symbol=symbol).order_by('date_predicted')[:30]

    # Convert to DataFrame for analysis
    actual_prices = pd.DataFrame(list(actual_data.values('date', 'close_price')))
    predicted_prices = pd.DataFrame(list(predictions.values('date_predicted', 'predicted_price')))

    # Ensure the date columns are correctly named in both DataFrames
    if 'date' not in actual_prices or 'date_predicted' not in predicted_prices:
        return JsonResponse({'error': 'Date fields are missing from the data'}, status=400)

    # Convert the columns to float if they are Decimal
    actual_prices['close_price'] = actual_prices['close_price'].astype(float)
    predicted_prices['predicted_price'] = predicted_prices['predicted_price'].astype(float)

    # Calculate metrics
    if not actual_prices.empty and not predicted_prices.empty:
        actual_avg = actual_prices['close_price'].mean()
        predicted_avg = predicted_prices['predicted_price'].mean()
        actual_min = actual_prices['close_price'].min()
        actual_max = actual_prices['close_price'].max()
        actual_std = actual_prices['close_price'].std()
    else:
        return JsonResponse({'error': 'No data available for the specified symbol'}, status=400)

    # Visualization
    plt.figure(figsize=(10, 6))

    # Convert dates to a datetime format if they're not already
    actual_prices['date'] = pd.to_datetime(actual_prices['date'])
    predicted_prices['date_predicted'] = pd.to_datetime(predicted_prices['date_predicted'])

    # Plot actual prices
    plt.plot(actual_prices['date'], actual_prices['close_price'], label='Actual Prices', color='blue')

    # Check if there are predicted prices to plot
    if not predicted_prices.empty:
       plt.plot(predicted_prices['date_predicted'], predicted_prices['predicted_price'], label='Predicted Prices', color='orange')

    plt.title(f'{symbol} Price Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()

    # Ensure the media directory exists
    media_dir = settings.MEDIA_ROOT
    if not os.path.exists(media_dir):
       os.makedirs(media_dir)

    # Save the plot
    image_path = os.path.join(media_dir, f'{symbol}_price_comparison.png')
    plt.savefig(image_path)
    plt.close()


    # Create PDF report
    pdf_path = os.path.join(media_dir, f'{symbol}_performance_report.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    
    # Adding detailed content to the PDF
    c.drawString(100, 750, f'Performance Report for {symbol}')
    c.drawString(100, 730, f'Average Actual Price: {actual_avg:.2f}')
    c.drawString(100, 710, f'Average Predicted Price: {predicted_avg:.2f}')
    c.drawString(100, 690, f'Minimum Actual Price: {actual_min:.2f}')
    c.drawString(100, 670, f'Maximum Actual Price: {actual_max:.2f}')
    c.drawString(100, 650, f'Standard Deviation of Actual Prices: {actual_std:.2f}')
    
    c.drawImage(image_path, 100, 400, width=400, height=200)

    # Adding predicted prices to PDF
    y_position = 380
    c.drawString(100, y_position, 'Predicted Prices:')
    y_position -= 20
    for index, row in predicted_prices.iterrows():
        c.drawString(100, y_position, f'Day {index + 1}: {row["predicted_price"]:.2f}')
        y_position -= 15
        if y_position < 100:  # Prevents overlap, create a new page if needed
            c.showPage()
            y_position = 750  # Reset y position for new page

    c.save()

    # Return JSON response with metrics
    metrics = {
        'average_actual_price': actual_avg,
        'average_predicted_price': predicted_avg,
        'report_url': pdf_path,
    }
    return JsonResponse({'metrics': metrics})
