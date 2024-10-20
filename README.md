# Blackhouse Trial

## Description

Blackhouse Trial is a Django application designed to provide stock price predictions using a linear regression model. The application fetches historical stock data, backtest the data, performs predictions, and generates performance reports that compare predicted stock prices against actual prices.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Usage](#usage)


## Features

- Fetches historical stock data from the Alpha Vantage API.
- Backtesting module allows users to implement a basic trading strategy based on historical stock data.
- Uses a linear regression model to predict future stock prices.
- Generates performance reports in PDF and JSON formats.
- Provides a user-friendly API for data access and manipulation.

## Technologies Used

- **Django**: Web framework for building the application.
- **Django REST Framework**: To create RESTful APIs.
- **pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computing.
- **scikit-learn**: For machine learning algorithms.
- **Matplotlib**: For data visualization.
- **ReportLab**: For generating PDF reports.
- **PostgreSQL**: Database for storing stock data and predictions.
- **Docker**: For containerization of the application.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/amineabbassi/Financial-Data-Backend.git
   cd Financial-Data-Backend

2. **Create a virtual environment (optional but recommended):**

   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required packages:**

   pip install -r requirements.txt

## Setup


1. **Set up your PostgreSQL database:**

    Make sure you have PostgreSQL installed and create a database named blackhouse.

2. **Migrate the database:**

    python manage.py migrate

3. **Create a .env file in the root directory and add the following variables:**

    SECRET_KEY=your_secret_key
    DEBUG=True  # Set to False in production
    DATABASE_HOST = 
    DATABASE_NAME = 
    DATABASE_PASSWORD = 
    DATABASE_USER = 

4. **Run the server:**

    python manage.py runserver

## Environment Variables

The following environment variables are required for the application to run correctly:

SECRET_KEY: A secret key for your Django application.
DEBUG: Set to True for development and False for production.
DATABASE_URL: The URL for connecting to your PostgreSQL database.

## Usage

To fetch stock data, use the following endpoint:
  
   **GET /stocks/fetch-data/**

To backtest the stock data, use the following endpoint:

   **/stocks/backtest/?symbol=YOUR_STOCK_SYMBOL&initial_investment=INITIAL_INVESTMENT**

To predict stock prices, use the following endpoint:

   **POST /stocks/predict/?symbol=YOUR_STOCK_SYMBOL**

To generate a performance report, use:

   **GET /stocks/report/?symbol=YOUR_STOCK_SYMBOL**



### Notes:
- Make sure to replace placeholders like `your-username`, `your_secret_key`, and `your_password` with your actual values.
- Adjust the sections according to your project's specific needs or features.
- If there are any additional features, endpoints, or configurations unique to your application, be sure to include them in the relevant sections.


