from django.urls import path
from . import views

urlpatterns = [
    path('fetch-data/', views.fetch_and_save_data, name='fetch_data'),  # Route for fetching data
    path('backtest/', views.backtest_view, name='backtest'),  # Route for backtesting
    path('predict/', views.predict_stock_prices_view, name='predict_stock'),  # Route for predictions
    path('report/', views.generate_report_view, name='generate_report'),  # Route for generating the report

]