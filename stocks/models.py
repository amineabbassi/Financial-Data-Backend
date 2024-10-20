from django.db import models

class StockData(models.Model):
    symbol = models.CharField(max_length=10)
    open_price = models.DecimalField(max_digits=10, decimal_places=2)
    close_price = models.DecimalField(max_digits=10, decimal_places=2)
    high_price = models.DecimalField(max_digits=10, decimal_places=2)
    low_price = models.DecimalField(max_digits=10, decimal_places=2)
    volume = models.BigIntegerField()
    date = models.DateField()
class StockPrediction(models.Model):
    symbol = models.CharField(max_length=10)
    predicted_price = models.FloatField()
    day = models.IntegerField()
    date_predicted = models.DateField(auto_now_add=True)


    def __str__(self):
        return f"{self.symbol} - {self.date}"
