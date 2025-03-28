from django.urls import path
from .views import predict_winner

urlpatterns = [
    path('predict/', predict_winner, name='predict_winner'),
]
