from django.urls import path
from .views import api_prediccion

urlpatterns = [
    path("pred", api_prediccion),
]