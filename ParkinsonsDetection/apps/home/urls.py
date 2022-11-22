"""
URL configuration for Home app
"""

from django.urls import path
from . import views

app_name = "home"
urlpatterns = [
    path('', views.home, name='home'),
    path('info/', views.info, name='info')
]
