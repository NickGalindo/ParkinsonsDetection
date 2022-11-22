"""
URL configuration for detect app
"""

from django.urls import path
from . import views

app_name = "detect"
urlpatterns = [
    path('instructions/', views.instructions, name='instructions'),
    path('detection/', views.detection, name='detection'),
    path('process/', views.process, name='process'),
    path('results/', views.results, name='results')
]
