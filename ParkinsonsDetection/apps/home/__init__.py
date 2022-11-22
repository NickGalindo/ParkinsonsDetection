from django.apps import AppConfig

class HomeConfig(AppConfig):
    name = 'ParkinsonsDetection.apps.home'
    label = 'home'
    verbose_name = 'Home'

default_app_config = 'ParkinsonsDetection.apps.home.HomeConfig'
