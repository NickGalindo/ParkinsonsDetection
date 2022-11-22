from django.apps import AppConfig

class DetectConfig(AppConfig):
    name = 'ParkinsonsDetection.apps.detect'
    label = 'detect'
    verbose_name = 'Detect'

default_app_config = 'ParkinsonsDetection.apps.detect.DetectConfig'
