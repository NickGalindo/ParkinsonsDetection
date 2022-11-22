from django import forms
from django.db import models

class ImagesForm(forms.Form):
    spiral_img = forms.ImageField(
        widget=forms.FileInput(
            attrs={
                'class': 'form-control p-2 border-primary',
                'style': "font-family: 'Raleway', sans-serif; margin-bottom: 20px;"
            }
        )
    )
    wave_img = forms.ImageField(
        widget=forms.FileInput(
            attrs={
                'class': 'form-control p-2 border-primary',
                'style': "font-family: 'Raleway', sans-serif; margin-bottom: 20px;"
            }
        )
    )
