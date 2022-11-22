from typing import Any

from django.shortcuts import render, redirect

# make the home view
def home(request: Any):
    context = {"base_template": "basetemplate.html"}

    return render(request=request, template_name='home/home.html', context=context)

def info(request: Any):
    context = {"base_template": "basetemplate.html"}

    return render(request=request, template_name='home/info.html', context=context)

