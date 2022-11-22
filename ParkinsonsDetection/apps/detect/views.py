from contextlib import contextmanager
from typing import Any

from django.shortcuts import render, redirect
from django.conf import settings

from Detection.detection import getModels
from .forms import ImagesForm

from PIL import Image
import numpy as np
import cv2
import base64
import secrets
import os

# Create your views here.

def instructions(request: Any):
    context = {"base_template": "basetemplate.html"}

    return render(request=request, template_name='detect/instructions.html', context=context)

def detection(request: Any):
    context = {"base_template": "basetemplate.html"}

    form = ImagesForm()

    if request.method == "POST":
        form = ImagesForm(request.POST, request.FILES)
        if form.is_valid():
            data = form.cleaned_data

            spiral_img = Image.open(data["spiral_img"])
            wave_img = Image.open(data["wave_img"])

            request.session['spiral_img'] = np.array(spiral_img).tolist()
            request.session['wave_img'] = np.array(wave_img).tolist()

            return redirect('detect:process', permanent=True)

    context["form"] = form

    return render(request=request, template_name='detect/detection.html', context=context)

def process(request: Any):
    context = {"base_template": "basetemplate.html"}

    return render(request=request, template_name='detect/process.html', context=context)

def results(request: Any):
    context = {"base_template": "basetemplate.html"}
    
    spiral_img = np.array(request.session['spiral_img'])
    wave_img = np.array(request.session['wave_img'])

    spiral_img = spiral_img.astype(np.uint8)
    wave_img = wave_img.astype(np.uint8)

    spiral_img = cv2.cvtColor(spiral_img, cv2.COLOR_RGB2BGR) # type: ignore
    wave_img = cv2.cvtColor(wave_img, cv2.COLOR_RGB2BGR) # type: ignore

    spiral_img_path = f"f.png"
    wave_img_path = f"g.png"

    cv2.imwrite(os.path.join(settings.BASE_DIR, "ParkinsonsDetection", "static", "imgs", spiral_img_path), spiral_img)
    cv2.imwrite(os.path.join(settings.BASE_DIR, "ParkinsonsDetection", "static", "imgs",wave_img_path), wave_img)

    spiral_model, wave_model = getModels()

    spiral_res = spiral_model.predict_proba([spiral_img])
    wave_res = wave_model.predict_proba([wave_img])

    context["spiral_result"] = spiral_res[0][0]
    context["spiral_confidence"] = f"{spiral_res[1][0]*100:.2f}"
    context["spiral_accuracy"] = f"{spiral_model.accuracy*100:.2f}"

    if spiral_res[0][0] == "parkinson":
        context["spiral_bayes"] = (spiral_model.accuracy * 0.04)/(spiral_model.accuracy*0.04 + (1-spiral_model.accuracy)*(1-0.04))
    else:
        context["spiral_bayes"] = (spiral_model.accuracy * (1-0.04))/(spiral_model.accuracy*(1-0.04) + (1-spiral_model.accuracy)*0.04)
    context["spiral_bayes"] = f"{context['spiral_bayes']*100:.2f}"

    context["wave_result"] = wave_res[0][0]
    context["wave_confidence"] = f"{wave_res[1][0]*100:.2f}"
    context["wave_accuracy"] = f"{wave_model.accuracy*100:.2f}"

    if wave_res[0][0] == "parkinson":
        context["wave_bayes"] = (wave_model.accuracy * 0.04)/(wave_model.accuracy*0.04 + (1-wave_model.accuracy)*(1-0.04))
    else:
        context["wave_bayes"] = (wave_model.accuracy * (1-0.04))/(wave_model.accuracy*(1-0.04) + (1-wave_model.accuracy)*0.04)
    context["wave_bayes"] = f"{context['wave_bayes']*100:.2f}"

    print(wave_res)
    print(spiral_res)

    context["precaution"] = True if context["spiral_result"] == "parkinson" or context["wave_result"] == "parkinson" else False

    return render(request=request, template_name='detect/results.html', context=context)
