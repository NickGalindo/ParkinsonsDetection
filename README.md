<h1 align="center">Parkinson's Detection Tool</h1>


>Made possible by :coffee:

This tool uses a machine learning model to detect early motor symptoms of Parkinson's using only a drawing of a spiral and a wave from potential patients.

<img src="gitstatic/base_img.png" alt="Base Parkinson Example IMG" title="">

Installation
------------

### Dependencies

```
python -m pip install -r requirements.txt
```

### Setting up databases

```
python manage.py makemigrations
python manage.py migrate
```

Usage
-----

The model already comes pretrained so no further training is needed. Run the webapp and then head to `http://localhost:8000` on any web browser.

```
python manage.py runserver
```

Retrain Model
-------------

The model can be retrained on a new dataset by inserting the dataset under the `dataset` directory with the following directory structure

```
├── Detection
│   ├── config.yaml
│   ├── data
│   │   ├── dataset
│   │   │   ├── spiral
│   │   │   │   ├── testing
│   │   │   │   │   ├── healthy
│   │   │   │   │   │   └── example.png
│   │   │   │   │   └── parkinson
│   │   │   │   │       └── example.png
│   │   │   │   └── training
│   │   │   │       ├── healthy
│   │   │   │       │   └── example.png
│   │   │   │       └── parkinson
│   │   │   │           └── example.png
│   │   │   └── wave
│   │   │       ├── testing
│   │   │       │   ├── healthy
│   │   │       │   │   └── example.png
│   │   │       │   └── parkinson
│   │   │       │       └── example.png
│   │   │       └── training
│   │   │           ├── healthy
│   │   │           │   └── example.png
│   │   │           └── parkinson
│   │   │               └── example.png
```

The directories `spiralModel` and `waveModel` under the `model` directory in `Detection` must be erased for every retrain



