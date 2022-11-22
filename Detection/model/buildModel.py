from ctypes import ArgumentError
import numpy as np # Calculate stats and grab rand indices

from imutils import paths # Extract th file paths to the images
import cv2 # read, process, display images
import os # accomodate file paths

from .parkinsonModel import ParkinsonModel, quantifyImageHOG

def __loadSplit(path: str):
    """
    Load the images and their paths. Convert the images into the respective HOG features and return them along with the corresponding labels

    :param path: the path to the images to be processed
    :return: tuple where the first array is the hog features of all the images and the second tuple is the labels of the respective images
    """

    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]

        if label != "parkinson" and label != "healthy":
            continue

        image = cv2.imread(imagePath) # type: ignore

        data.append(image)
        labels.append(label)

    return (data, labels)

def buildModels(dataset_path: str, spiral_model_path: str=os.path.join("data", "model", "spiralModel"), wave_model_path: str=os.path.join("data", "model", "waveModel"), concurrent_model_nums: int = 10):
    """
    Build multiple models per spiral/wave parkinsons detection in order for to demonstarate accuracy per prediction

    :param dataset_path: the path to the datasets of both spiral and wave datasets

    :return: a tuple where the first element is a multimodel for spiral detection and the second element is a multimodel for wave detection
    """

    
    waveTrainingPath = os.path.join(dataset_path, "wave", "training")
    waveTestingPath = os.path.join(dataset_path, "wave", "testing")
    spiralTrainingPath = os.path.join(dataset_path, "spiral", "training")
    spiralTestingPath = os.path.join(dataset_path, "spiral", "testing")

    print("[INFO] loading data...")
    waveTrainX, waveTrainY = __loadSplit(waveTrainingPath)
    waveTestX, waveTestY = __loadSplit(waveTestingPath)
    spiralTrainX, spiralTrainY = __loadSplit(spiralTrainingPath)
    spiralTestX, spiralTestY = __loadSplit(spiralTestingPath)

    waveModel = ParkinsonModel()
    spiralModel = ParkinsonModel()

    waveModel.train(
        trainX=waveTrainX,
        trainY=waveTrainY,
        testX=waveTestX,
        testY=waveTestY, 
        concurrent_model_nums=concurrent_model_nums
    )
    spiralModel.train(
        trainX=spiralTrainX, 
        trainY=spiralTrainY,
        testX=spiralTestX,
        testY=spiralTestY,
        concurrent_model_nums=concurrent_model_nums
    )

    waveModel.save(wave_model_path)
    spiralModel.save(spiral_model_path)

    return spiralModel, waveModel


def loadModels(spiral_model_path: str, wave_model_path: str):
    """
    Load the models in the paths

    :param spiral_model_path: the path to the spiral model
    :param wave_model_path: the path to the wave model
    
    :return: both a spiralModel and a waveModel
    """

    if not os.path.exists(spiral_model_path) or not os.path.exists(wave_model_path):
        raise ArgumentError(f"[ERROR] paths don't exist")

    spiralModel, waveModel = ParkinsonModel(), ParkinsonModel()

    if "classifier_0" in os.listdir(spiral_model_path): # If there's at least one classifier in the dir then a model exists there
        spiralModel.load(spiral_model_path)
    else:
        raise ArgumentError(f"[ERROR] {spiral_model_path} does not contain a valid model")

    if "classifier_0" in os.listdir(wave_model_path):
        waveModel.load(wave_model_path)
    else:
        raise ArgumentError(f"[ERROR] {wave_model_path} does not contain a valid model")

    print("[INFO] Models loaded succesfully")

    return spiralModel, waveModel
