from ctypes import ArgumentError
from typing import Dict, List
from scipy.sparse import data

from sklearn.ensemble import RandomForestClassifier # The classifier
from sklearn.preprocessing import LabelEncoder # encode labels as integers
from sklearn.metrics import confusion_matrix # confusion matrix to compute accuracy of classification
from skimage import feature # Histogram of Oriented Gradients
import numpy as np # Calculate stats and grab rand indices

import cv2

import joblib
import os
from pathlib import Path

def quantifyImageHOG(image: np.ndarray):
    """
    Compute the histogram of oriented gradients feature vector for the input image

    :param image: the image to be converted through HOG
    :return: the HOG features of the image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # type: ignore
    image = cv2.resize(image, (200, 200)) # type: ignore
        
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # type: ignore

    features = feature.hog(image, orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1") # COmpute histogram of oriented gradients

    return features

def _optimizeMetrics(m_a: Dict[str, int], m_b: Dict[str, int]):
    """
    Return tue if b is a more optimal metric than a

    :param m_a: The first metric 
    :param m_b: The second metric

    :return: True if m_b is more optimal than m_a false otherwise
    """
    for i in ("acc", "sensitivity", "specificity"):
        if m_b[i] < m_a[i]:
            return False
        if m_b[i] > m_a[i]:
            return True
    return False

def _trainClassifier(trainX: np.ndarray, trainY: np.ndarray, testX: np.ndarray, testY: np.ndarray, forest_size: int=100):
    """
    Train the model as a Random Forest Classifier over the training X data and the training Y labels. Additionally compute metrics over these using the testing X data and the testing Y labels.

    :param trainX: the training X data of the HOG image features
    :param trainY: the labels of the training data
    :param testX: the testing X data of the HOG image features
    :param testY: the labels of the testing data

    :return: tuple where the first element is the trained model and the second element are the model metrics
    """
    model = RandomForestClassifier(n_estimators=forest_size)
    model.fit(trainX, trainY)
    
    predictions = model.predict(testX)
    metrics = {}
    
    cm = confusion_matrix(testY, predictions).flatten()

    true_negative, false_positive, false_negative, true_positive = cm

    metrics["acc"] = (true_positive + true_negative) / float(cm.sum())
    metrics["sensitivity"] = true_positive / float(true_positive + false_negative)
    metrics["specificity"] = true_negative / float(true_negative + false_positive)

    return model, metrics


def _optimizeClassifier(trainX: np.ndarray, trainY: np.ndarray, testX: np.ndarray, testY: np.ndarray, num_trials: int = 10, forest_size: int=100):
    """
    Optimize the model over several amounts of trials.

    :param trainX: the HOG features of the training data
    :param trainY: the labels of the training data
    :param testX: the HOG featurea of the testing data
    :param testY: the labels of the testing data

    :return: the most optimal model calculated
    """


    trials_metrics = {}

    max_model = None
    max_model_metrics = {
        "acc": 0,
        "sensitivity": 0,
        "specificity": 0
    }

    for i in range(0, num_trials):
        print(f"[INFO] training model {i+1} of {num_trials}")

        model, metrics = _trainClassifier(trainX, trainY, testX, testY, forest_size=forest_size)

        if _optimizeMetrics(max_model_metrics, metrics):
            max_model_metrics = metrics
            max_model = model

        for i in ("acc", "sensitivity", "specificity"):
            if i not in trials_metrics:
                trials_metrics[i] = []
            trials_metrics[i].append(metrics[i])

    assert max_model is not None

    print("\n[INFO] Optimal model")
    print("\t[DATA] metrics")
    print(f"\tAccuracy: {max_model_metrics['acc']}")
    print(f"\tSensitivity: {max_model_metrics['sensitivity']}")
    print(f"\tSpecificity: {max_model_metrics['specificity']}")

    print("\n[INFO] Total training data metrics")
    print(f"\t[DATA] Accuracy\n\t\tmean: {np.mean(trials_metrics['acc'])} std: {np.std(trials_metrics['acc'])}")
    print(f"\t[DATA] Sensitivity\n\t\tmean: {np.mean(trials_metrics['sensitivity'])} std: {np.std(trials_metrics['sensitivity'])}")
    print(f"\t[DATA] Specificity\n\t\tmean: {np.mean(trials_metrics['specificity'])} std: {np.std(trials_metrics['specificity'])}")

    return max_model, np.mean(trials_metrics['acc'])

class ParkinsonModel:
    def __init__(self):
        """
        The ParkinsonModel saves multiple random forest classifiers used for inference on parkinson image data
        """
        self.classifiers = []
        self.label_encoder = None
        self.accuracy = 0

    def addLabelEncoder(self, label_encoder: LabelEncoder):
        """
        Add a new label encoder to the model

        :param label_encoder: the encoder to add
        """
        self.label_encoder = label_encoder

    def addClassifier(self, model: RandomForestClassifier):
        """
        Add a new classifier to the model

        :param model: the new classfieir to add
        """
        self.classifiers.append(model)

    def save(self, save_path: str, compress: int=3):
        """
        Save the parkinson model to memory

        :param save_path: the path where it should be saved
        :param compress: the level of compression to be used
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)

        for i, c in enumerate(self.classifiers):
            c_path = os.path.join(save_path, f"classifier_{i}")
            joblib.dump(c, c_path, compress=compress)

        joblib.dump(self.label_encoder, os.path.join(save_path, "label_encoder"), compress=compress)
        joblib.dump(self.accuracy, os.path.join(save_path, "accuracy_rating"), compress=compress)

    def load(self, load_path: str):
        """
        Load the Parkinson Model from memory

        :param load_path: the path to load from
        """
        if "classifier_0" not in os.listdir(load_path):
            raise ArgumentError(f"[ERROR] {load_path} dos not contain a valid model")

        for c in os.listdir(load_path):
            if c.startswith("classifier_"):
                self.classifiers.append(joblib.load(os.path.join(load_path, c)))

        try:
            self.label_encoder = joblib.load(os.path.join(load_path, "label_encoder"))
        except Exception as e:
            raise ArgumentError(f"[ERROR] {load_path} does not contain a valid label encoder. Model not valid")

        try:
            self.accuracy = joblib.load(os.path.join(load_path, "accuracy_rating"))
        except Exception as e:
            raise ArgumentError(f"[ERROR] {load_path} does not contain a valid accuracy rating. Model not valid")


    def train(self, trainX: List[np.ndarray], trainY: List[str], testX: List[np.ndarray], testY: List[str], concurrent_model_nums: int = 10, forest_size: int=100):
        """
        Train the parkinson model based on some trainX, trainY, testX, testY data

        :param trainX: 
        :param trainY:
        :param testX:
        :param textY:
        """

        for i in range(len(trainX)):
            trainX[i] = quantifyImageHOG(trainX[i])

        for i in range(len(testX)):
            testX[i] = quantifyImageHOG(testX[i])

        np_trainX = np.array(trainX) 
        np_trainY = np.array(trainY) 
        np_testX = np.array(testX)
        np_testY = np.array(testY)

        le =  LabelEncoder()
        trainY = le.fit_transform(np_trainY)
        testY = le.transform(np_testY) # type: ignore

        self.label_encoder = le

        acc_arr = []

        for _ in range(concurrent_model_nums):
            curClassifier, acc = _optimizeClassifier(np_trainX, np_trainY, np_testX, np_testY, forest_size=forest_size)

            self.classifiers.append(curClassifier)
            acc_arr.append(acc)

        self.accuracy = np.mean(acc_arr)

    def predict(self, dataX: List[np.ndarray]):
        """
        Run inference on the list of images passed in by dataX. Returns a list where each element is the element with the label with the highest calculated probability

        :param dataX: A List of images (np.ndarrays)
        :return: A list of labels where each element in the list is the predicted inference for the same element in dataX
        """
        for i in range(len(dataX)):
            dataX[i] = quantifyImageHOG(dataX[i])

        assert self.label_encoder is not None
        assert len(self.classifiers) != 0

        arr = np.zeros((len(dataX), len(self.label_encoder.classes_)))
        for c in self.classifiers:
            pred = c.predict(dataX)
            for i in range(len(pred)):
                arr[i][pred[i]] += 1

        res = []
        for i in arr:
            res.append(np.argmax(i))

        return self.label_encoder.inverse_transform(res)

    def predict_proba(self, dataX: List[np.ndarray]):
        """
        Run inference on the list of images passed in by dataX. Returns two lists where the first list are the labels predicted for each image and the second list is the probability calculated for that label

        :param dataX: A List of images (np.ndarrays)
        :return: Two lists. The first one of the labels, the second of the probabilitis
        """
        for i in range(len(dataX)):
            dataX[i] = quantifyImageHOG(dataX[i])

        assert self.label_encoder is not None
        assert len(self.classifiers) != 0

        arr = np.zeros((len(dataX), len(self.label_encoder.classes_)))
        for c in self.classifiers:
            arr += c.predict_proba(dataX)

        labels = []
        probabilities = []
        for i in range(len(arr)):
            arr[i] /= len(self.classifiers)
            labels.append(np.argmax(arr[i]))
            probabilities.append(np.amax(arr[i]))

        return self.label_encoder.inverse_transform(labels), np.array(probabilities)

        


if __name__ == "__main___":
    b = [1, 2, 3]
    a = ParkinsonModel()
    a.train(b, b, b, b) # type: ignore
