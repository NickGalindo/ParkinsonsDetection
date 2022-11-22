from .manager.load_config import CONFIG, LOCAL
from .model.buildModel import buildModels, loadModels

import os

def getModels():
    spiralModel, waveModel = None, None

    CONFIG["SPIRAL_MODEL_LOCATION"] = os.path.join(LOCAL, CONFIG["SPIRAL_MODEL_LOCATION"])
    CONFIG["WAVE_MODEL_LOCATION"] = os.path.join(LOCAL, CONFIG["WAVE_MODEL_LOCATION"])
    CONFIG["DEFAULT_DATASET_PATH"] = os.path.join(LOCAL, CONFIG["DEFAULT_DATASET_PATH"])

    try:
        spiralModel, waveModel = loadModels(CONFIG["SPIRAL_MODEL_LOCATION"], CONFIG["WAVE_MODEL_LOCATION"])
    except Exception as e:
        print(e)
        print("[WARN] Models don't exist at specified location. Creating new models at location")

        spiralModel, waveModel = buildModels(
            CONFIG["DEFAULT_DATASET_PATH"],
            wave_model_path=CONFIG["WAVE_MODEL_LOCATION"],
            spiral_model_path=CONFIG["SPIRAL_MODEL_LOCATION"]
        )

    return spiralModel, waveModel

