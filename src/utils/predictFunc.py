import onnxruntime as rt

from models.item_model import HousingFeatures
from utils import config
import numpy
import pickle

session = rt.InferenceSession(config.MODEL_PATH)
first_input_name = session.get_inputs()[0].name
first_output_name = session.get_outputs()[0].name

with open('ml_pipelines/features.pickle', 'rb') as f:
    feature = pickle.load(f)
    print("features:", feature)


def predict(data: HousingFeatures):
    try:
        # Extract data in correct order
        data_dict = data.dict()
        to_predict_dict = [data_dict[feature] for feature in feature]

        # dict to array
        to_predict_array = numpy.array(to_predict_dict).reshape(1, -1)
        pred_onx = session.run(
            [], {first_input_name: to_predict_array.astype(numpy.float32)})[0]
        return {"prediction": float(pred_onx[0])}
    except Exception:
        return {"prediction": "error"}
