import os
import logging.config
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import onnxruntime as rt

from models.item_model import HousingFeatures
from utils import config, predictFunc

logging.config.fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger(__name__)
logger.propagate = False

ENV_LOCAL = 'local'
ENV_LIVE = 'live'

CACHE: Dict[str, Any] = {}

# enable documentation for specific environment only
docs_url = '/docs' if os.getenv('ENV') == ENV_LOCAL else None
app = FastAPI(docs_url=docs_url)

sess = rt.InferenceSession(config.MODEL_PATH)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name


@app.get('/health')
async def health_check():
    content = {'Server status': 'Ok'}

    return JSONResponse(content=content)


@app.post("/predict")
def predict(
    housing_features: HousingFeatures,
):
    return predictFunc.predict(housing_features)
