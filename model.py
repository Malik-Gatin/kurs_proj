import streamlit as st
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import logging
from config import WEIGHTS, BOX_SCORE_THRESH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=WEIGHTS, box_score_thresh=BOX_SCORE_THRESH)
    model.eval()
    logger.info("Модель успешно загружена")
    return model

model = load_model()
categories = WEIGHTS.meta["categories"]
img_preprocess = WEIGHTS.transforms()