import torch
from torchvision.utils import draw_bounding_boxes
import logging

from model import model, img_preprocess, categories

logger = logging.getLogger(__name__)

def make_prediction(img):
    logger.info("Предварительная обработка изображения")
    img_processed = img_preprocess(img)
    logger.info("Составление прогноза")
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    logger.info("Создание изображения с ограничивающими рамками")
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, 
        boxes=prediction["boxes"], 
        labels=prediction["labels"],
        colors=["red" if label == "person" else "green" for label in prediction["labels"]], 
        width=4,
        font="arial.ttf",
        font_size=22
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np