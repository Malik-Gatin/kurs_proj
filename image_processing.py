from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging

from model import model, img_preprocess, categories

logger = logging.getLogger(__name__)

def make_prediction(img):
    """
    Функция для предобработки изображения и получения предсказаний от модели.
    """
    logger.info("Предварительная обработка изображения")
    img_processed = img_preprocess(img)
    logger.info("Составление прогноза")
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]

    # Преобразование в сериализуемые типы данных
    prediction["boxes"] = prediction["boxes"].detach().cpu().tolist()
    prediction["scores"] = prediction["scores"].detach().cpu().tolist()

    return prediction

def create_image_with_bboxes(img, prediction, font_path="arial.ttf", font_size=20):
    """
    Функция для создания изображения с нарисованными bounding boxes и метками.
    """
    logger.info("Creating image with bounding boxes")
    img_pil = Image.fromarray(img.transpose(1, 2, 0))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    for box, label in zip(prediction["boxes"], prediction["labels"]):
        box = [int(coord) for coord in box] 
        color = "red" if label == "person" else "green"
        draw.rectangle(box, outline=color, width=4)
        # Используем textbbox для получения размеров текста
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        text_location = [box[0], box[1] - text_size[1]]
        draw.text(text_location, label, fill=color, font=font)

    return np.array(img_pil).transpose(2, 0, 1)
    