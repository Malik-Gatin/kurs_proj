import torch
from PIL import Image
import numpy as np
from image_processing import make_prediction, create_image_with_bboxes


def test_make_prediction():
    img = Image.new("RGB", (500, 500))
    prediction = make_prediction(img)
    assert "boxes" in prediction
    assert "labels" in prediction
    assert "scores" in prediction

def test_create_image_with_bboxes():
    img = Image.new("RGB", (500, 500))
    img_array = np.array(img).transpose(2, 0, 1)
    prediction = {
        "boxes": torch.tensor([[10, 10, 100, 100]]),
        "labels": ["person"],
        "scores": torch.tensor([0.9])
    }
    img_with_bbox = create_image_with_bboxes(img_array, prediction)
    assert img_with_bbox is not None