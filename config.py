from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

# Конфигурационные параметры
WEIGHTS = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
BOX_SCORE_THRESH = 0.5
IMAGE_SIZE = (500, 500)