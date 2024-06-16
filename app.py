import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging

from image_processing import make_prediction, create_image_with_bboxes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.title("Детектор Объектов")
    upload = st.file_uploader(label="Загрузите изображение здесь:", type=["png", "jpg", "jpeg"])

    if upload:
        logger.info("Изображение загружено")
        img = Image.open(upload)
        img_array = np.array(img).transpose(2, 0, 1)

        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(img_array, prediction)

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([])
        plt.yticks([])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.pyplot(fig, use_container_width=True)

        del prediction["boxes"]
        st.header("Предсказанные вероятности")
        st.write(prediction)

if __name__ == "__main__":
    main()