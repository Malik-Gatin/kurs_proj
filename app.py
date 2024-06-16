import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import hashlib

from image_processing import make_prediction, create_image_with_bboxes
from database import init_db, log_entry, fetch_logs, get_prediction_by_hash, clear_all_logs, clear_logs_by_parameter

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def hash_image(image):
    """
    Хэширование изображения для уникальной идентификации.
    """
    img_bytes = np.array(image).tobytes()
    return hashlib.md5(img_bytes).hexdigest()

def main():
    init_db()
    st.title("Детектор объектов")
    upload = st.file_uploader(label="Загрузите изображение:", type=["png", "jpg", "jpeg"])

    if upload:
        logger.info("Изображение загружено")
        img = Image.open(upload)
        img_array = np.array(img).transpose(2, 0, 1)
        img_hash = hash_image(img)
        logger.info(f"Хеш изображения: {img_hash}")

        # Проверка наличия изображения в базе данных
        prediction_data = get_prediction_by_hash(img_hash)
        
        if prediction_data:
            # Изображение найдено в базе данных, используем сохраненные результаты
            labels, scores, boxes = prediction_data
            prediction = {"labels": labels, "scores": scores, "boxes": boxes}
            
            # Отображение результата из базы данных
            st.header("Используем сохраненные результаты из базы данных")
            img_with_bbox = create_image_with_bboxes(img_array, prediction)
        else:
            # Изображение не найдено, делаем предсказание
            prediction = make_prediction(img)
            img_with_bbox = create_image_with_bboxes(img_array, prediction)

            # Логирование данных в базу данных``
            image_size = str(img.size)
            image_mode = img.mode
            labels = json.dumps(prediction["labels"])
            scores = json.dumps(prediction["scores"])
            boxes = json.dumps(prediction["boxes"])
            log_entry(img_hash, image_size, image_mode, labels, scores, boxes)

        # Отображение изображения с ограничительными рамками
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox.transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(False)

        st.pyplot(fig, use_container_width=True)

        del prediction["boxes"]
        st.header("Предсказанные вероятности")
        st.write(prediction)

    # Отображение журнала в боковой панели
    st.sidebar.title("Журнал")
    if st.sidebar.button("Показать журнал"):
        logs = fetch_logs()
        for log in logs:
            log_id, timestamp, image_hash, image_size, image_mode, labels, scores, boxes = log
            st.sidebar.write(f"ID: {log_id}")
            st.sidebar.write(f"Временная метка: {timestamp}")
            st.sidebar.write(f"Хеш изображения: {image_hash}")
            st.sidebar.write(f"Размер изображения: {image_size}")
            st.sidebar.write(f"Режим изображения: {image_mode}")
            st.sidebar.write(f"Метки: {labels}")
            st.sidebar.write(f"Вероятности: {scores}")
            st.sidebar.write(f"Координаты рамок: {boxes}")
            st.sidebar.write("-" * 40)

    # Очистка базы данных
    st.sidebar.title("Очистка журнала")
    if st.sidebar.button("Очистить все записи"):
        clear_all_logs()
        st.sidebar.write("Все записи очищены")

    parameter = st.sidebar.selectbox("Выберите параметр для очистки записей:", ["image_size", "image_mode", "Метки", "Вероятности"])
    value = st.sidebar.text_input(f"Введите значение для {parameter}:")
    if st.sidebar.button(f"Очистить записи по {parameter}"):
        clear_logs_by_parameter(parameter, value)
        st.sidebar.write(f"Записи очищены по {parameter} = {value}")

if __name__ == "__main__":
    main()