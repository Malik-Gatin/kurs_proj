import sqlite3
from datetime import datetime
import json
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """
    Инициализация базы данных.
    Создает таблицу 'logs', если она не существует.
    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  image_hash TEXT UNIQUE,
                  image_size TEXT,
                  image_mode TEXT,
                  labels TEXT,
                  scores TEXT,
                  boxes TEXT)''')
    conn.commit()
    conn.close()

def log_entry(image_hash, image_size, image_mode, labels, scores, boxes):
    """
    Логирует данные о изображении в базу данных.

    Args:
        image_hash (str): Хеш изображения.
        image_size (str): Размер изображения в формате "ширина, высота".
        image_mode (str): Режим изображения (RGB, RGBA и т.д.).
        labels (str): JSON-строка с метками объектов.
        scores (str): JSON-строка с предсказанными вероятностями объектов.
    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO logs (timestamp, image_hash, image_size, image_mode, labels, scores, boxes) VALUES (?, ?, ?, ?, ?, ?, ?)',
              (timestamp, image_hash, image_size, image_mode, labels, scores, boxes))
    conn.commit()
    conn.close()
    logger.info(f"Запись добавлена с хешем {image_hash}")

def fetch_logs():
    """
    Получает все записи из журнала.

    Returns:
        list: Список записей из базы данных.
    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('SELECT * FROM logs')
    logs = c.fetchall()
    conn.close()
    return logs

def get_prediction_by_hash(image_hash):
    """
    Получает предсказания по хешу изображения.

    Args:
        image_hash (str): Хеш изображения.

    Returns:
        tuple or None: Кортеж с метками и вероятностями или None, если запись не найдена.
    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    logger.info(f"Получение предсказаний для хеша {image_hash}")
    c.execute('SELECT labels, scores, boxes FROM logs WHERE image_hash = ?', (image_hash,))
    result = c.fetchone()
    conn.close()
    logger.info(f"Получен результат: {result}")
    if result:
        labels, scores, boxes = result
        labels = json.loads(labels)
        scores = json.loads(scores)
        boxes = json.loads(boxes)
        return labels, scores, boxes
    return None

def clear_all_logs():
    """
    Очищает все записи из журнала.
    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('DELETE FROM logs')
    conn.commit()
    conn.close()
    logger.info("Все записи в журнале очищены")

def clear_logs_by_parameter(parameter, value):
    """
    Очищает записи из журнала по заданному параметру и значению.

    Args:
        parameter (str): Параметр для фильтрации записей (image_size, image_mode, labels, scores).
        value (str): Значение параметра.

    """
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    query = f'DELETE FROM logs WHERE {parameter} = ?'
    c.execute(query, (value,))
    conn.commit()
    conn.close()
    logger.info(f"Очищены записи по параметру {parameter} = {value}")