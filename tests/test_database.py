import sqlite3
import os
import pytest
from database import init_db, log_entry, fetch_logs

@pytest.fixture
def setup_db():
    # Удаление существующей базы данных и инициализация новой
    if os.path.exists('predictions.db'):
        os.remove('predictions.db')
    init_db()
    yield
    # Удаление базы данных после тестов
    if os.path.exists('predictions.db'):
        os.remove('predictions.db')

def test_log_entry(setup_db):
    input_data = '{"image_size": [500, 500], "image_mode": "RGB"}'
    output_data = '{"boxes": [], "labels": [], "scores": []}'
    log_entry(input_data, output_data)

    logs = fetch_logs()
    assert len(logs) == 1
    assert logs[0][2] == input_data
    assert logs[0][3] == output_data

def test_fetch_logs(setup_db):
    input_data = '{"image_size": [500, 500], "image_mode": "RGB"}'
    output_data = '{"boxes": [], "labels": [], "scores": []}'
    log_entry(input_data, output_data)

    logs = fetch_logs()
    assert len(logs) == 1
    assert logs[0][2] == input_data
    assert logs[0][3] == output_data