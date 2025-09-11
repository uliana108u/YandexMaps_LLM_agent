# run_baseline.py

import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv

"""
run_baseline.py

Скрипт для запуска бейзлайна по оценке релевантности организаций текстовому запросу с использованием LLM (например, GPT).

Основной функционал:
1. Загружает переменные окружения из `.env` (в частности, `OPENAI_API_KEY`).
2. Загружает датасет (train/val/test) с помощью `load_dataset`, с опцией фильтрации неуверенных примеров.
3. Инициализирует бейзлайн-модель `RelevanceBaseline`, использующую шаблонные промпты и вызов LLM.
4. Выполняет инференс на валидационной и тестовой выборках.
5. Сохраняет предсказания в CSV-файлы в директории `EXPERIMENTS_DIR`.

Параметры командной строки:
--batch_size:     размер батча для LLM-инференса (по умолчанию 5)
--data_path:      путь к входному CSV-файлу (если не указан, используется дефолтный из `config.py`)
--output_prefix:  префикс для файлов с результатами (по умолчанию: "baseline")

Пример запуска:
python run_baseline.py --batch_size 10 --data_path data/dataset.csv --output_prefix gpt4_baseline
"""

def main(args):
    # --- 1. Настройка окружения и импорт ---
    try:
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        CURRENT_DIR = os.getcwd()

    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from utils.data_loader import load_dataset
    from utils.config import DATA_PATH, EXPERIMENTS_DIR, ENV_PATH
    from baseline.core import RelevanceBaseline

    # --- 2. Загрузка API ключа ---
    load_dotenv(ENV_PATH)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key is None:
        raise ValueError("OPENAI_API_KEY не найден в .env файле")

    print("OPENAI_API_KEY загружен.")

    # --- 3. Загрузка данных ---
    data_path = args.data_path or DATA_PATH
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Файл с данными не найден: {data_path}")

    train_data, val_data, test_data = load_dataset(data_path, drop_uncertain=True, val_frac=0.01)
    print(f"Данные загружены. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # --- 4. Инициализация бейзлайна ---
    baseline = RelevanceBaseline()

    # --- 5. Валидация ---
    print("Запуск на валидации...")
    val_preds, val_acc = baseline.run_full_evaluation(val_data, batch_size=args.batch_size)
    print(f"Validation accuracy: {val_acc:.4f}")

    # --- 6. Тест ---
    print("Запуск на тесте...")
    test_preds, test_acc = baseline.run_full_evaluation(test_data, batch_size=args.batch_size)
    print(f"Test accuracy: {test_acc:.4f}")

    # --- 7. Сохранение ---
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    val_file = os.path.join(EXPERIMENTS_DIR, f"{args.output_prefix}_val_predictions.csv")
    test_file = os.path.join(EXPERIMENTS_DIR, f"{args.output_prefix}_test_predictions.csv")
    val_preds.to_csv(val_file, index=False)
    test_preds.to_csv(test_file, index=False)
    print(f"Результаты сохранены: {val_file}, {test_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск бейзлайна для оценки релевантности.")
    parser.add_argument("--batch_size", type=int, default=5, help="Размер батча для инференса")
    parser.add_argument("--data_path", type=str, default=None, help="Путь к CSV с датасетом")
    parser.add_argument("--output_prefix", type=str, default="baseline", help="Префикс для сохранённых файлов")
    args = parser.parse_args()
    main(args)
