# agent_runner.py

import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Подавляем лишние логи от httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

def main(version="v1", batch_size=5):
    # Добавляем корень проекта в PYTHONPATH
    from utils.config import BASE_DIR
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    # Импорт после добавления BASE_DIR
    from utils.data_loader import load_dataset
    from utils.config import (
        DATA_PATH, AGENT_RESULTS_DIR, ENV_PATH,
        validate_config, create_directories
    )
    from agent.eval_agent import RelevanceAgentEvaluator

    # Загрузка переменных окружения
    load_dotenv(ENV_PATH)

    # Проверка конфигурации
    issues = validate_config()
    if issues:
        raise RuntimeError("Обнаружены ошибки конфигурации:\n" + "\n".join(issues))

    # Создание директорий
    create_directories()

    # Загрузка данных
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Файл с данными не найден: {DATA_PATH}")
    train_data, val_data, test_data = load_dataset(DATA_PATH, drop_uncertain=True, val_frac=0.01)
    print(f" Данные загружены. Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Инициализация агента
    agent_evaluator = RelevanceAgentEvaluator(use_cache=True, prompt_version=version)

    # Оценка на валидации
    print(f"\n Запуск на валидации (версия промта: {version})...")
    val_preds, val_acc = agent_evaluator.run_full_evaluation(val_data, batch_size=batch_size)
    print(f" Validation accuracy: {val_acc:.4f}")

    # Оценка на тесте
    print(f"\n Запуск на тесте (версия промта: {version})...")
    test_preds, test_acc = agent_evaluator.run_full_evaluation(test_data, batch_size=batch_size)
    print(f" Test accuracy: {test_acc:.4f}")

    # Сохранение результатов
    val_filename = os.path.join(AGENT_RESULTS_DIR, f"agent_val_predictions_{version}.csv")
    test_filename = os.path.join(AGENT_RESULTS_DIR, f"agent_test_predictions_{version}.csv")

    val_preds.to_csv(val_filename, index=False)
    test_preds.to_csv(test_filename, index=False)
    print(f" Результаты сохранены в:\n- {val_filename}\n- {test_filename}")

# Точка входа при запуске из командной строки
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск агента для оценки релевантности.")
    parser.add_argument("--version", type=str, default="v1", help="Версия промта для агента (например: v1, v2, v3)")
    parser.add_argument("--batch_size", type=int, default=5, help="Размер batch'а для инференса")
    args = parser.parse_args()

    # Вызов основного метода
    main(version=args.version, batch_size=args.batch_size)
