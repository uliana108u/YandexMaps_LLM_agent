import pandas as pd
from typing import Tuple
from utils.config import RELEVANCE_COL, RANDOM_STATE

def _filter_uncertain(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтрует строки с неопределенной релевантностью (0.1)."""
    return df[df[RELEVANCE_COL] != 0.1]

def load_dataset(
    path: str,
    drop_uncertain: bool = True,
    val_frac: float = 0.2,
    test_size: int = 570,  # Задано условиями учебного проекта
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает jsonl и делит на train/val/test с настраиваемым размером теста.

    Параметры:
    - path: путь к файлу .jsonl
    - drop_uncertain: удалять ли строки с релевантностью 0.1
    - val_frac: доля валидации от train данных (после выделения теста)
    - test_size: количество строк в тестовой выборке (по умолчанию 570)
    - random_state: для воспроизводимости разбиения

    Возвращает:
    Кортеж (train_data, val_data, test_data)
    """
    
    # Загрузка данных
    data = pd.read_json(path, lines=True)
    data.columns = data.columns.str.lower()

    # Валидация данных
    if RELEVANCE_COL not in data.columns:
        raise ValueError(f"Столбец {RELEVANCE_COL} не найден в данных")
    
    if len(data) < test_size:
        raise ValueError(f"Данные должны содержать как минимум {test_size} строк для выделения теста")

    # Разбиение на тест и временный train
    test_data = data.iloc[:test_size].copy()
    temp_train = data.iloc[test_size:].copy()

    # Фильтрация неопределенных значений
    if drop_uncertain:
        temp_train = _filter_uncertain(temp_train)
        test_data = _filter_uncertain(test_data)

    # Разделение временного train на train и validation
    val_data = temp_train.sample(frac=val_frac, random_state=random_state)
    train_data = temp_train.drop(val_data.index)

    return (
        train_data.reset_index(drop=True),
        val_data.reset_index(drop=True),
        test_data.reset_index(drop=True)
    )