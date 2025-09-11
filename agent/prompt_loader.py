# llm_relevance_agent\agent\prompt_loader.py

import os
from utils.config import AGENT_PROMPT_DIR, PROMPT_VERSION

def load_prompt(prompt_type: str, version: str = None) -> str:
    """
    Загружает текстовый промт по заданному типу и версии из директории шаблонов.

    Параметры:
        prompt_type (str): Тип промта, например "classify" или "need_search".
        version (str, optional): Версия промта, например "v1", "v2".
            Если не указано, используется значение PROMPT_VERSION из конфигурации.

    Возвращает:
        str: Содержимое текстового файла с промтом.

    Исключения:
        FileNotFoundError: Если соответствующий файл не найден по ожидаемому пути.

    Пример:
        >>> load_prompt("classify", "v1")
        "...шаблон промта..."

    Ожидаемый формат имени файла:
        {prompt_type}_{version}.txt, например: classify_v1.txt

    Расположение:
        Файлы промтов должны находиться в директории, указанной в AGENT_PROMPT_DIR (config.py).
    """
    
    version = version or PROMPT_VERSION
    path = os.path.join(AGENT_PROMPT_DIR, f"{prompt_type}_{version}.txt")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()
