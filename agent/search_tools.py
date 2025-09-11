# llm_relevance_agent\agent\search_tools.py
import os
import json
import hashlib
import logging
from dotenv import load_dotenv
from utils.config import SEARCH_CACHE_DIR

# ✅ ДОБАВЛЕНО: Безопасный импорт Tavily
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logging.warning("Tavily не установлен. Поиск будет возвращать заглушку.")

# Убедимся, что .env загружен
load_dotenv()

logger = logging.getLogger(__name__)

# ✅ ДОБАВЛЕНО: Создание директории с обработкой ошибок
try:
    os.makedirs(SEARCH_CACHE_DIR, exist_ok=True)
except Exception as e:
    logger.error(f"Не удалось создать директорию кэша: {e}")

def search_info(query: str, use_cache: bool = True) -> str:
    """
    Выполняет поиск информации по текстовому запросу через Tavily API с поддержкой кэширования.

    Если Tavily не установлен, API-ключ не найден или происходит ошибка,
    возвращается заглушка или сообщение об ошибке.

    Параметры:
        query (str): Текст запроса для поиска (должен быть непустым).
        use_cache (bool, optional): Использовать ли кэш для избежания повторных запросов. По умолчанию True.

    Возвращает:
        str: Сниппеты (фрагменты) текста из результатов поиска, объединённые через двойной перевод строки.
             Если используется кэш — из кэша. Если нет Tavily или API недоступен — возвращается заглушка/ошибка.

    Кэширование:
        - Использует md5-хэш от запроса как имя файла.
        - Кэш хранится в директории, указанной в `SEARCH_CACHE_DIR` из `config.py`.
        - Результаты сохраняются как JSON с ключом "results".

    Обработка ошибок:
        - Безопасная загрузка `.env` и API ключа.
        - Логгируются ошибки чтения/записи кэша и обращения к API.

    Пример:
        >>> search_info("кафе с завтраками на арбате")
        "Заведение X предлагает завтраки ежедневно с 8:00...\\n\\nЗаведение Y находится недалеко от Арбата..."

    Зависимости:
        - Требуется TavilyClient и переменная окружения TAVILY_API_KEY.
    """
    if not query.strip():
        return ""
    
    cache_key = hashlib.md5(query.encode("utf-8")).hexdigest()
    cache_path = os.path.join(SEARCH_CACHE_DIR, f"{cache_key}.json")

    # ✅ ДОБАВЛЕНО: Обработка ошибок при чтении кэша
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
                return cached_data.get("results", "")
        except Exception as e:
            logger.error(f"Ошибка при чтении кэша: {e}")

    # ✅ ДОБАВЛЕНО: Проверка доступности Tavily
    if not TAVILY_AVAILABLE:
        logger.warning("Tavily недоступен, возвращаем заглушку")
        return f"[ЗАГЛУШКА] Результаты поиска для: {query}"

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        logger.error("TAVILY_API_KEY не найден в .env файле")
        return f"[ОШИБКА] Не найден API ключ для поиска по запросу: {query}"

    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        result = tavily.search(query=query, max_results=3)
        snippets = "\n\n".join([r.get("content", "") for r in result.get("results", [])])
        
        # ✅ ДОБАВЛЕНО: Обработка ошибок при сохранении кэша
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"results": snippets}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении в кэш: {e}")
        
        return snippets
        
    except Exception as e:
        logger.error(f"Ошибка при выполнении поиска: {e}")
        return f"[ОШИБКА] Не удалось выполнить поиск по запросу: {query}"