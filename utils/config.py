import os
import logging

# ✅ ДОБАВЛЕНО: Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Корень проекта
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Пути к данным, env-файлу
DATA_PATH = os.path.join(BASE_DIR, "data", "data_final_for_dls_new.jsonl")
ENV_PATH = os.path.join(BASE_DIR, ".env")
RANDOM_STATE = 42

# Столбец с таргетом
RELEVANCE_COL = "relevance_new"

# --- Каталоги для экспериментов ---
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
AGENT_RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "agent")
AGENT_LOGS_DIR = os.path.join(AGENT_RESULTS_DIR, "agent_logs")
SEARCH_CACHE_DIR = os.path.join(AGENT_RESULTS_DIR, "search_cache")

# --- Агент: настройки и пути к промтам ---
AGENT_PROMPT_DIR = os.path.join(BASE_DIR, "agent", "prompts")
PROMPT_VERSION = os.getenv("AGENT_PROMPT_VERSION", "v1")

# Пути к конкретным промтам (по версии)
PROMPT_CLASSIFY_PATH = os.path.join(AGENT_PROMPT_DIR, f"classify_{PROMPT_VERSION}.txt")
PROMPT_NEED_SEARCH_PATH = os.path.join(AGENT_PROMPT_DIR, f"need_search_{PROMPT_VERSION}.txt")

# --- Агент: флаги управления ---
AGENT_USE_CACHE = os.getenv("AGENT_USE_CACHE", "true").lower() == "true"

# ✅ ДОБАВЛЕНО: Функция валидации
def validate_config():
    """
    Проверяет корректность конфигурации
    """
    issues = []
    
    # Проверка файлов
    if not os.path.exists(DATA_PATH):
        issues.append(f"Файл данных не найден: {DATA_PATH}")
    
    if not os.path.exists(ENV_PATH):
        issues.append(f"Файл .env не найден: {ENV_PATH}")
    
    # Проверка переменных окружения
    required_vars = ["OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            issues.append(f"Не найдена переменная окружения: {var}")
    
    # Проверка промтов
    if not os.path.exists(PROMPT_CLASSIFY_PATH):
        issues.append(f"Промт для классификации не найден: {PROMPT_CLASSIFY_PATH}")
    
    if not os.path.exists(PROMPT_NEED_SEARCH_PATH):
        issues.append(f"Промт для поиска не найден: {PROMPT_NEED_SEARCH_PATH}")
    
    return issues

# ✅ ДОБАВЛЕНО: Создание необходимых директорий
def create_directories():
    """
    Создает необходимые директории
    """
    dirs_to_create = [
        EXPERIMENTS_DIR,
        AGENT_RESULTS_DIR,
        AGENT_LOGS_DIR,
        SEARCH_CACHE_DIR
    ]
    
    for dir_path in dirs_to_create:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            logging.error(f"Не удалось создать директорию {dir_path}: {e}")