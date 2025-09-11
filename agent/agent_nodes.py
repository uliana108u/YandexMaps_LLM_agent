# llm_relevance_agent/agent/agent_nodes.py
# Финальный агент (версия 3)
# Добавлен clean_search_results по сравнение с версией 2

"""
agent_nodes.py

Содержит узлы для графа LLM-агента, оценивающего релевантность организаций широким пользовательским запросам.
Узлы реализуют логику:
- определения необходимости внешнего поиска (decide_need_search_node)
- выполнения поиска (search_node)
- классификации релевантности (classify_node)

LLM используется через GPTInterface (обёртка над OpenAI API).
"""

from baseline.llm_interface import GPTInterface
from agent.search_tools import search_info
from agent.prompt_loader import load_prompt
import logging
import re

# Логгер для ошибок
logger = logging.getLogger(__name__)

try:
    llm = GPTInterface()
except Exception as e:
    logger.error(f"Ошибка при создании GPTInterface: {e}")
    llm = None

def fill_prompt(template: str, **kwargs) -> str:
    """
    Подставляет значения в шаблон промта.
    Все пустые значения заменяются на '—'.

    Args:
        template (str): Шаблон промта с плейсхолдерами.
        **kwargs: Аргументы для подстановки.

    Returns:
        str: Заполненный промт.
    """
    return template.format(**{k: v or "—" for k, v in kwargs.items()})

def extract_first_name(full_name: str) -> str:
    """
    Извлекает первое имя организации до точки с запятой.

    Args:
        full_name (str): Строка с полным именем.

    Returns:
        str: Первое имя без лишних символов.
    """
    if not full_name:
        return ""
    return full_name.split(";")[0].strip()

def build_search_query(org_name: str, rubric: str, address: str, user_query: str) -> str:
    """
    Формирует строку поискового запроса из нескольких атрибутов.

    Args:
        org_name (str): Название организации.
        rubric (str): Рубрика организации.
        address (str): Адрес.
        user_query (str): Исходный запрос пользователя.

    Returns:
        str: Строка запроса для поисковой системы.
    """
    first_name = extract_first_name(org_name)
    query_parts = [part.strip() for part in [first_name, rubric, address, user_query] if part and part.strip()]
    return " ".join(query_parts)

def clean_search_results(raw_results: str) -> str:
    """
    Удаляет строки с 'Missing:' из текста поисковой выдачи.

    Args:
        raw_results (str): Исходный текст результата поиска.

    Returns:
        str: Очищенный результат поиска.
    """
    if not raw_results:
        return ""
    lines = raw_results.splitlines()
    cleaned_lines = [line for line in lines if "Missing:" not in line]
    return "\n".join(cleaned_lines).strip()

def decide_need_search_node(state):
    """
    Узел агента: принимает решение, нужен ли дополнительный поиск.

    Args:
        state (dict): Состояние агента, включая `query`, `org`, `prompt_version`.

    Returns:
        dict: Обновлённое состояние с полем `next_action` ('search' или 'classify').
    """
    if not llm:
        logger.error("LLM не инициализирован")
        state["next_action"] = "classify"
        return state
    
    org = state["org"]
    query = state["query"]
    version = state.get("prompt_version", "v1")
    
    try:
        prompt_template = load_prompt("need_search", version)
        prompt = fill_prompt(
            prompt_template,
            query=query,
            name=org.get("name"),
            address=org.get("address"),
            rubric=org.get("normalized_main_rubric_name_ru"),
            reviews=org.get("reviews_summarized"),
        )
        
        decision = llm.call_gpt(prompt).strip().upper()
        
        if "log" not in state:
            state["log"] = {}
        
        state["log"]["need_search_decision"] = decision
        state["log"]["search_prompt"] = prompt
        
        state["next_action"] = "search" if "YES" in decision else "classify"
        
    except Exception as e:
        logger.error(f"Ошибка в decide_need_search_node: {e}")
        state["next_action"] = "classify"
    
    return state

def search_node(state):
    """
    Узел агента: выполняет поиск дополнительной информации об организации.

    Args:
        state (dict): Состояние агента с полями `query`, `org`, `use_cache`.

    Returns:
        dict: Обновлённое состояние с добавленным `search_info` и логами.
    """
    org = state["org"]
    query = state["query"]
    use_cache = state.get("use_cache", True)
    
    name = org.get("name", "")
    rubric = org.get("normalized_main_rubric_name_ru", "")
    address = org.get("address", "")
    
    search_query = build_search_query(name, rubric, address, query)
    
    try:
        search_results = search_info(search_query, use_cache=use_cache)
        search_results_cleaned = clean_search_results(search_results)
        
        state["org"]["search_info"] = search_results_cleaned
        
        if "log" not in state:
            state["log"] = {}
        state["log"]["search_query"] = search_query
        state["log"]["search_results"] = search_results_cleaned
        
    except Exception as e:
        logger.error(f"Ошибка в search_node: {e}")
        if "log" not in state:
            state["log"] = {}
        state["log"]["search_error"] = str(e)
        state["org"]["search_info"] = ""
    
    return state

def classify_node(state):
    """
    Узел агента: классифицирует релевантность организации запросу.

    Args:
        state (dict): Состояние агента, включая `query`, `org`, `prompt_version`.

    Returns:
        dict: Обновлённое состояние с ответом (`response`) и логами.
    """
    if not llm:
        logger.error("LLM не инициализирован")
        state["response"] = "ERROR"
        return state
    
    org = state["org"]
    query = state["query"]
    version = state.get("prompt_version", "v1")
    
    try:
        prompt_template = load_prompt("classify", version)
        search_info = org.get("search_info", "")
        reviews = org.get("reviews_summarized", "")
        
        if isinstance(search_info, str) and search_info.startswith("[ОШИБКА]"):
            search_info = "" 
        
        prompt = fill_prompt(
            prompt_template,
            query=query,
            name=org.get("name"),
            address=org.get("address"),
            rubric=org.get("normalized_main_rubric_name_ru"),
            reviews=reviews,
            search_info=search_info,
        )
        
        response = llm.call_gpt(prompt).strip()
        
        if "log" not in state:
            state["log"] = {}
        
        state["log"]["classification_prompt"] = prompt
        state["log"]["classification_response"] = response
        state["response"] = response
        
    except Exception as e:
        logger.error(f"Ошибка в classify_node: {e}")
        state["response"] = "ERROR"
        if "log" not in state:
            state["log"] = {}
        state["log"]["classification_error"] = str(e)
    
    return state
