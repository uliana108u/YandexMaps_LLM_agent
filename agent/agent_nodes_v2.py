# llm_relevance_agent/agent/agent_nodes_v2.py
"""
В отличие от версии 1: 
Отзывы (reviews_summarized) и результаты поиска (search_info) хранятся раздельно.
В запрос отдаем только первое название организации и формиируем из названия, рубрики, адреса и запроса
Результаты поиска кладутся в org["search_info"] - разделяем источники, чтобы улучшить контроль над контекстом в LLM-промте
"""
from baseline.llm_interface import GPTInterface
from agent.search_tools import search_info
from agent.prompt_loader import load_prompt
import logging
import re

logger = logging.getLogger(__name__)

try:
    llm = GPTInterface()
except Exception as e:
    logger.error(f"Ошибка при создании GPTInterface: {e}")
    llm = None

def fill_prompt(template: str, **kwargs) -> str:
    return template.format(**{k: v or "—" for k, v in kwargs.items()})

# В отличие от прежней версии извлекаем не все варианты названий в запрос
def extract_first_name(full_name: str) -> str:
    """
    Извлекает первое название до точки с запятой
    """
    if not full_name:
        return ""
    return full_name.split(";")[0].strip()

def build_search_query(org_name: str, rubric: str, address: str, user_query: str) -> str:
    """
    Формирует поисковый запрос из компонентов
    """
    first_name = extract_first_name(org_name)
    
    # Убираем лишние пробелы и объединяем компоненты
    query_parts = [part.strip() for part in [first_name, rubric, address, user_query] if part and part.strip()]
    
    return " ".join(query_parts)

def decide_need_search_node(state):
    """
    Определяет, нужен ли поиск дополнительной информации
    Возвращает next_action = "search" или "classify".
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
    Выполняет поиск дополнительной информации 
    """
    org = state["org"]
    query = state["query"]
    use_cache = state.get("use_cache", True)
    
    # Формируем улучшенный поисковый запрос
    name = org.get("name", "")
    rubric = org.get("normalized_main_rubric_name_ru", "")
    address = org.get("address", "")
    
    # Формируем поисковый запрос
    search_query = build_search_query(name, rubric, address, query)
    
    try:
        search_results = search_info(search_query, use_cache=use_cache)
        
        # Сохраняем информацию из поиска отдельно
        state["org"]["search_info"] = search_results
        
        if "log" not in state:
            state["log"] = {}
        state["log"]["search_query"] = search_query
        state["log"]["search_results"] = search_results
        
    except Exception as e:
        logger.error(f"Ошибка в search_node: {e}")
        if "log" not in state:
            state["log"] = {}
        state["log"]["search_error"] = str(e)
        # В случае ошибки устанавливаем пустую строку
        state["org"]["search_info"] = ""
    
    return state

def classify_node(state):
    """
    Классифицирует релевантность организации
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
        
        # Разделяем информацию из поиска и отзывы
        search_info = org.get("search_info", "")
        reviews = org.get("reviews_summarized", "")

        # [FIX] Убираем сообщение об ошибке из search_info
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