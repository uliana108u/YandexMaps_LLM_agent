# llm_relevance_agent/agent/agent_nodes_v1.py
from baseline.llm_interface import GPTInterface
from agent.search_tools import search_info
from agent.prompt_loader import load_prompt
import logging

logger = logging.getLogger(__name__)

try:
    llm = GPTInterface()
except Exception as e:
    logger.error(f"Ошибка при создании GPTInterface: {e}")
    llm = None

def fill_prompt(template: str, **kwargs) -> str:
    return template.format(**{k: v or "—" for k, v in kwargs.items()})

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
    
    name = org.get("name", "")
    full_query = f"{name} {query}"

    try:
        search_results = search_info(full_query, use_cache=use_cache)
        
        current_reviews = org.get("reviews_summarized", "")
        combined_reviews = f"{current_reviews}\n\n{search_results}".strip()
        
        state["org"]["reviews_summarized"] = combined_reviews

        if "log" not in state:
            state["log"] = {}
        state["log"]["search_query"] = full_query
        state["log"]["search_results"] = search_results
        
    except Exception as e:
        logger.error(f"Ошибка в search_node: {e}")
        if "log" not in state:
            state["log"] = {}
        state["log"]["search_error"] = str(e)

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
        prompt = fill_prompt(
            prompt_template,
            query=query,
            name=org.get("name"),
            address=org.get("address"),
            rubric=org.get("normalized_main_rubric_name_ru"),
            reviews=org.get("reviews_summarized"),
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