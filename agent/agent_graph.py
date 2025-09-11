# llm_relevance_agent/agent/agent_graph.py
from typing import TypedDict, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from agent.agent_nodes import decide_need_search_node, search_node, classify_node
 
class AgentState(TypedDict):
    query: str
    org: Dict[str, Any]
    log: Dict[str, Any]
    response: Optional[str]
    use_cache: bool
    prompt_version: str
    next_action: Optional[str]  # Для условных переходов

def build_relevance_graph():
    """
    Строит и компилирует граф агента для оценки релевантности организации запросу.

    Граф состоит из следующих узлов:
    - decide_need_search: принимает решение, требуется ли внешний поиск для понимания запроса.
    - search: выполняет внешний поиск (например, через Tavily) и добавляет дополнительный контекст.
    - classify: классифицирует релевантность организации запросу на основе доступной информации.

    Управляющая логика:
    - Старт в узле "decide_need_search".
    - Переход либо напрямую в "classify", либо сначала в "search", затем в "classify" — 
      в зависимости от значения поля `next_action` в состоянии.

    Возвращает:
        Скомпилированный объект графа агента (`CompiledGraph`), готовый к запуску.
    """
    # Передаем типизированное состояние
    builder = StateGraph(AgentState)
    
    # Добавляем узлы с правильными именами
    builder.add_node("decide_need_search", decide_need_search_node)
    builder.add_node("search", search_node)
    builder.add_node("classify", classify_node)
    
    #  точка входа
    builder.set_entry_point("decide_need_search")
    
    # Условные переходы через функцию
    def route_decision(state: AgentState) -> str:
        return state.get("next_action", "classify")
    
    builder.add_conditional_edges(
        "decide_need_search", 
        route_decision,
        {"search": "search", "classify": "classify"}
    )
    builder.add_edge("search", "classify")
    builder.add_edge("classify", END)
    
    return builder.compile()