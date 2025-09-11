import os
import time
import pandas as pd
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm
from agent.agent_graph import build_relevance_graph
from utils.config import RELEVANCE_COL
import logging

logger = logging.getLogger(__name__)

"""
RelevanceAgentEvaluator

Этот модуль предназначен для оценки агента, определяющего релевантность организаций заданному текстовому запросу. Агент построен с использованием графа `langgraph`, который может по необходимости выполнять внешний поиск и использовать разные версии промптов.

Основные компоненты:
- Класс `RelevanceAgentEvaluator` инициализирует граф и предоставляет методы для пакетной оценки (`evaluate_batch`) и полной оценки всего датасета (`run_full_evaluation`).
- Метод `map_response_to_label` преобразует ответ агента в числовую метку: 
    1.0 — релевантно (RELEVANT_PLUS), 
    0.0 — нерелевантно (IRRELEVANT), 
    -1.0 — ошибка или неопознанный ответ.
- Используется кеширование (`use_cache`) и указание версии промпта (`prompt_version`) для гибкости.

Результаты включают предсказания агента, логгирование шагов внутри графа, метки релевантности и метрики качества.

Требования:
- В проекте должны быть определены: 
    - `build_relevance_graph` из `agent.agent_graph`,
    - `RELEVANCE_COL` из `utils.config`.
"""


class RelevanceAgentEvaluator:
    def __init__(self, use_cache=True, prompt_version="v1"):
        try:
            self.graph = build_relevance_graph()
        except Exception as e:
            logger.error(f"Ошибка при создании графа: {e}")
            raise
        
        self.use_cache = use_cache
        self.prompt_version = prompt_version
    
    def map_response_to_label(self, response):
        """
        Маппинг ответа модели в численную метку
        """
        if not response or response == "ERROR":
            return -1.0
        
        if "RELEVANT_PLUS" in response:
            return 1.0
        elif "IRRELEVANT" in response:
            return 0.0
        else:
            return -1.0
    
    def evaluate_batch(self, batch):
        """
        Оценка батча данных
        """
        results = []
        logs = []
        
        for _, row in batch.iterrows():
            org = {
                "name": row.get("name", "—"),
                "address": row.get("address", "—"),
                "normalized_main_rubric_name_ru": row.get("normalized_main_rubric_name_ru", "—"),
                "reviews_summarized": row.get("reviews_summarized", "—"),
                "search_info": "",  # Будет заполнено в search_node
            }
            
            # Полная инициализация состояния
            inputs = {
                "query": row["text"],
                "org": org,
                "use_cache": self.use_cache,
                "prompt_version": self.prompt_version,
                "log": {},
                "response": None,
                "next_action": None
            }
            
            try:
                output = self.graph.invoke(inputs)
                results.append(output.get("response", "ERROR"))
                logs.append(output.get("log", {}))
            except Exception as e:
                logger.error(f"Ошибка при обработке строки: {e}")
                results.append("ERROR")
                logs.append({"error": str(e)})
            
            # Небольшая задержка для избежания rate limiting
            time.sleep(0.1)
        
        return results, logs
    
    def run_full_evaluation(self, data_eval, batch_size=5):  
        """
        Полная оценка на всем датасете
        """
        all_preds = []
        all_logs = []
        
        for start in tqdm(range(0, len(data_eval), batch_size), desc="Agent Evaluation"):
            batch = data_eval.iloc[start:start + batch_size]
            preds, logs = self.evaluate_batch(batch)
            all_preds.extend(preds)
            all_logs.extend(logs)
        
        # Создаем копию для безопасности
        data_eval = data_eval.copy()
        data_eval["agent_response"] = all_preds
        data_eval["agent_log"] = all_logs
        data_eval["agent_pred_relevance"] = data_eval["agent_response"].apply(self.map_response_to_label)
        
        # Более детальная статистика
        valid = data_eval[data_eval["agent_pred_relevance"] != -1.0]
        error_count = len(data_eval[data_eval["agent_pred_relevance"] == -1.0])
        
        # Статистика по использованию поиска
        search_used = sum(1 for log in all_logs if log.get("need_search_decision") == "YES")
        
        if len(valid) > 0:
            acc = (valid[RELEVANCE_COL] == valid["agent_pred_relevance"]).mean()
            print(f"Accuracy (по {len(valid)} валидным примерам): {acc:.4f}")
            print(f"Ошибок обработки: {error_count}")
            print(f"Поиск использован в {search_used} из {len(data_eval)} случаев ({search_used/len(data_eval)*100:.1f}%)")
        else:
            acc = 0.0
            print("Нет валидных предсказаний для вычисления accuracy")
        
        return data_eval, acc