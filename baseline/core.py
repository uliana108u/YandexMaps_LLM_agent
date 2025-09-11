import time
import pandas as pd
from tqdm.notebook import tqdm  
from sklearn.metrics import accuracy_score
from baseline.llm_interface import GPTInterface
from baseline.prompt_templates import build_relevance_prompt
from utils.config import RELEVANCE_COL

"""
RelevanceBaseline

Этот модуль реализует базовый метод оценки релевантности организации текстовому запросу с помощью LLM.
Бейзлайн использует фиксированный шаблон промпта (`build_relevance_prompt`) без дополнительных механизмов поиска или логики.

Класс `RelevanceBaseline` предоставляет следующие возможности:
- `evaluate_batch`: вызывает LLM для небольшого батча данных, генерируя ответы на основе входных полей (название, адрес, рубрика, отзывы).
- `map_response_to_label`: преобразует ответ модели в числовую метку:
    - 1.0 — "RELEVANT_PLUS"
    - 0.0 — "IRRELEVANT"
    - -1.0 — ошибка или неизвестный ответ
- `run_full_evaluation`: запускает оценку на всем датасете, собирает предсказания, сохраняет ошибки и считает accuracy по валидным примерам.

Параметры:
- `llm_interface`: объект интерфейса LLM (по умолчанию — `GPTInterface`)

Требования:
- `GPTInterface` из `baseline.llm_interface`
- `build_relevance_prompt` из `baseline.prompt_templates`
- `RELEVANCE_COL` из `utils.config`
"""

class RelevanceBaseline:
    def __init__(self, llm_interface=None):
        self.llm = llm_interface or GPTInterface()

    def map_response_to_label(self, response):
        if "RELEVANT_PLUS" in response:
            return 1.0
        elif "IRRELEVANT" in response:
            return 0.0
        else:
            return -1.0  # ошибка или непонятный ответ

    def evaluate_batch(self, batch):
        results = []
        for _, row in batch.iterrows():
            prompt = build_relevance_prompt(
                query=row["text"],
                name=row.get("name", "—"),
                address=row.get("address", "—"),
                rubric=row.get("normalized_main_rubric_name_ru", "—"),
                reviews=row.get("reviews_summarized", "—")
            )
            response = self.llm.call_gpt(prompt)
            results.append(response)
            time.sleep(0.1)  # задержка для API
        return results

    def run_full_evaluation(self, data_eval, batch_size=5):
        all_preds = []
        all_errors = []
        
        n_batches = (len(data_eval) + batch_size - 1) // batch_size

        for start in tqdm(range(0, len(data_eval), batch_size), desc="Evaluating batches"):
            batch = data_eval.iloc[start:start + batch_size]
            responses = self.evaluate_batch(batch)
            all_preds.extend(responses)

            for i, r in enumerate(responses):
                if isinstance(r, str) and r.startswith("ERROR"):
                    error_row = batch.iloc[i].copy()
                    error_row["error_message"] = r
                    all_errors.append(error_row)

        # Сохраняем ошибки
        if all_errors:
            pd.DataFrame(all_errors).to_csv("errors.csv", index=False)
            print(f"Сохранено ошибок: {len(all_errors)}")

        data_eval["gpt_response"] = all_preds
        data_eval["gpt_pred_relevance"] = data_eval["gpt_response"].apply(self.map_response_to_label)

        valid = data_eval[data_eval["gpt_pred_relevance"] != -1.0]
        acc = accuracy_score(valid[RELEVANCE_COL], valid["gpt_pred_relevance"])
        print(f"Accuracy (по {len(valid)} примерам): {acc:.4f}")

        return data_eval, acc
