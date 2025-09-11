"""
inspect_utils.py

Модуль для интерактивного визуального анализа строк DataFrame с предсказаниями моделей.

Содержит функции:
- `inspect_row_html`: формирует HTML-контент на основе данных одной строки DataFrame, включая входные данные, предсказания, истинную метку и лог агента.
- `inspect_row`: выводит сгенерированный HTML-блок в ячейке Jupyter Notebook через IPython.display.

Применение:
Позволяет удобно просматривать кейсы, где модель ошибается, а также анализировать внутренние логи агентов agent_log (если есть). Используется в процессе отладки моделей, построения гипотез и анализа ошибок.

Аргументы:
- `df`: pandas DataFrame с полями, включая текст запроса, название, описание и предсказания.
- `idx`: индекс строки в DataFrame.
- `pred_col`: имя колонки с предсказанием модели.
- `label_col`: имя колонки с истинной меткой (по умолчанию `RELEVANCE_COL` из конфигурации).

Формат вывода:
Визуально оформленный HTML-блок с раскрывающимся логом агента, если он есть.

Зависимости:
- IPython.display
- pandas
- utils.config (для импорта `RELEVANCE_COL`)
"""

import pandas as pd
from IPython.display import display, HTML
from utils.config import RELEVANCE_COL

def inspect_row_html(row, idx, *, pred_col="pred_relevance", label_col=RELEVANCE_COL):
    """Формирует HTML для визуального анализа строки DataFrame."""
    
    def safe_get(key):
        return row[key] if key in row and pd.notna(row[key]) else "—"

    # Экранирование лога
    agent_log_html = ""
    if "agent_log" in row and pd.notna(row["agent_log"]):
        agent_log_html = f"""
        <details style="margin-top:10px;">
            <summary style="cursor:pointer;"><strong>🧠 Agent log (раскрыть)</strong></summary>
            <pre style="white-space:pre-wrap; background:#eee; padding:10px; border-radius:6px;">
{row['agent_log']}
            </pre>
        </details>
        """

    return f"""
<div style="border:1px solid #ccc; padding:16px; border-radius:10px;
            font-family:sans-serif; background-color:#f9f9f9; margin-bottom:20px">
  <h2>📌 Index: {idx}</h2>
  <p><strong>Запрос:</strong><br>{safe_get('text')}</p>
  <p><strong>Адрес:</strong><br>{safe_get('address')}</p>
  <p><strong>Название:</strong><br>{safe_get('name')}</p>
  <p><strong>ID Организации:</strong><br>{safe_get('permalink')}</p>
  <p><strong>Рубрика:</strong><br>{safe_get('normalized_main_rubric_name_ru')}</p>
  <p><strong>Описание:</strong><br>{safe_get('prices_summarized')}</p>
  <p><strong>Истинная релевантность:</strong> {safe_get(label_col)}</p>
  <p><strong>Отзывы:</strong><br>{safe_get('reviews_summarized')}</p>
  <p><strong>Предсказание модели ({pred_col}):</strong><br>{safe_get(pred_col)}</p>
  {agent_log_html}
</div>
"""

def inspect_row(df, idx, *, pred_col="pred_relevance", label_col=RELEVANCE_COL):
    """Отображает HTML-блок для визуального анализа строки DataFrame."""
    row = df.loc[idx]
    html_block = inspect_row_html(row, idx, pred_col=pred_col, label_col=label_col)
    display(HTML(html_block))
