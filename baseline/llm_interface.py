## Обёртка над OpenAI, простой вызов GPT 
import os
from openai import OpenAI
"""
    Интерфейс для взаимодействия с моделью GPT через API (по умолчанию — https://api.vsegpt.ru/v1).

    Атрибуты:
        api_key (str): Ключ API OpenAI. Может быть передан напрямую или считан из переменной окружения OPENAI_API_KEY.
        model_name (str): Название модели, используемой для генерации (по умолчанию "gpt-4o-mini").
        client (OpenAI): Клиент OpenAI для отправки запросов к модели.

    Методы:
        call_gpt(prompt): Отправляет запрос к модели с заданным промтом и возвращает сгенерированный ответ.
    """

class GPTInterface:
    def __init__(self, api_key=None, model_name="gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.vsegpt.ru/v1")

    def call_gpt(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Ты классификатор релевантности."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=5,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print("Ошибка запроса:", e)
            return "ERROR"
