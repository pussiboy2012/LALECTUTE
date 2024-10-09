import openai
import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')  # Замените это на ваш метод получения ключа

def generate_markdown_concept(text):
    """
    Преобразует текст лекции в структурированный Markdown-конспект с помощью GPT-модели.

    :param text: Строка с текстом лекции.
    :return: Строка с Markdown-конспектом.
    """
    prompt = f"""
    Преобразуй следующий текст лекции в структурированный конспект в формате Markdown. 
    Раздели текст на разделы и подразделы с соответствующими заголовками, 
    выдели ключевые моменты списками и добавь форматирование для улучшения читаемости.

    Текст лекции:
    {text}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты помощник, который преобразует текст лекций в структурированные конспекты."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.5,
        )

        # Проверяем, что ответ содержит 'choices'
        if 'choices' in response and len(response['choices']) > 0:
            markdown_text = response['choices'][0]['message']['content']
            return markdown_text
        else:
            print("Ошибка: нет доступных выборов в ответе.")
            return None
    except Exception as e:
        print(f"Ошибка при вызове OpenAI API: {e}")
        return None
