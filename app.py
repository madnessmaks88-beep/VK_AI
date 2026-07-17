"""
FastAPI-сервис для Math Tutor - генерация через Groq API.

Больше НЕ загружает никакую модель локально - не нужен GPU, не нужно
скачивать веса, не нужно ждать загрузки при старте. Вся генерация
происходит на серверах Groq, наш сервис просто пересылает вопрос
и возвращает ответ.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from groq import Groq
from openai import OpenAI  # используем для DeepSeek - его API совместим с форматом OpenAI

# Ключ читаем из переменной окружения - НЕ храним в коде напрямую,
# чтобы случайно не закоммитить его в git
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"  # модель по умолчанию ("smart")

# Доступные текстовые модели для команды /model - НАЗВАНИЯ ПРОВЕРЬ на
# console.groq.com/docs/models и api-docs.deepseek.com, списки меняются.
# Формат значения: (провайдер, имя_модели) - провайдер определяет, какой
# клиент и какой base_url использовать при вызове.
AVAILABLE_MODELS = {
    "fast": ("groq", "llama-3.1-8b-instant"),        # маленькая и очень быстрая
    "smart": ("groq", "llama-3.3-70b-versatile"),    # крупная, качественнее
    "deepseek": ("deepseek", "deepseek-v4-flash"),   # DeepSeek - быстрый вариант V4
    "deepseek-pro": ("deepseek", "deepseek-v4-pro"), # DeepSeek - тяжёлый вариант для сложных рассуждений
}
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # мультимодальная модель Groq, умеет читать изображения

app = FastAPI(title="Math Tutor API", description="API для решения задач по высшей математике")

if not GROQ_API_KEY:
    print("ВНИМАНИЕ: переменная окружения GROQ_API_KEY не установлена!")
    print("Установи её перед запуском: set GROQ_API_KEY=твой_ключ")

client = Groq(api_key=GROQ_API_KEY)

# DeepSeek API совместим по формату с OpenAI - используем тот же SDK,
# просто указываем другой base_url и другой ключ
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

if not DEEPSEEK_API_KEY:
    print("ВНИМАНИЕ: переменная окружения DEEPSEEK_API_KEY не установлена - модели DeepSeek работать не будут!")


class Question(BaseModel):
    text: str
    max_tokens: int = 300
    history: list[dict] = []
    image_url: str | None = None
    model: str = "smart"  # ключ из AVAILABLE_MODELS: "fast" / "smart" / "deepseek" / "deepseek-pro"
    thinking: bool = True  # только для DeepSeek - включён ли режим размышления (thinking mode)


class Answer(BaseModel):
    question: str
    answer: str


@app.get("/")
def root():
    """Отдаёт HTML-интерфейс чата."""
    return FileResponse("static/index.html")


@app.get("/api/status")
def status():
    return {"status": "ok", "message": "Math Tutor API работает (через Groq). Отправь POST на /ask"}


@app.post("/ask", response_model=Answer)
def ask(question: Question):
    """
    Основной эндпоинт: пересылает вопрос в Groq, возвращает ответ.

    В отличие от локальной модели, здесь нет apply_chat_template,
    токенизации вручную и т.д. - Groq (как и любой чат-API) сам
    принимает готовый формат messages и сам всё форматирует внутри.
    """
    # Собираем полный список сообщений: системная инструкция + вся история
    # диалога (если есть) + новый вопрос. Именно так модель "помнит"
    # предыдущие сообщения - на самом деле она не имеет постоянной памяти,
    # просто мы каждый раз заново присылаем ей всю историю целиком.
    messages = [
        {"role": "system", "content": "Ты - полезный ассистент. Отвечай ясно и по существу."},
    ]
    messages.extend(question.history)

    if question.image_url:
        print(f"[DEBUG] Получена картинка, URL: {question.image_url}")
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        (question.text or "Опиши эту картинку.")
                        + " Дай развёрнутый ответ (минимум 2-3 предложения), "
                        "опиши конкретные детали, цвета, форму, содержимое - "
                        "не ограничивайся одним словом."
                    ),
                },
                {"type": "image_url", "image_url": {"url": question.image_url}},
            ],
        })
        provider, model_to_use = "groq", GROQ_VISION_MODEL  # картинка - всегда через vision-модель Groq
    else:
        messages.append({"role": "user", "content": question.text})
        provider, model_to_use = AVAILABLE_MODELS.get(question.model, ("groq", GROQ_MODEL))

    print(f"[DEBUG] Провайдер: {provider}, модель: {model_to_use}")

    # Выбираем нужный клиент в зависимости от провайдера - у DeepSeek
    # свой ключ и свой сервер, хотя формат запроса (messages, max_tokens
    # и т.д.) одинаковый благодаря OpenAI-совместимому API у обоих.
    active_client = deepseek_client if provider == "deepseek" else client

    # extra_body - способ передать параметры, специфичные для конкретного
    # провайдера. У DeepSeek через него управляется thinking mode.
    # ВАЖНО: SDK библиотеки groq и openai - это РАЗНЫЕ пакеты, и groq
    # может не поддерживать параметр extra_body вообще (кинет ошибку
    # "unexpected keyword argument"). Поэтому собираем аргументы вызова
    # динамически и добавляем extra_body, только когда реально обращаемся
    # к DeepSeek - для Groq вызов остаётся ровно таким же, как раньше.
    call_kwargs = {
        "model": model_to_use,
        "messages": messages,
        "max_tokens": question.max_tokens,
        "temperature": 0.3,
    }
    if provider == "deepseek":
        call_kwargs["extra_body"] = {
            "thinking": {"type": "enabled" if question.thinking else "disabled"}
        }

    try:
        response = active_client.chat.completions.create(**call_kwargs)
    except Exception as e:
        # И groq, и openai SDK кидают исключения со status_code при ошибках
        # API (в т.ч. 429 - превышение лимита запросов). Проверяем это поле
        # и пробрасываем правильный HTTP-статус дальше клиенту (боту),
        # чтобы он мог отличить "лимит превышен" от прочих ошибок.
        status_code = getattr(e, "status_code", None)
        if status_code == 429:
            print(f"[DEBUG] Получена ошибка 429 от {provider}")
            raise HTTPException(status_code=429, detail="Превышен лимит запросов к модели")
        print(f"[DEBUG] Ошибка при обращении к {provider}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    answer_text = response.choices[0].message.content
    print(f"[DEBUG] Сырой ответ от {provider}: {answer_text!r}")

    return Answer(question=question.text, answer=answer_text)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
