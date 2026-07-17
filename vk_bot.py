"""
VK-бот: SQLite-память, rate limiting, индикатор печати, кнопки, фото.

Изменения по сравнению с предыдущей версией:
- История хранится в SQLite (db.py) - переживает перезапуски бота
- Rate limiting - защита от злоупотребления одним пользователем
- messages.setActivity - показывает "печатает..." пока модель генерирует ответ
- Клавиатура с кнопками - не нужно помнить текстовые команды
- Поддержка фото - можно прислать картинку, бот прочитает и ответит
"""
import re
import time
import requests
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor

import db

# --- Настройки ---
VK_TOKEN = "ВСТАВЬ_СВОЙ_ТОКЕН_СООБЩЕСТВА_СЮДА"
MODEL_API_URL = "http://localhost:8000/ask"
MAX_HISTORY_MESSAGES = 20

# --- Rate limiting: максимум сообщений за окно времени на пользователя ---
RATE_LIMIT_MESSAGES = 10   # не больше 10 сообщений
RATE_LIMIT_WINDOW = 60     # за 60 секунд
# {user_id: [timestamp1, timestamp2, ...]} - список меток времени недавних сообщений
rate_limit_tracker: dict[int, list[float]] = {}

# {user_id: "fast" или "smart"} - выбранная пользователем модель.
# Хранится только в памяти - при перезапуске бота сбросится на "smart" (по умолчанию).
user_model_choice: dict[int, str] = {}

# {user_id: bool} - включён ли thinking mode для DeepSeek у конкретного
# пользователя. По умолчанию True, т.к. это дефолт самого DeepSeek.
user_thinking_choice: dict[int, bool] = {}

COMMANDS = {
    "/help": "Показать список команд",
    "/clear": "Очистить историю переписки",
    "/start": "Начать сначала / приветствие",
    "/model fast": "Переключиться на быструю модель (проще, но мгновенные ответы)",
    "/model smart": "Переключиться на мощную модель (умнее, но чуть медленнее)",
    "/model deepseek": "Переключиться на DeepSeek V4 Flash",
    "/model deepseek-pro": "Переключиться на DeepSeek V4 Pro (для сложных рассуждений)",
    "/thinking on": "Включить режим размышления DeepSeek (качественнее, но медленнее)",
    "/thinking off": "Выключить режим размышления DeepSeek (быстрее)",
}


def is_rate_limited(user_id: int) -> bool:
    """
    Проверяет, не превысил ли пользователь лимит сообщений.
    Использует технику "скользящего окна" - храним метки времени последних
    сообщений, отбрасываем те, что старше RATE_LIMIT_WINDOW секунд, и
    смотрим, сколько осталось актуальных.
    """
    now = time.time()
    timestamps = rate_limit_tracker.get(user_id, [])
    # Оставляем только метки за последнее окно времени
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]

    if len(timestamps) >= RATE_LIMIT_MESSAGES:
        rate_limit_tracker[user_id] = timestamps  # сохраняем очищенный список
        return True

    timestamps.append(now)
    rate_limit_tracker[user_id] = timestamps
    return False


def strip_markdown(text: str) -> str:
    """ВК не поддерживает Markdown - убираем разметку перед отправкой."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    return text.strip()


def ask_model(question: str, history: list[dict], image_url: str | None = None, model: str = "smart", thinking: bool = True) -> str:
    """Отправляем вопрос (и, если есть, картинку) в FastAPI-сервис."""
    payload = {"text": question, "max_tokens": 1000, "history": history, "model": model, "thinking": thinking}
    if image_url:
        payload["image_url"] = image_url

    # Пробуем до 2 раз при ошибке 429 (превышение лимита запросов) -
    # часто такая ошибка кратковременная, и повтор через пару секунд
    # решает проблему без участия пользователя.
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(MODEL_API_URL, json=payload, timeout=60)

            if response.status_code == 429:
                if attempt < max_attempts:
                    print(f"[INFO] Получена ошибка 429 (лимит запросов), попытка {attempt}/{max_attempts}, жду 3 секунды...")
                    time.sleep(3)
                    continue
                return (
                    "Сервис сейчас перегружен (превышен лимит запросов к модели). "
                    "Попробуй задать вопрос ещё раз через минуту."
                )

            response.raise_for_status()
            return response.json()["answer"]

        except requests.exceptions.ConnectionError:
            return "Не могу связаться с сервисом модели. Убедись, что запущен app.py."
        except requests.exceptions.Timeout:
            return "Модель слишком долго думает, попробуй ещё раз."
        except Exception as e:
            return f"Произошла ошибка: {e}"


def build_keyboard() -> str:
    """
    Строит клавиатуру с кнопками-командами - показывается под сообщениями
    бота, пользователь может нажать вместо того чтобы печатать команду руками.
    """
    keyboard = VkKeyboard(one_time=False)
    keyboard.add_button("Очистить историю", color=VkKeyboardColor.NEGATIVE)
    keyboard.add_line()
    keyboard.add_button("Помощь", color=VkKeyboardColor.PRIMARY)
    return keyboard.get_keyboard()


def get_largest_photo_url(attachment: dict) -> str | None:
    """
    Фото во ВКонтакте приходит в нескольких размерах одновременно
    (для разных экранов) - выбираем самый крупный вариант, чтобы
    модели было проще разобрать детали на картинке.
    """
    sizes = attachment.get("photo", {}).get("sizes", [])
    if not sizes:
        return None
    largest = max(sizes, key=lambda s: s.get("width", 0))
    return largest.get("url")


def extract_photo_url(vk, event) -> str | None:
    """Достаёт URL фото из сообщения, если оно там есть."""
    full_message = vk.messages.getById(message_ids=event.message_id)["items"][0]
    for attachment in full_message.get("attachments", []):
        if attachment.get("type") == "photo":
            return get_largest_photo_url(attachment)
    return None


def handle_command(command: str, user_id: int) -> str:
    command = command.lower().strip()

    if command in ("/model fast", "/model smart", "/model deepseek", "/model deepseek-pro"):
        chosen = command.split()[1]  # "fast" / "smart" / "deepseek" / "deepseek-pro"
        user_model_choice[user_id] = chosen
        model_names = {
            "fast": "быструю (Llama 3.1 8B)",
            "smart": "мощную (Llama 3.3 70B)",
            "deepseek": "DeepSeek V4 Flash",
            "deepseek-pro": "DeepSeek V4 Pro",
        }
        return f"Переключился на {model_names[chosen]} модель."

    if command in ("/thinking on", "/thinking off"):
        enabled = command.endswith("on")
        user_thinking_choice[user_id] = enabled
        status = "включён" if enabled else "выключен"
        return f"Режим размышления DeepSeek {status}. (Работает только для моделей DeepSeek - на Groq не влияет.)"

    if command in ("/clear", "очистить историю"):
        db.clear_history(user_id)
        return "История переписки очищена. Начинаем с чистого листа."

    if command == "/start":
        db.clear_history(user_id)
        return (
            "Привет! Я ассистент, готовый помочь с любыми вопросами - "
            "текстом или фото. Помню контекст переписки, пока не отправишь "
            "команду очистки.\n\nИспользуй кнопки внизу или команду /help."
        )

    if command in ("/help", "помощь"):
        lines = ["Доступные команды:"]
        for cmd, description in COMMANDS.items():
            lines.append(f"{cmd} - {description}")
        lines.append("\nМожно присылать и фото - опишу или отвечу по содержимому.")
        return "\n".join(lines)

    return f"Неизвестная команда: {command}\nНапиши /help для списка команд."


def main():
    db.init_db()  # создаёт таблицы при первом запуске, при повторных - ничего не делает

    vk_session = vk_api.VkApi(token=VK_TOKEN)
    vk = vk_session.get_api()
    longpoll = VkLongPoll(vk_session)
    keyboard = build_keyboard()

    print("Бот запущен, слушаю сообщения...")
    print(f"Модель ожидается на: {MODEL_API_URL}")

    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            user_id = event.user_id
            text = event.text.strip()

            # --- Rate limiting: проверяем ДО любой обработки ---
            if is_rate_limited(user_id):
                vk.messages.send(
                    user_id=user_id,
                    message=f"Слишком много сообщений подряд. Подожди немного (лимит: {RATE_LIMIT_MESSAGES} сообщений в {RATE_LIMIT_WINDOW} секунд).",
                    random_id=0,
                )
                continue

            # --- Индикатор "печатает..." - показываем сразу, до генерации ---
            vk.messages.setActivity(user_id=user_id, type="typing")

            # --- Проверяем, есть ли фото в сообщении ---
            photo_url = None
            if event.attachments:
                photo_url = extract_photo_url(vk, event)

            if not text and not photo_url:
                continue

            print(f"[Сообщение от {user_id}]: {text or '[фото без текста]'}")

            # --- Команды (только текстовые, без фото) ---
            if text.startswith("/") or text.lower() in ("очистить историю", "помощь"):
                answer = handle_command(text, user_id)
            else:
                history = db.get_history(user_id, max_messages=MAX_HISTORY_MESSAGES)
                chosen_model = user_model_choice.get(user_id, "smart")
                chosen_thinking = user_thinking_choice.get(user_id, True)
                answer = ask_model(text, history, image_url=photo_url, model=chosen_model, thinking=chosen_thinking)
                answer = strip_markdown(answer)

                # Сохраняем в SQLite - переживёт перезапуск бота
                db.add_message(user_id, "user", text or "[отправил фото]")
                db.add_message(user_id, "assistant", answer)

            print(f"[Ответ]: {answer[:100]}...")

            vk.messages.send(
                user_id=user_id,
                message=answer,
                random_id=0,
                keyboard=keyboard,
            )


if __name__ == '__main__':
    main()
