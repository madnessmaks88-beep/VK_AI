"""
Хранение истории диалогов в SQLite вместо оперативной памяти.

Почему SQLite, а не PostgreSQL/MySQL: это просто ФАЙЛ на диске, не нужен
отдельный сервер базы данных - идеально для одного бота с умеренной
нагрузкой. Если бот вырастет до тысяч одновременных пользователей,
тогда есть смысл перейти на "взрослую" СУБД.
"""
import sqlite3
import time

DB_PATH = "bot_history.db"


def init_db():
    """Создаёт таблицы, если их ещё нет. Вызывается один раз при старте бота."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    # Индекс по user_id - ускоряет выборку истории конкретного пользователя,
    # без него SQLite пришлось бы сканировать всю таблицу на каждый запрос
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON messages(user_id)")
    conn.commit()
    conn.close()


def get_history(user_id: int, max_messages: int = 20) -> list[dict]:
    """Возвращает последние max_messages сообщений пользователя в формате для модели."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, max_messages),
    ).fetchall()
    conn.close()
    # Разворачиваем - из базы пришли в обратном порядке (новые первыми)
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def add_message(user_id: int, role: str, content: str):
    """Сохраняет одно сообщение (вопрос пользователя или ответ модели)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (user_id, role, content, time.time()),
    )
    conn.commit()
    conn.close()


def clear_history(user_id: int):
    """Удаляет всю историю конкретного пользователя (команда /clear)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()
