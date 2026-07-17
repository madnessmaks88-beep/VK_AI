"""Проверка видеокарты - сколько VRAM доступно для запуска моделей."""
import torch

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # в ГБ
    print(f"Видеокарта: {device_name}")
    print(f"Всего видеопамяти: {total_memory:.1f} ГБ")

    # Небольшая рекомендация по размеру модели
    if total_memory >= 16:
        print("\nМожешь спокойно запускать модели 14B и даже больше (в квантизации).")
    elif total_memory >= 10:
        print("\nПодходит для моделей 7B-14B (14B возможно потребует квантизацию).")
    elif total_memory >= 6:
        print("\nПодходит для моделей 7B (в квантизации 4-8 бит).")
    else:
        print("\nЛучше остаться на моделях 1.5B-3B, либо использовать квантизацию.")
else:
    print("CUDA недоступна - видеокарта не обнаружена или драйверы не настроены.")
