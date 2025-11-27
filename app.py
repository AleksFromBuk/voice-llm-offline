# app.py
# Русский голосовой транскриптор (полностью офлайн)
#
#  - Vosk: офлайн-распознавание речи
#  - cointegrated/rut5-small: русская LLM для улучшения текста
#  - Tkinter: простой десктопный UI
#
# Основные фичи:
#  * Кнопка "Начать запись" – запись с микрофона, распознавание в фоне
#  * Кнопка "Остановить запись" – аккуратная остановка потока без падений Vosk
#  * Текст сразу появляется внизу в статусе ("Распознано: ..."),
#    затем финальная версия попадает в центральное окно
#  * LLM дополнительно улучшает текст (орфография/стиль) и тоже
#    добавляет результат в центральное окно
#  * Модели загружаются в отдельном потоке; UI не зависает


import json
import logging
import os
import queue
import threading
from typing import Optional, Tuple

import sounddevice as sd
import torch
import tkinter as tk
from tkinter import ttk, messagebox

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from vosk import Model, KaldiRecognizer


# ---------------------- Константы и настройки ----------------------

# Частота дискретизации аудио для Vosk
RATE = 16000

# Размер блока аудио (количество сэмплов за один callback)
BLOCK_SIZE = 8000

# Путь до папки с моделью Vosk.
# ВАЖНО: внутри этой папки должны лежать подпапки am/, conf/, graph/ и т.д.
VOSK_MODEL_PATH = os.path.join("models", "vosk-model-small-ru-0.22")
# Если у тебя модель называется по-другому, скорректируй путь, например:
# VOSK_MODEL_PATH = os.path.join("models", "vosk-model-ru-0.22")

# Имя модели LLM на HuggingFace
HF_MODEL_NAME = "cointegrated/rut5-small"

# Индекс устройства микрофона.
# Если оставить None, приложение выберет первое устройство с входным каналом.
# Если через check_audio.py ты узнал точный индекс, можешь прописать его сюда.
MIC_DEVICE_INDEX: Optional[int] = None


# ---------------------- Основное приложение ----------------------


class SpeechApp(tk.Tk):
    """
    Главное окно приложения.
    Управляет UI, загрузкой моделей и фоновыми потоками записи/LLM.
    """

    def __init__(self) -> None:
        super().__init__()

        self.title("Русский голосовой транскриптор (полностью офлайн)")
        self.geometry("900x500")

        # Очередь сообщений между фоновыми потоками и UI-потоком
        self.gui_queue: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()

        # Событие для остановки записи
        self.stop_event = threading.Event()

        # Флаги состояния
        self.is_recording: bool = False
        self.models_loaded: bool = False

        # Модели
        self.vosk_model: Optional[Model] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.llm_model: Optional[AutoModelForSeq2SeqLM] = None

        # Элементы UI
        self.text_area: tk.Text
        self.status_label: tk.Label
        self.start_button: ttk.Button
        self.stop_button: ttk.Button
        self.clear_button: ttk.Button
        self.progress: ttk.Progressbar

        self._build_ui()

        # Запускаем обработку очереди GUI-событий
        self.after(50, self.process_gui_queue)

        # Стартуем отдельный поток для загрузки моделей
        threading.Thread(target=self.load_models, daemon=True).start()

    # ---------------------- Построение UI ----------------------

    def _build_ui(self) -> None:
        """
        Создание всех виджетов интерфейса.
        """
        # Верхняя "шапка"
        header = tk.Frame(self, bg="#1f3b4d", height=60)
        header.pack(side="top", fill="x")

        header_label = tk.Label(
            header,
            text="Русский голосовой транскриптор (полностью офлайн)",
            fg="white",
            bg="#1f3b4d",
            font=("Segoe UI", 14, "bold"),
            padx=10,
            pady=10,
        )
        header_label.pack(side="left", anchor="w")

        # Подзаголовок
        sub_label = tk.Label(
            self,
            text="• 100% офлайн: Vosk для распознавания речи, русская LLM для улучшения текста\n"
                 "• Нажмите «Начать запись» и говорите в микрофон — текст появится ниже по мере речи",
            justify="left",
            anchor="w",
            padx=10,
            pady=5,
        )
        sub_label.pack(side="top", fill="x")

        # Текстовое поле
        text_frame = tk.Frame(self)
        text_frame.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 5))

        self.text_area = tk.Text(
            text_frame,
            wrap="word",
            font=("Segoe UI", 11),
        )
        self.text_area.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(text_frame, command=self.text_area.yview)
        scroll.pack(side="right", fill="y")
        self.text_area.config(yscrollcommand=scroll.set)

        # Прогресс загрузки моделей
        self.progress = ttk.Progressbar(
            self,
            mode="indeterminate",
            length=300
        )
        self.progress.pack(side="top", pady=(0, 5))
        self.progress.start(10)

        # Нижняя панель с кнопками
        controls = tk.Frame(self)
        controls.pack(side="bottom", fill="x", pady=5, padx=10)

        self.start_button = ttk.Button(
            controls, text="🎤 Начать запись", command=self.start_recording, state="disabled"
        )
        self.start_button.pack(side="right", padx=(0, 10))

        self.stop_button = ttk.Button(
            controls, text="⏹ Остановить запись", command=self.stop_recording, state="disabled"
        )
        self.stop_button.pack(side="right", padx=(0, 10))

        self.clear_button = ttk.Button(
            controls, text="🗑 Очистить", command=self.clear_text, state="normal"
        )
        self.clear_button.pack(side="right", padx=(0, 10))

        # Статус внизу
        self.status_label = tk.Label(
            self,
            text="Загрузка моделей...",
            anchor="w",
            fg="#444",
            padx=10,
            pady=5,
        )
        self.status_label.pack(side="bottom", fill="x")

    # ---------------------- Вспомогательные методы UI ----------------------

    def set_status(self, text: str) -> None:
        """Обновить строку статуса внизу окна."""
        self.status_label.config(text=text)

    def append_text(self, text: str) -> None:
        """Добавить текст в центральное поле."""
        self.text_area.insert("end", text + "\n")
        self.text_area.see("end")

    def clear_text(self) -> None:
        """Очистить поле вывода."""
        self.text_area.delete("1.0", "end")
        self.set_status("Поле очищено.")

    # ---------------------- Загрузка моделей ----------------------

    def load_models(self) -> None:
        """
        Загрузка модели Vosk и LLM в отдельном потоке.
        Все уведомления в UI идут через очередь.
        """
        try:
            logging.info("Загрузка модели Vosk из %s", VOSK_MODEL_PATH)
            if not os.path.isdir(VOSK_MODEL_PATH):
                raise RuntimeError(
                    "Модель Vosk не найдена: %s. "
                    "Скачайте и распакуйте vosk-model-ru-0.22 или vosk-model-small-ru-0.22 "
                    "в папку models/ и скорректируйте VOSK_MODEL_PATH." % VOSK_MODEL_PATH
                )

            self.vosk_model = Model(VOSK_MODEL_PATH)
            logging.info("Модель Vosk успешно загружена.")

            logging.info("Загрузка LLM %s", HF_MODEL_NAME)
            self.tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)
            self.llm_model.eval()
            logging.info("LLM успешно загружена.")

            self.gui_queue.put(("ready", None))

        except Exception as e:
            logging.exception("Ошибка загрузки моделей: %s", e)
            self.gui_queue.put(("error", "Ошибка загрузки моделей: %s" % e))

    # ---------------------- Обработка очереди событий UI ----------------------

    def process_gui_queue(self) -> None:
        """
        Периодически вызывается в UI-потоке и обрабатывает
        сообщения, которые кладут фоновые потоки.
        """
        try:
            while True:
                kind, payload = self.gui_queue.get_nowait()

                if kind == "status":
                    self.set_status(payload or "")

                elif kind == "partial":
                    # Краткий текст распознавания внизу окна
                    self.set_status(f"Распознано: {payload}")

                elif kind == "final":
                    # Финальный текст – в центральное поле
                    self.append_text(payload or "")
                    # Параллельно запускаем улучшение через LLM
                    if payload:
                        threading.Thread(
                            target=self.enhance_and_append,
                            args=(payload,),
                            daemon=True,
                        ).start()

                elif kind == "llm":
                    # Результат после LLM
                    self.append_text("LLM: " + (payload or ""))

                elif kind == "error":
                    msg = payload or "Неизвестная ошибка"
                    self.append_text("[Ошибка] " + msg)
                    self.set_status("Ошибка: " + msg)

                elif kind == "ready":
                    # Модели успешно загружены
                    self.models_loaded = True
                    self.progress.stop()
                    self.progress.pack_forget()
                    self.start_button.config(state="normal")
                    self.set_status("Модели загружены. Нажмите «Начать запись».")

                elif kind == "recording_started":
                    self.start_button.config(state="disabled")
                    self.stop_button.config(state="normal")
                    self.set_status("Запись... Говорите в микрофон.")

                elif kind == "recording_stopped":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    # статус обновится позже final/llm-сообщениями

        except queue.Empty:
            # Очередь опустела — просто переодически дергаем себя дальше
            pass

        self.after(50, self.process_gui_queue)

    # ---------------------- Улучшение текста через LLM ----------------------

    def enhance_and_append(self, text: str) -> None:
        """
        Отдельный поток: берёт текст, прогоняет через LLM
        и добавляет улучшенную версию в центральное поле.
        """
        if self.tokenizer is None or self.llm_model is None:
            return

        try:
            # Вместо длинной русской инструкции даём модели
            # короткий "служебный" префикс, как принято для T5:
            #   "grammar: <исходный текст>"
            prompt = f"grammar: {text}"

            inputs = self.tokenizer(
                [prompt],
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=64,  # можно уменьшить до 32, если захочешь
                    num_beams=4,
                    do_sample=False,
                )

            result = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            if result:
                # Вставляем в поле уже только исправленный текст
                self.gui_queue.put(("llm", result))

        except Exception as e:
            logging.exception("Ошибка работы LLM: %s", e)
            self.gui_queue.put(("error", "Ошибка LLM: %s" % e))

    # ---------------------- Работа с микрофоном ----------------------

    def choose_input_device(self) -> Optional[int]:
        """
        Выбор входного аудиоустройства.
        Если MIC_DEVICE_INDEX задан явно – используем его.
        Иначе берём первое устройство с входными каналами > 0.
        """
        if MIC_DEVICE_INDEX is not None:
            logging.info("Используем явно заданное устройство микрофона: %s", MIC_DEVICE_INDEX)
            return MIC_DEVICE_INDEX

        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    logging.info(
                        "Автовыбран микрофон %s: %s", idx, dev.get("name", "unknown")
                    )
                    return idx
        except Exception as e:
            logging.exception("Не удалось получить список устройств: %s", e)

        return None

    def start_recording(self) -> None:
        """
        Обработчик кнопки «Начать запись».
        """
        if not self.models_loaded or self.vosk_model is None:
            messagebox.showwarning(
                "Модели ещё загружаются",
                "Подождите, пожалуйста, пока завершится загрузка моделей.",
            )
            return

        if self.is_recording:
            return

        self.is_recording = True
        self.stop_event.clear()
        self.gui_queue.put(("recording_started", None))

        threading.Thread(target=self.recording_worker, daemon=True).start()

    def stop_recording(self) -> None:
        """
        Обработчик кнопки «Остановить запись».
        Просто ставим флаг – остальное делает поток записи.
        """
        if self.is_recording:
            self.stop_event.set()

    # ---------------------- Фоновый поток записи и распознавания ----------------------

    def recording_worker(self) -> None:
        """
        Фоновый поток: получает аудио с микрофона, отправляет в Vosk,
        частичные результаты кидает в очередь, финальный – тоже.
        Использует аккуратную остановку через sd.CallbackStop,
        чтобы не провоцировать падения Vosk/Kaldi.
        """
        logging.info("Поток записи запущен")
        recognizer = KaldiRecognizer(self.vosk_model, RATE)
        recognizer.SetWords(True)

        device_index = self.choose_input_device()
        if device_index is None:
            self.gui_queue.put(("error", "Не найдено подходящее устройство ввода (микрофон)."))
            self.is_recording = False
            self.gui_queue.put(("recording_stopped", None))
            return

        try:
            def callback(indata, frames, time_info, status):
                if self.stop_event.is_set():
                    raise sd.CallbackStop()

                if status:
                    logging.warning("Статус аудиопотока: %s", status)

                try:
                    # InputStream отдаёт NumPy-массив -> конвертируем в bytes
                    if recognizer.AcceptWaveform(indata.tobytes()):
                        res = json.loads(recognizer.Result())
                        text = res.get("text", "").strip()
                        if text:
                            self.gui_queue.put(("partial", text))
                except Exception as e:
                    logging.exception("Ошибка внутри callback распознавания: %s", e)
                    self.gui_queue.put(("error", str(e)))
                    raise sd.CallbackStop()

            with sd.InputStream(
                    samplerate=RATE,
                    blocksize=BLOCK_SIZE,
                    dtype="int16",
                    channels=1,
                    callback=callback,
                    device=device_index,
            ):
                while not self.stop_event.is_set():
                    sd.sleep(100)

            # Микрофон уже закрыт – безопасно добираем финальный результат
            try:
                final = json.loads(recognizer.FinalResult())
                final_text = final.get("text", "").strip()
            except Exception as e:
                logging.exception("Ошибка при получении FinalResult: %s", e)
                final_text = ""

            if final_text:
                self.gui_queue.put(("final", final_text))

        except sd.CallbackStop:
            # Нормальное завершение через CallbackStop – можем попробовать всё равно взять финальный результат
            logging.info("Запись остановлена через CallbackStop")
            try:
                final = json.loads(recognizer.FinalResult())
                final_text = final.get("text", "").strip()
                if final_text:
                    self.gui_queue.put(("final", final_text))
            except Exception:
                pass

        except Exception as e:
            logging.exception("Грубая ошибка в recording_worker: %s", e)
            self.gui_queue.put(("error", str(e)))

        finally:
            self.is_recording = False
            self.gui_queue.put(("recording_stopped", None))
            logging.info("Поток записи завершён")


# ---------------------- Точка входа ----------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    app = SpeechApp()
    app.mainloop()


if __name__ == "__main__":
    main()
