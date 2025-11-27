import json
import logging
import os
import queue
import threading
import re
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk, Checkbutton, BooleanVar

from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class ProfessionalVoiceTranscriber:
    """
    Голосовой транскриптор с офлайн-распознаванием Vosk
    и опциональным улучшением текста с помощью русской LLM (RUT5 Normalizer).

    Основные задачи класса:
    - инициализировать и держать UI (Tkinter);
    - в фоне загрузить акустическую модель Vosk и языковую модель T5;
    - управлять потоком записи с микрофона (sounddevice);
    - передавать аудио в распознаватель и публиковать результаты в UI;
    - по необходимости пропускать текст через LLM-нормализатор;
    - сохранять и очищать текст, обрабатывать ошибки.
    """

    def __init__(self, root: tk.Tk) -> None:
        """
        Конструктор основного приложения.

        :param root: уже созданный экземпляр Tk (главное окно).
        """
        # --- Базовая настройка окна ---
        self.root = root
        self.root.title("🎤 Тестовый транскриптор")
        self.root.geometry("950x750")

        # --- Потокобезопасные очереди для общения фоновых потоков с UI ---
        # text_queue: сюда воркер записи и LLM кладут результаты,
        #             а главный поток периодически их забирает и обновляет UI.
        # status_queue: сюда складываются текстовые статусы для статус-строки.
        self.text_queue: "queue.Queue[Tuple[str, str, dict]]" = queue.Queue()
        self.status_queue: "queue.Queue[str]" = queue.Queue()

        # --- Синхронизация потоков ---
        # stop_event: сигнал «остановить запись»
        # recording_lock: защита от повторного запуска/остановки
        self.stop_event = threading.Event()
        self.recording_lock = threading.Lock()

        # models_loaded: успешно ли загрузились Vosk + LLM
        # is_recording: идёт ли сейчас запись
        self.models_loaded = False
        self.is_recording = False

        # --- Настройки аудио-потока ---
        # sample_rate: частота дискретизации (должна совпадать с моделью Vosk)
        # chunk_size: размер блока, который отдаётся в распознаватель
        self.sample_rate = 16000
        self.chunk_size = 4000

        # --- Настройки LLM ---
        # use_llm: привязана к чекбоксу «Включить улучшение текста (LLM)»
        self.use_llm = BooleanVar(value=True)
        # llm_processing: флаг, если понадобится отслеживать, что LLM сейчас занята
        self.llm_processing = False

        # Храним последний «сырой» текст, чтобы при желании его анализировать/подменять
        self.last_raw_text: str = ""

        # Инициализация пользовательского интерфейса
        self._init_ui()

        # Отдельный поток загрузки моделей, чтобы не блокировать UI
        self.models_thread = threading.Thread(target=self._load_models, daemon=True)
        self.models_thread.start()

        # Периодический опрос очередей UI (без блокировки mainloop)
        self.root.after(100, self._process_queues)

    # -------------------------------------------------------------------------
    # UI
    # -------------------------------------------------------------------------

    def _init_ui(self) -> None:
        """
        Создание всех элементов пользовательского интерфейса.

        Структура:
        - верхняя панель с заголовком;
        - панель с настройками (чекбокс LLM + подпись о моделях);
        - прогресс-бар загрузки моделей;
        - большое текстовое поле для результата;
        - нижняя панель: статусная строка + кнопки управления.
        """
        # --- Заголовок окна ---
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = tk.Label(
            title_frame,
            text="🎤Голосовой Транскриптор",
            font=("Arial", 16, "bold"),
            fg="white",
            bg="#2c3e50",
        )
        title_label.pack(pady=10)

        # --- Панель настроек (чекбокс LLM + информация о моделях) ---
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=15, pady=5)

        self.llm_checkbox = Checkbutton(
            settings_frame,
            text="Включить улучшение текста (LLM)",
            variable=self.use_llm,
            font=("Arial", 10),
        )
        self.llm_checkbox.pack(side=tk.LEFT)

        model_info = tk.Label(
            settings_frame,
            text="Vosk (распознавание) + RUT5-Normalizer (улучшение)",
            font=("Arial", 9),
            fg="#666",
        )
        model_info.pack(side=tk.RIGHT)

        # --- Прогресс-бар загрузки моделей ---
        self.progress_frame = tk.Frame(self.root)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=10)

        self.progress_label = tk.Label(
            self.progress_frame,
            text="Загрузка моделей...",
            font=("Arial", 10),
        )
        self.progress_label.pack()

        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()

        # --- Основное поле с текстом транскрипта ---
        text_frame = tk.Frame(self.root)
        text_frame.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            width=90,
            height=22,
            font=("Arial", 11),
            bg="#f8f9fa",
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)
        # по умолчанию блокируем прямой ввод
        self.text_widget.config(state=tk.DISABLED)

        # --- Панель статуса и блок кнопок управления ---
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=15, pady=15)

        self.status_var = tk.StringVar(value="⏳ Загрузка моделей...")
        status_label = tk.Label(
            control_frame,
            textvariable=self.status_var,
            font=("Arial", 10),
            fg="#666666",
        )
        status_label.pack(side=tk.LEFT, anchor=tk.W)

        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)

        # Кнопка «Начать / Остановить запись»
        self.record_btn = tk.Button(
            button_frame,
            text="🎤 Начать запись",
            command=self.toggle_recording,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED,  # пока модели не загрузились
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка очистки
        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Очистить",
            command=self.clear_text,
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            padx=15,
            pady=8,
            state=tk.DISABLED,
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка сохранения
        self.save_btn = tk.Button(
            button_frame,
            text="💾 Сохранить",
            command=self.save_text,
            font=("Arial", 11),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=8,
            state=tk.DISABLED,
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

    # -------------------------------------------------------------------------
    # Загрузка моделей (Vosk + LLM)
    # -------------------------------------------------------------------------

    def _load_models(self) -> None:
        """
        Фоновая загрузка моделей Vosk и RUT5-Normalizer.

        Важные моменты:
        - выполняется в отдельном потоке, чтобы не блокировать UI;
        - путь к модели Vosk жёстко ожидается как `models/vosk-model-small-ru-0.22`;
        - модель T5 скачивается автоматически через Hugging Face;
        - по завершении выставляет `self.models_loaded = True` и
          отправляет статус в очередь для UI.
        """
        try:
            self.status_queue.put("📥 Загружаем модель распознавания речи...")

            # --- Загрузка модели Vosk ---
            model_path = os.path.join("models", "vosk-model-small-ru-0.22")
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Модель Vosk не найдена: {model_path}")

            self.vosk_model = Model(model_path)

            # --- Загрузка специализированной LLM для нормализации ---
            self.status_queue.put("📥 Загружаем улучшенную LLM (RUT5-Normalizer)...")

            model_name = "cointegrated/rut5-small-normalizer"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.llm_model.eval()

            # Выбор устройства (GPU при наличии, иначе CPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.llm_model.to(self.device)

            self.models_loaded = True
            self.status_queue.put("✅ Модели загружены! Готов к работе.")

        except Exception as e:
            logging.error(f"Ошибка загрузки моделей: {e}")
            self.status_queue.put(f"❌ Ошибка загрузки: {str(e)}")

    # -------------------------------------------------------------------------
    # Обработка очередей статусов и текстов (UI-цикл)
    # -------------------------------------------------------------------------

    def _process_queues(self) -> None:
        """
        Периодический опрос очередей `status_queue` и `text_queue`.

        Метод вызывается по таймеру (root.after) каждые ~100 мс и:
        - обновляет статусную строку и прогресс-бар;
        - обрабатывает новые текстовые события (сырое/улучшенное/частичное).
        """
        # --- Обрабатываем статусные сообщения ---
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_var.set(status)

                # Как только модели загружены — убираем прогресс-бар и
                # разблокируем кнопки.
                if status.startswith("✅ Модели загружены"):
                    self.progress.stop()
                    self.progress_frame.pack_forget()
                    self.record_btn.config(state=tk.NORMAL)
                    self.clear_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass

        # --- Обрабатываем текстовые сообщения ---
        try:
            while True:
                text_data = self.text_queue.get_nowait()
                self._process_text_data(text_data)
        except queue.Empty:
            pass

        # Планируем следующий опрос
        self.root.after(100, self._process_queues)

    def _process_text_data(self, text_data: Tuple[str, str, dict]) -> None:
        """
        Обработка одного элемента из `text_queue`.

        :param text_data: кортеж вида (text_type, text, metadata), где:
            - text_type: "raw" | "enhanced" | "partial" | "llm_processing";
            - text: сам текст (может быть пустым для служебных статусов);
            - metadata: дополнительные сведения (например, список изменений).
        """
        text_type, text, metadata = text_data

        if text_type == "raw":
            # Сырой текст от Vosk показываем сразу (серым цветом)
            self._append_text(f"🔹 {text}\n", "raw")
            self.last_raw_text = text

        elif text_type == "enhanced":
            # Улучшенный текст заменяет последнюю «сырую» строку
            changes = metadata.get("changes", [])
            if changes:
                self.status_queue.put(f"✅ Улучшено: {', '.join(changes)}")
            self._replace_last_text(f"✨ {text}\n\n")

        elif text_type == "partial":
            # Частичный результат распознавания выводим в статус
            self.status_var.set(f"🎤 Распознаю: {text}...")

        elif text_type == "llm_processing":
            # Показать пользователю, что идёт обработка LLM
            self.status_var.set("✍️ Улучшаем текст...")

    def _append_text(self, text: str, text_type: str = "normal") -> None:
        """
        Добавление текста в основное поле.

        :param text: вставляемая строка (уже с переводом строки в конце).
        :param text_type: тип оформления:
            - "raw"  — строка помечается серым цветом и иконкой "🔹";
            - "normal" — обычный текст.
        """
        self.text_widget.config(state=tk.NORMAL)

        if text_type == "raw":
            # Добавляем строку и оформляем её отдельным тегом
            self.text_widget.insert(tk.END, text)
            self.text_widget.tag_add("raw", "end-2l", "end-1l")
            self.text_widget.tag_config("raw", foreground="gray")
        else:
            self.text_widget.insert(tk.END, text)

        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def _replace_last_text(self, enhanced_text: str) -> None:
        """
        Заменяет последнюю строку «сырого» текста (`🔹 ...`)
        на улучшенный вариант (строка, пришедшая от LLM).

        Логика:
        - читаем всё содержимое поля;
        - идём с конца и ищем строку, начинающуюся с "🔹";
        - удаляем её и вставляем `enhanced_text` в конец.
        """
        self.text_widget.config(state=tk.NORMAL)

        content = self.text_widget.get("1.0", tk.END)
        lines = content.split("\n")

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("🔹"):
                line_start = f"{i + 1}.0"
                line_end = f"{i + 2}.0"
                self.text_widget.delete(line_start, line_end)
                break

        self.text_widget.insert(tk.END, enhanced_text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    # -------------------------------------------------------------------------
    # Хэвристики и работа с LLM
    # -------------------------------------------------------------------------

    def _needs_llm_correction(self, text: str) -> bool:
        """
        Решение, нужно ли для данной фразы вызывать LLM.

        Идея:
        - короткие фразы (1–2 слова) обычно не трогаем;
        - если уже есть пунктуация и фраза короткая — тоже пропускаем;
        - включаем LLM, если:
          * фраза достаточно длинная и без знаков препинания;
          * есть цифры — модель может нормализовать числа;
          * присутствуют потенциально «слепленные» слова / союзы;
          * в типичных конструкциях не хватает предлогов.
        """
        words = text.split()

        if len(words) < 3:
            return False

        if any(punct in text for punct in ".!?") and len(words) < 6:
            return False

        needs_correction = (
            # Длинные фразы без пунктуации
            (len(words) >= 4 and not any(punct in text for punct in ".!?,:"))
            or
            # Есть числа (например, "25") — их можно нормализовать
            any(word.isdigit() for word in words)
            or
            # Потенциальные случаи слитного написания/омофонов
            any(
                pattern in text.lower()
                for pattern in ["какдела", "чтоты", "чтобы", "зачемты", "потомучто"]
            )
            or
            # Проверка отсутствующих предлогов в типичных конструкциях
            self._missing_prepositions(text)
        )

        return needs_correction

    def _missing_prepositions(self, text: str) -> bool:
        """
        Простая эвристика для обнаружения пропущенных предлогов.

        Пример: «пошел магазин» → ожидается «пошел В магазин».
        Возвращает True, если конструкция похожа на глагол + существительное
        без предлога между ними.
        """
        words = text.split()
        common_verbs = ["пошел", "пришел", "ушел", "вернулся", "зашел"]
        following_nouns = ["магазин", "дом", "работа", "улица", "парк"]

        for i, word in enumerate(words[:-1]):
            if word in common_verbs and words[i + 1] in following_nouns:
                return True
        return False

    def _enhance_with_llm(self, text: str) -> Tuple[str, list]:
        """
        Отправляет текст в LLM-нормализатор и возвращает улучшенную версию.

        :param text: исходная фраза от Vosk.
        :return: кортеж (result_text, changes), где:
                 - result_text: либо улучшенный текст, либо исходный,
                   если модель не смогла дать полезный результат;
                 - changes: список кратких описаний изменений
                   (пунктуация, структура, предлоги и т.п.).
        """
        if not text.strip():
            return text, []

        try:
            # Для RUT5-Normalizer не нужен специальный префикс —
            # подаём текст "как есть".
            prompt = text

            inputs = self.llm_tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=150,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.1,  # консервативная генерация
                    no_repeat_ngram_size=2,
                )

            result = self.llm_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()

            changes = self._analyze_changes(text, result)

            # Если улучшение выглядит сомнительным — оставляем исходный текст
            if self._is_improvement_worthwhile(text, result, changes):
                return result, changes
            else:
                return text, []

        except Exception as e:
            logging.error(f"Ошибка LLM улучшения: {e}")
            return text, []

    def _analyze_changes(self, original: str, enhanced: str) -> list:
        """
        Анализирует, какие типы изменений внесла LLM в текст.

        Сейчас это простая эвристика:
        - изменение длины/структуры предложения;
        - добавление новых знаков препинания;
        - увеличение количества предлогов.
        """
        if original == enhanced:
            return []

        changes = []
        orig_words = original.split()
        enh_words = enhanced.split()

        if len(orig_words) != len(enh_words):
            changes.append("структура предложения")

        # Пунктуация
        orig_punct = set(re.findall(r"[.,!?;:]", original))
        enh_punct = set(re.findall(r"[.,!?;:]", enhanced))
        new_punct = enh_punct - orig_punct
        if new_punct:
            changes.append("пунктуация")

        # Предлоги
        prepositions = ["в", "на", "за", "под", "о", "у", "с", "по"]
        orig_prep = sum(1 for word in orig_words if word in prepositions)
        enh_prep = sum(1 for word in enh_words if word in prepositions)
        if enh_prep > orig_prep:
            changes.append("предлоги")

        return changes

    def _is_improvement_worthwhile(
        self, original: str, enhanced: str, changes: list
    ) -> bool:
        """
        Решает, стоит ли принимать улучшение от LLM.

        Критерии:
        - что-то реально изменилось (`changes` не пустой);
        - длина фразы не изменилась более чем на 50%;
        - хотя бы 60% исходных слов сохранены (иначе модель «переписала» фразу).
        """
        if not changes:
            return False

        len_diff = abs(len(enhanced) - len(original)) / max(len(original), 1)
        if len_diff > 0.5:
            return False

        orig_words = set(original.lower().split())
        enh_words = set(enhanced.lower().split())
        common_words = orig_words.intersection(enh_words)

        if len(common_words) / max(len(orig_words), 1) < 0.6:
            return False

        return True

    # -------------------------------------------------------------------------
    # Управление записью (кнопки, воркер, callback sounddevice)
    # -------------------------------------------------------------------------

    def toggle_recording(self) -> None:
        """
        Обработчик кнопки «Начать / Остановить запись».

        В зависимости от текущего состояния переключает
        между `start_recording()` и `stop_recording()`.
        """
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self) -> None:
        """
        Запуск записи с микрофона.

        Проверяет, что модели загружены, и защищается от повторного запуска
        через `recording_lock`. Создаёт фоновый поток `_recording_worker`.
        """
        if not self.models_loaded:
            messagebox.showerror("Ошибка", "Модели еще не загружены!")
            return

        with self.recording_lock:
            if self.is_recording:
                return
            self.is_recording = True
            self.stop_event.clear()

        self.record_btn.config(text="⏹️ Остановить запись", bg="#c0392b")
        self.status_var.set("🎤 Запись... Говорите!")

        self.worker_thread = threading.Thread(
            target=self._recording_worker,
            daemon=True,
        )
        self.worker_thread.start()

    def _recording_worker(self) -> None:
        """
        Основной рабочий цикл записи и распознавания.

        Внутри:
        - создаётся `KaldiRecognizer`;
        - запускается `sounddevice.InputStream` с callbackом;
        - callback отправляет аудио-фреймы в Vosk, публикует результаты в очередь;
        - при остановке инициирует финализацию и обновление UI.
        """
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

        def audio_callback(indata, frames, time, status):
            """
            Callback для `sounddevice`.

            Вызывается на каждом блоке аудио:
            - проверяет флаг остановки;
            - конвертирует float32 → int16;
            - отдаёт блок в Vosk и обрабатывает полный/частичный результат;
            - при ошибках заканчивает стрим через `sd.CallbackStop`.
            """
            if self.stop_event.is_set():
                raise sd.CallbackStop()

            if status:
                logging.warning(f"Аудио статус: {status}")

            try:
                # Преобразование аудио в формат, ожидаемый Vosk
                pcm_data = (indata * 32767).astype(np.int16).tobytes()

                if recognizer.AcceptWaveform(pcm_data):
                    # Полный результат
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        # Показываем «сырой» текст
                        self.text_queue.put(("raw", text, {}))

                        # Решаем, нужно ли улучшение через LLM
                        if self.use_llm.get() and self._needs_llm_correction(text):
                            self.text_queue.put(("llm_processing", "", {}))
                            threading.Thread(
                                target=self._process_with_llm,
                                args=(text,),
                                daemon=True,
                            ).start()
                        else:
                            # Просто дублируем текст как «улучшенный»
                            self.text_queue.put(
                                ("enhanced", text, {"changes": []})
                            )

                else:
                    # Частичный результат распознавания
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        self.text_queue.put(("partial", partial_text, {}))

            except Exception as e:
                logging.error(f"Ошибка в callback: {e}")
                self.stop_event.set()

        try:
            # Запуск аудио-потока
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=audio_callback,
                latency="low",
            ):
                while not self.stop_event.is_set():
                    sd.sleep(100)

        except sd.CallbackStop:
            logging.info("Запись остановлена пользователем")
        except Exception as e:
            logging.error(f"Ошибка записи: {e}")
            self.status_queue.put(f"❌ Ошибка записи: {str(e)}")
        finally:
            self._finalize_recording(recognizer)

    def _process_with_llm(self, text: str) -> None:
        """
        Фоновая LLM-обработка одной фразы.

        Вызывается из callback-а в отдельном потоке:
        - прогоняет текст через `_enhance_with_llm`;
        - кладёт результат обратно в очередь `text_queue`.
        """
        try:
            enhanced, changes = self._enhance_with_llm(text)
            self.text_queue.put(("enhanced", enhanced, {"changes": changes}))
        except Exception as e:
            logging.error(f"Ошибка LLM обработки: {e}")
            self.text_queue.put(("enhanced", text, {"changes": []}))

    def _finalize_recording(self, recognizer: KaldiRecognizer) -> None:
        """
        Корректное завершение записи.

        - запрашивает `FinalResult` у Vosk, чтобы не потерять хвост;
        - сбрасывает флаг `is_recording`;
        - передаёт управление в `_recording_stopped_ui`.
        """
        try:
            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                self.text_queue.put(
                    ("enhanced", final_text, {"changes": []})
                )
        except Exception as e:
            logging.warning(f"Ошибка финализации: {e}")
        finally:
            with self.recording_lock:
                self.is_recording = False
            self.root.after(0, self._recording_stopped_ui)

    def _recording_stopped_ui(self) -> None:
        """
        Обновляет UI после остановки записи:
        - возвращает кнопку в состояние «Начать запись»;
        - обновляет статусную строку.
        """
        self.record_btn.config(
            text="🎤 Начать запись",
            bg="#27ae60",
            state=tk.NORMAL,
        )
        self.status_var.set("✅ Запись завершена")

    def stop_recording(self) -> None:
        """
        Безопасная остановка записи (обработчик кнопки).

        Ставит флаг `stop_event` и временно дизейблит кнопку,
        реальное завершение происходит в `_recording_worker`.
        """
        if not self.is_recording:
            return

        self.stop_event.set()
        self.record_btn.config(state=tk.DISABLED)
        self.status_var.set("🔄 Безопасная остановка...")

    # -------------------------------------------------------------------------
    # Служебные действия: очистка и сохранение текста
    # -------------------------------------------------------------------------

    def clear_text(self) -> None:
        """
        Очищает текстовое поле и сбрасывает последний «сырой» текст.
        """
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state=tk.DISABLED)
        self.last_raw_text = ""
        self.status_var.set("📝 Текст очищен")

    def save_text(self) -> None:
        """
        Сохраняет текущий текст транскрипта в файл `.txt`.

        - открывает диалог выбора имени файла;
        - при успешном сохранении пишет статус с именем файла;
        - при ошибках показывает messagebox с описанием.
        """
        try:
            text_content = self.text_widget.get("1.0", tk.END).strip()
            if not text_content:
                messagebox.showwarning(
                    "Предупреждение", "Нет текста для сохранения"
                )
                return

            from tkinter import filedialog

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[
                    ("Текстовые файлы", "*.txt"),
                    ("Все файлы", "*.*"),
                ],
            )

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_content)
                self.status_var.set(
                    f"✅ Текст сохранен в {os.path.basename(file_path)}"
                )

        except Exception as e:
            messagebox.showerror(
                "Ошибка", f"Не удалось сохранить файл: {str(e)}"
            )


def main() -> None:
    """
    Точка входа приложения.

    Настраивает логирование, создаёт окно Tkinter и запускает главный цикл.
    В случае фатальной ошибки показывает диалог и пишет в лог.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("transcriber.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    try:
        root = tk.Tk()
        app = ProfessionalVoiceTranscriber(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        messagebox.showerror(
            "Ошибка", f"Не удалось запустить приложение:\n{str(e)}"
        )


if __name__ == "__main__":
    main()
