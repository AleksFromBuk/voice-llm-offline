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
    Профессиональный голосовой транскриптор с реально полезной LLM интеграцией
    """

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("🎤 Профессиональный Транскриптор")
        self.root.geometry("950x750")

        # Очереди для межпоточного общения
        self.text_queue = queue.Queue()
        self.status_queue = queue.Queue()

        # Синхронизация
        self.stop_event = threading.Event()
        self.recording_lock = threading.Lock()
        self.models_loaded = False
        self.is_recording = False

        # Настройки
        self.sample_rate = 16000
        self.chunk_size = 4000

        # LLM настройки
        self.use_llm = BooleanVar(value=True)  # Галочка включения LLM
        self.llm_processing = False

        # История для замены текста
        self.last_raw_text = ""

        self._init_ui()
        self.models_thread = threading.Thread(target=self._load_models, daemon=True)
        self.models_thread.start()
        self.root.after(100, self._process_queues)

    def _init_ui(self):
        """Профессиональный интерфейс с настройками"""
        # Заголовок
        title_frame = tk.Frame(self.root, bg="#2c3e50")
        title_frame.pack(fill=tk.X, padx=10, pady=10)

        title_label = tk.Label(
            title_frame,
            text="🎤 Профессиональный Голосовой Транскриптор",
            font=("Arial", 16, "bold"),
            fg="white", bg="#2c3e50"
        )
        title_label.pack(pady=10)

        # Панель настроек
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill=tk.X, padx=15, pady=5)

        # Галочка включения LLM
        self.llm_checkbox = Checkbutton(
            settings_frame,
            text="Включить улучшение текста (LLM)",
            variable=self.use_llm,
            font=("Arial", 10)
        )
        self.llm_checkbox.pack(side=tk.LEFT)

        # Информация о моделях
        model_info = tk.Label(
            settings_frame,
            text="Vosk (распознавание) + RUT5-Normalizer (улучшение)",
            font=("Arial", 9),
            fg="#666"
        )
        model_info.pack(side=tk.RIGHT)

        # Прогресс-бар загрузки
        self.progress_frame = tk.Frame(self.root)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=10)

        self.progress_label = tk.Label(
            self.progress_frame,
            text="Загрузка моделей...",
            font=("Arial", 10)
        )
        self.progress_label.pack()

        self.progress = ttk.Progressbar(self.progress_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=5)
        self.progress.start()

        # Основное текстовое поле
        text_frame = tk.Frame(self.root)
        text_frame.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        self.text_widget = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            width=90,
            height=22,
            font=("Arial", 11),
            bg="#f8f9fa"
        )
        self.text_widget.pack(fill=tk.BOTH, expand=True)

        # Панель статуса и управления
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=15, pady=15)

        self.status_var = tk.StringVar(value="⏳ Загрузка моделей...")
        status_label = tk.Label(
            control_frame,
            textvariable=self.status_var,
            font=("Arial", 10),
            fg="#666666"
        )
        status_label.pack(side=tk.LEFT, anchor=tk.W)

        # Кнопки управления
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.RIGHT)

        self.record_btn = tk.Button(
            button_frame,
            text="🎤 Начать запись",
            command=self.toggle_recording,
            font=("Arial", 12, "bold"),
            bg="#27ae60",
            fg="white",
            padx=20,
            pady=10,
            state=tk.DISABLED
        )
        self.record_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(
            button_frame,
            text="🗑️ Очистить",
            command=self.clear_text,
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            padx=15,
            pady=8,
            state=tk.DISABLED
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(
            button_frame,
            text="💾 Сохранить",
            command=self.save_text,
            font=("Arial", 11),
            bg="#3498db",
            fg="white",
            padx=15,
            pady=8,
            state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=5)

    def _load_models(self):
        """Загрузка УЛУЧШЕННЫХ моделей"""
        try:
            self.status_queue.put("📥 Загружаем модель распознавания речи...")

            # Vosk модель
            model_path = os.path.join("models", "vosk-model-small-ru-0.22")
            if not os.path.isdir(model_path):
                raise FileNotFoundError(f"Модель Vosk не найдена: {model_path}")

            self.vosk_model = Model(model_path)

            # ⚡ УЛУЧШЕНИЕ: Специализированная модель для нормализации
            self.status_queue.put("📥 Загружаем улучшенную LLM (RUT5-Normalizer)...")

            # Используем специализированную модель для нормализации текста
            model_name = "cointegrated/rut5-small-normalizer"
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.llm_model.eval()

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.llm_model.to(self.device)

            self.models_loaded = True
            self.status_queue.put("✅ Модели загружены! Готов к работе.")

        except Exception as e:
            logging.error(f"Ошибка загрузки моделей: {e}")
            self.status_queue.put(f"❌ Ошибка загрузки: {str(e)}")

    def _process_queues(self):
        """Обработка очередей"""
        # Статусы
        try:
            while True:
                status = self.status_queue.get_nowait()
                self.status_var.set(status)

                if status.startswith("✅ Модели загружены"):
                    self.progress.stop()
                    self.progress_frame.pack_forget()
                    self.record_btn.config(state=tk.NORMAL)
                    self.clear_btn.config(state=tk.NORMAL)
                    self.save_btn.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        # Текст
        try:
            while True:
                text_data = self.text_queue.get_nowait()
                self._process_text_data(text_data)
        except queue.Empty:
            pass

        self.root.after(100, self._process_queues)

    def _process_text_data(self, text_data):
        """Обработка разных типов текстовых данных"""
        text_type, text, metadata = text_data

        if text_type == "raw":
            # Сырой текст - показываем сразу
            self._append_text(f"🔹 {text}\n", "raw")
            self.last_raw_text = text

        elif text_type == "enhanced":
            # Улучшенный текст - заменяем сырой
            changes = metadata.get('changes', [])
            if changes:
                self.status_queue.put(f"✅ Улучшено: {', '.join(changes)}")
            self._replace_last_text(f"✨ {text}\n\n")

        elif text_type == "partial":
            # Частичный результат
            self.status_var.set(f"🎤 Распознаю: {text}...")

        elif text_type == "llm_processing":
            # Статус обработки LLM
            self.status_var.set("✍️ Улучшаем текст...")

    def _append_text(self, text, text_type="normal"):
        """Добавление текста с разными стилями"""
        self.text_widget.config(state=tk.NORMAL)

        if text_type == "raw":
            # Сырой текст - серый цвет
            self.text_widget.insert(tk.END, text)
            self.text_widget.tag_add("raw", "end-2l", "end-1l")
            self.text_widget.tag_config("raw", foreground="gray")
        else:
            self.text_widget.insert(tk.END, text)

        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def _replace_last_text(self, enhanced_text):
        """Замена последнего сырого текста улучшенным"""
        self.text_widget.config(state=tk.NORMAL)

        # Находим и удаляем последнюю сырую строку
        content = self.text_widget.get("1.0", tk.END)
        lines = content.split('\n')

        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("🔹"):
                # Удаляем эту строку
                line_start = f"{i + 1}.0"
                line_end = f"{i + 2}.0"
                self.text_widget.delete(line_start, line_end)
                break

        # Добавляем улучшенный текст
        self.text_widget.insert(tk.END, enhanced_text)
        self.text_widget.see(tk.END)
        self.text_widget.config(state=tk.DISABLED)

    def _needs_llm_correction(self, text):
        """⚡ УЛУЧШЕНИЕ: Умная проверка необходимости LLM"""
        words = text.split()

        # Слишком короткие фразы не нуждаются в LLM
        if len(words) < 3:
            return False

        # Если уже есть пунктуация и фраза короткая - не нужно
        if any(punct in text for punct in '.!?') and len(words) < 6:
            return False

        # Проверяем признаки, где LLM реально поможет
        needs_correction = (
            # Длинные фразы без пунктуации
                (len(words) >= 4 and not any(punct in text for punct in '.!?,:')) or
                # Есть числа, которые можно нормализовать
                any(word.isdigit() for word in words) or
                # Потенциальные омофоны или слитное написание
                any(pattern in text.lower() for pattern in [
                    'какдела', 'чтоты', 'чтобы', 'зачемты', 'потомучто'
                ]) or
                # Отсутствуют предлоги в нужных местах
                self._missing_prepositions(text)
        )

        return needs_correction

    def _missing_prepositions(self, text):
        """Проверка отсутствия предлогов"""
        words = text.split()
        common_verbs = ['пошел', 'пришел', 'ушел', 'вернулся', 'зашел']
        following_nouns = ['магазин', 'дом', 'работа', 'улица', 'парк']

        for i, word in enumerate(words[:-1]):
            if word in common_verbs and words[i + 1] in following_nouns:
                return True
        return False

    def _enhance_with_llm(self, text):
        """⚡ УЛУЧШЕНИЕ: Реально полезное улучшение текста"""
        if not text.strip():
            return text, []

        try:
            # ⚡ УЛУЧШЕНИЕ: Подаем текст БЕЗ префиксов - нормализатор сам понимает задачу
            prompt = text

            inputs = self.llm_tokenizer(
                [prompt],
                return_tensors="pt",
                max_length=150,
                truncation=True,
                padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_length=200,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.1,  # Консервативная генерация
                    no_repeat_ngram_size=2
                )

            result = self.llm_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # ⚡ УЛУЧШЕНИЕ: Анализируем изменения
            changes = self._analyze_changes(text, result)

            # Проверяем, что улучшение действительно полезно
            if self._is_improvement_worthwhile(text, result, changes):
                return result, changes
            else:
                return text, []

        except Exception as e:
            logging.error(f"Ошибка LLM улучшения: {e}")
            return text, []

    def _analyze_changes(self, original, enhanced):
        """Анализ изменений между оригиналом и улучшенной версией"""
        if original == enhanced:
            return []

        changes = []
        orig_words = original.split()
        enh_words = enhanced.split()

        # Простые сравнения для демонстрации
        if len(orig_words) != len(enh_words):
            changes.append("структура предложения")

        # Проверяем добавление пунктуации
        orig_punct = set(re.findall(r'[.,!?;:]', original))
        enh_punct = set(re.findall(r'[.,!?;:]', enhanced))
        new_punct = enh_punct - orig_punct
        if new_punct:
            changes.append("пунктуация")

        # Проверяем добавление предлогов
        prepositions = ['в', 'на', 'за', 'под', 'о', 'у', 'с', 'по']
        orig_prep = sum(1 for word in orig_words if word in prepositions)
        enh_prep = sum(1 for word in enh_words if word in prepositions)
        if enh_prep > orig_prep:
            changes.append("предлоги")

        return changes

    def _is_improvement_worthwhile(self, original, enhanced, changes):
        """Проверка, что улучшение действительно полезно"""
        if not changes:
            return False

        # Если текст стал значительно короче/длиннее без явной пользы
        len_diff = abs(len(enhanced) - len(original)) / len(original)
        if len_diff > 0.5:  # Более 50% изменения длины
            return False

        # Проверяем, что основные слова сохранились
        orig_words = set(original.lower().split())
        enh_words = set(enhanced.lower().split())
        common_words = orig_words.intersection(enh_words)

        if len(common_words) / max(len(orig_words), 1) < 0.6:
            return False  # Слишком много изменений

        return True

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
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

        self.worker_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self.worker_thread.start()

    def _recording_worker(self):
        """Улучшенный рабочий процесс записи"""
        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

        def audio_callback(indata, frames, time, status):
            if self.stop_event.is_set():
                raise sd.CallbackStop()

            if status:
                logging.warning(f"Аудио статус: {status}")

            try:
                pcm_data = (indata * 32767).astype(np.int16).tobytes()

                if recognizer.AcceptWaveform(pcm_data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        # Всегда показываем сырой текст
                        self.text_queue.put(("raw", text, {}))

                        # ⚡ УЛУЧШЕНИЕ: Умное использование LLM
                        if self.use_llm.get() and self._needs_llm_correction(text):
                            self.text_queue.put(("llm_processing", "", {}))
                            threading.Thread(
                                target=self._process_with_llm,
                                args=(text,),
                                daemon=True
                            ).start()
                        else:
                            # Если LLM не нужна, просто копируем текст как улучшенный
                            self.text_queue.put(("enhanced", text, {'changes': []}))

                else:
                    # Частичные результаты
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        self.text_queue.put(("partial", partial_text, {}))

            except Exception as e:
                logging.error(f"Ошибка в callback: {e}")
                self.stop_event.set()

        try:
            with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32',
                    blocksize=self.chunk_size,
                    callback=audio_callback,
                    latency='low'
            ):
                while not self.stop_event.is_set():
                    sd.sleep(100)

        except sd.CallbackStop:
            logging.info("Запись остановлена")
        except Exception as e:
            logging.error(f"Ошибка записи: {e}")
            self.status_queue.put(f"❌ Ошибка записи: {str(e)}")
        finally:
            self._finalize_recording(recognizer)

    def _process_with_llm(self, text):
        """Асинхронная обработка LLM"""
        try:
            enhanced, changes = self._enhance_with_llm(text)
            self.text_queue.put(("enhanced", enhanced, {'changes': changes}))
        except Exception as e:
            logging.error(f"Ошибка LLM обработки: {e}")
            self.text_queue.put(("enhanced", text, {'changes': []}))

    def _finalize_recording(self, recognizer):
        """Завершение записи"""
        try:
            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                self.text_queue.put(("enhanced", final_text, {'changes': []}))
        except Exception as e:
            logging.warning(f"Ошибка финализации: {e}")
        finally:
            with self.recording_lock:
                self.is_recording = False
            self.root.after(0, self._recording_stopped_ui)

    def _recording_stopped_ui(self):
        """Восстановление UI"""
        self.record_btn.config(
            text="🎤 Начать запись",
            bg="#27ae60",
            state=tk.NORMAL
        )
        self.status_var.set("✅ Запись завершена")

    def stop_recording(self):
        if not self.is_recording:
            return

        self.stop_event.set()
        self.record_btn.config(state=tk.DISABLED)
        self.status_var.set("🔄 Безопасная остановка...")

    def clear_text(self):
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.config(state=tk.DISABLED)
        self.last_raw_text = ""
        self.status_var.set("📝 Текст очищен")

    def save_text(self):
        """Сохранение текста в файл"""
        try:
            text_content = self.text_widget.get("1.0", tk.END).strip()
            if not text_content:
                messagebox.showwarning("Предупреждение", "Нет текста для сохранения")
                return

            from tkinter import filedialog
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("Все файлы", "*.*")]
            )

            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                self.status_var.set(f"✅ Текст сохранен в {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {str(e)}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("transcriber.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    try:
        root = tk.Tk()
        app = ProfessionalVoiceTranscriber(root)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Критическая ошибка: {e}")
        messagebox.showerror("Ошибка", f"Не удалось запустить приложение:\n{str(e)}")


if __name__ == "__main__":
    main()
