import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from MarkDown import generate_markdown_concept
import pygame
import os

# Инициализация Pygame для воспроизведения звука
pygame.mixer.init()

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚠ LALECTUTE ⚠")
        self.root.geometry("500x300")
        self.root.resizable(False, False)

        # Переменные
        self.file_path = tk.StringVar()
        self.language = tk.StringVar(value="russian")
        self.result_text = tk.StringVar()

        # Создание интерфейса
        self.create_widgets()

        # Загрузка модели (может занять время)
        self.load_model_thread = threading.Thread(target=self.load_model)
        self.load_model_thread.start()

    def create_widgets(self):
        # Выбор файла
        file_frame = ttk.Frame(self.root)
        file_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(file_frame, text="Выберите аудиофайл:").pack(side='left')
        ttk.Entry(file_frame, textvariable=self.file_path, width=40).pack(side='left', padx=5)
        ttk.Button(file_frame, text="Обзор", command=self.browse_file).pack(side='left')

        # Выбор языка
        lang_frame = ttk.Frame(self.root)
        lang_frame.pack(pady=10, padx=10, fill='x')

        ttk.Label(lang_frame, text="Выберите язык:").pack(side='left')
        ttk.Radiobutton(lang_frame, text='Русский', variable=self.language, value='russian').pack(side='left', padx=5)
        ttk.Radiobutton(lang_frame, text='Английский', variable=self.language, value='english').pack(side='left', padx=5)

        # Кнопки
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10, padx=10, fill='x')

        self.generate_button = ttk.Button(button_frame, text="Генерировать", command=self.start_generation)
        self.generate_button.pack(side='left', padx=5)

        self.save_button = ttk.Button(button_frame, text="Сохранить как...", command=self.save_text, state='disabled')
        self.save_button.pack(side='left', padx=5)

        self.play_button = ttk.Button(button_frame, text="Прослушать", command=self.play_audio, state='disabled')
        self.play_button.pack(side='left', padx=5)

        # Новая кнопка для преобразования в Markdown
        self.markdown_button = ttk.Button(button_frame, text="Markdown", command=self.convert_to_markdown)
        self.markdown_button.pack(side='left', padx=5)

        # Новая кнопка для информации
        info_button = ttk.Button(button_frame, text=" ⚠ ", command=self.show_info)
        info_button.pack(side='left', padx=5)

        # Поле результата
        result_frame = ttk.Frame(self.root)
        result_frame.pack(pady=10, padx=10, fill='both', expand=True)

        ttk.Label(result_frame, text="Результат:").pack(anchor='w')
        self.result_textbox = tk.Text(result_frame, height=5, wrap='word', state='disabled')
        self.result_textbox.pack(fill='both', expand=True)

        # Индикатор загрузки
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(pady=10, padx=10, fill='x')
        self.progress.stop()



    def show_info(self):
        info_message = (
            "🎉 Добро пожаловать в LALECTUTE! 🎉\n\n"
            "Это первая демо-версия приложения. Возможны некоторые ограничения в функционале.\n"
            "📢 Пожалуйста, используйте только подходящие аудиофайлы для этого проекта: *.wav *.mp3 *.ogg *.flac\n"
            "🍪 Я желаю вам удачи и много печенек! 🍪"
        )
        messagebox.showinfo("Информация", info_message)

    def convert_to_markdown(self):
        text = self.result_text.get()
        if not text:
            messagebox.showwarning("Предупреждение", "Нет текста для преобразования.")
            return

        self.markdown_button.config(state='disabled')
        self.progress.start()

        convert_thread = threading.Thread(target=self.generate_markdown)
        convert_thread.start()

    def generate_markdown(self):
        try:
            markdown = generate_markdown_concept(self.result_text.get())

            # Отладочный вывод
            print("Markdown сгенерирован:", markdown)  # Вывод сгенерированного Markdown

            if markdown:
                # Отображение Markdown в новом окне или сохранение в файл
                self.display_markdown(markdown)
            else:
                messagebox.showerror("Ошибка", "Не удалось преобразовать текст в Markdown.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при генерации Markdown: {e}")
        finally:
            self.progress.stop()
            self.markdown_button.config(state='normal')

    def display_markdown(self, markdown_text):
        # Пример отображения в новом окне
        markdown_window = tk.Toplevel(self.root)
        markdown_window.title("Markdown Конспект")
        text_box = tk.Text(markdown_window, wrap='word')
        text_box.insert(tk.END, markdown_text)
        text_box.config(state='disabled')
        text_box.pack(fill='both', expand=True)

        # Кнопка сохранения
        save_md_button = ttk.Button(markdown_window, text="Сохранить как...",
                                    command=lambda: self.save_markdown(markdown_text))
        save_md_button.pack(pady=5)

    def save_markdown(self, markdown_text):
        file = filedialog.asksaveasfilename(defaultextension=".md",
                                            filetypes=[("Markdown files", "*.md"), ("All files", "*.*")])
        if file:
            try:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(markdown_text)
                messagebox.showinfo("Успех", "Markdown успешно сохранён.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def browse_file(self):
        filetypes = (
            ("Audio files", "*.wav *.mp3 *.ogg *.flac"),
            ("All files", "*.*")
        )
        filename = filedialog.askopenfilename(title="Выберите аудиофайл", initialdir="/", filetypes=filetypes)
        if filename:
            self.file_path.set(filename)
            self.play_button.config(state='normal')

    def load_model(self):
        try:
            self.update_status("⚠⚠⚠ Загрузка модели...⚠⚠⚠")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            model_id = "openai/whisper-large-v3"

            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            self.model.to(device)

            self.processor = AutoProcessor.from_pretrained(model_id)

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                torch_dtype=torch_dtype,
                device=0 if device == "cuda:0" else -1,
            )

            self.update_status("Модель загружена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
            self.update_status("Ошибка загрузки модели.")

    def start_generation(self):
        if not self.file_path.get():
            messagebox.showwarning("Предупреждение", "Пожалуйста, выберите аудиофайл.")
            return
        self.generate_button.config(state='disabled')
        self.save_button.config(state='disabled')
        self.progress.start()
        self.result_textbox.config(state='normal')
        self.result_textbox.delete(1.0, tk.END)
        self.result_textbox.config(state='disabled')

        gen_thread = threading.Thread(target=self.generate_text)
        gen_thread.start()

    def generate_text(self):
        try:
            self.update_status("Генерация текста...")
            result = self.pipe(self.file_path.get(), generate_kwargs={"language": self.language.get()})
            text = result["text"]

            self.result_text.set(text)
            self.result_textbox.config(state='normal')
            self.result_textbox.delete(1.0, tk.END)
            self.result_textbox.insert(tk.END, text)
            self.result_textbox.config(state='disabled')

            self.save_button.config(state='normal')
            self.update_status("Генерация завершена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сгенерировать текст: {e}")
            self.update_status("Ошибка генерации.")
        finally:
            self.progress.stop()
            self.generate_button.config(state='normal')
    def save_text(self):
        if not self.result_text.get():
            messagebox.showwarning("Предупреждение", "Нет текста для сохранения.")
            return
        file = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file:
            try:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get())
                messagebox.showinfo("Успех", "Текст успешно сохранен.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def play_audio(self):
        try:
            if os.path.exists(self.file_path.get()):
                pygame.mixer.music.load(self.file_path.get())
                pygame.mixer.music.play()
            else:
                messagebox.showerror("Ошибка", "Файл не найден.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось воспроизвести аудио: {e}")

    def update_status(self, message):
        self.root.title(f"LALECTUTE - {message}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
