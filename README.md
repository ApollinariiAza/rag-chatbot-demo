# RAG Chatbot Demo

Retrieval-Augmented Generation чатбот для работы с локальными документами. Система извлекает текст из различных форматов, создает семантические эмбеддинги и использует их для контекстных ответов через LLM API.

## Возможности

- **Обработка документов**: PDF, DOCX, PPTX, TXT, MD
- **Семантический поиск**: SentenceTransformers + FAISS для быстрого поиска
- **LLM интеграция**: Mistral API и локальные модели
- **Веб-интерфейсы**: Streamlit и Gradio
- **Сохранение базы знаний**: Экспорт/импорт индексированных документов

## Технологии

- **Python 3.8+**
- **FAISS** - векторная база данных
- **SentenceTransformers** - семантические эмбеддинги
- **LangChain** - обработка текста
- **PyPDF2, python-docx, python-pptx** - парсинг документов
- **Streamlit/Gradio** - веб-интерфейсы

## Установка

### Автоматическая установка

```bash
python setup_rag.py
```

### Ручная установка

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

### Настройка API ключей

```bash
cp .env.example .env
# Отредактируйте .env и добавьте MISTRAL_API_KEY
```

## Быстрый старт

### 1. Создание демо документов

```bash
python create_demo_docs.py
```

### 2. Запуск через launcher

```bash
python launch_demo.py
```

### 3. Командная строка

```bash
# Загрузка конкретных файлов (Windows-совместимо)
python main.py --documents demo_documents/machine_learning_guide.docx demo_documents/ai_ethics.txt --interactive

# Одиночный вопрос
python main.py --documents demo_documents/ai_ethics.txt --query "Что такое этика ИИ?"

# Сохранение базы знаний
python main.py --documents demo_documents/* --save-kb my_knowledge_base
```

### 4. Веб-интерфейсы

```bash
# Streamlit
streamlit run web_interface.py

# Gradio
python gradio_interface.py
```

## Структура проекта

```
rag-chatbot-demo/
├── main.py                    # Основной RAG чатбот
├── web_interface.py           # Streamlit интерфейс
├── gradio_interface.py        # Gradio интерфейс
├── create_demo_docs.py        # Генератор демо документов
├── test_rag.py               # Тесты системы
├── setup_rag.py              # Установщик
├── launch_demo.py            # Launcher для быстрого запуска
├── requirements.txt          # Зависимости Python
├── .env.example             # Пример конфигурации
├── demo_documents/          # Папка с демо файлами
├── knowledge_bases/         # Сохраненные базы знаний
└── README.md               # Документация
```

## Использование

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--documents` | Пути к документам | - |
| `--interactive` | Интерактивный чат | - |
| `--query` | Одиночный вопрос | - |
| `--save-kb` | Сохранить базу знаний | - |
| `--load-kb` | Загрузить базу знаний | - |
| `--embedding-model` | Модель эмбеддингов | all-MiniLM-L6-v2 |
| `--llm-provider` | Провайдер LLM | mistral |
| `--chunk-size` | Размер фрагментов | 500 |

### Модели эмбеддингов

- **all-MiniLM-L6-v2** - быстрая, 384 измерения
- **all-mpnet-base-v2** - качественная, 768 измерений
- **paraphrase-MiniLM-L6-v2** - для поиска парафразов

### Поддерживаемые форматы

| Формат | Расширения | Особенности |
|--------|------------|-------------|
| PDF | .pdf | Извлечение текста из страниц |
| Word | .docx, .doc | Параграфы и форматирование |
| PowerPoint | .pptx, .ppt | Текст из слайдов |
| Текст | .txt, .md | Прямое чтение |

## Примеры

### Базовый пример

```bash
python main.py --documents document.pdf --query "О чем этот документ?"
```

### Работа с множественными документами

```bash
python main.py \
  --documents reports/report1.pdf reports/report2.pdf \
  --embedding-model all-mpnet-base-v2 \
  --chunk-size 300 \
  --interactive
```

### Программное использование

```python
from main import RAGChatbot
import asyncio

async def example():
    chatbot = RAGChatbot(
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="mistral"
    )
    
    # Загружаем документы
    num_chunks = chatbot.load_documents(["document.pdf"])
    
    # Задаем вопрос
    response = await chatbot.ask("Главные выводы документа?")
    print(response['answer'])

asyncio.run(example())
```

## Производительность

### Системные требования

- **RAM**: 8GB+, рекомендуется 16GB
- **CPU**: Многоядерный процессор
- **Место**: 5GB для моделей

### Время обработки

| Объем | Индексация | Поиск |
|-------|------------|-------|
| 10 страниц | ~30 сек | ~0.1 сек |
| 100 страниц | ~5 мин | ~0.2 сек |
| 1000 страниц | ~30 мин | ~0.5 сек |

### Оптимизация

```bash
# GPU версия FAISS (быстрее)
pip uninstall faiss-cpu
pip install faiss-gpu

# Уменьшение размера модели
python main.py --embedding-model all-MiniLM-L6-v2 --chunk-size 800
```

## Тестирование

```bash
# Автоматические тесты
pytest test_rag.py

# Ручное тестирование
python test_rag.py --manual

# Тест с демо данными
python main.py --documents demo_documents/ai_ethics.txt --query "Что такое этика ИИ?"
```

## Устранение проблем

### Частые ошибки

**Ошибка "No module named"**
```bash
pip install -r requirements.txt
```

**Проблемы с wildcard в Windows**
```bash
# Не работает: demo_documents/*
# Используйте полные пути:
python main.py --documents demo_documents/file1.pdf demo_documents/file2.txt
```

**Недостаточно памяти**
```bash
# Уменьшите размер батча
export TOKENIZERS_PARALLELISM=false
```

**Медленная работа**
```bash
# Используйте быструю модель
python main.py --embedding-model all-MiniLM-L6-v2 --chunk-size 800
```

### Логирование

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## API Reference

### RAGChatbot

```python
chatbot = RAGChatbot(
    embedding_model="all-MiniLM-L6-v2",    # Модель эмбеддингов
    llm_provider="mistral",                 # Провайдер LLM
    llm_model="mistral-small",             # Модель LLM
    chunk_size=500,                        # Размер фрагмента
    chunk_overlap=50                       # Перекрытие фрагментов
)

# Загрузка документов
chunks = chatbot.load_documents(["file.pdf"])

# Вопрос-ответ
response = await chatbot.ask("Вопрос", k=5)  # k - количество контекстных документов

# Сохранение/загрузка
chatbot.save_knowledge_base("path")
chatbot.load_knowledge_base("path")
```

### VectorStore

```python
from main import VectorStore

store = VectorStore("all-MiniLM-L6-v2")
store.add_documents(texts, metadatas)
results = store.search(query, k=5)
store.save("path")
store.load("path")
```

## Веб-интерфейсы

### Streamlit

- Загрузка файлов через браузер
- Интерактивный чат с историей
- Просмотр источников и релевантности
- Экспорт базы знаний

### Gradio

- Простой интерфейс
- Настройка параметров модели
- Статистика базы знаний
- Экспорт результатов

## Лицензия

MIT License

## Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Добавьте тесты
4. Создайте Pull Request

## Поддержка

При возникновении проблем:
1. Проверьте раздел "Устранение проблем"
2. Запустите тесты: `python test_rag.py --manual`
3. Создайте issue с подробным описанием проблемы
