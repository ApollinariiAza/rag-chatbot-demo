#!/usr/bin/env python3
"""
Скрипт установки и настройки RAG Chatbot проекта
"""

import subprocess
import sys
import os
from pathlib import Path
import urllib.request
import zipfile
import tempfile

def install_requirements():
    """Устанавливает основные зависимости"""
    print("Установка основных зависимостей...")
    
    basic_requirements = [
        "torch",
        "sentence-transformers",
        "faiss-cpu",
        "nltk",
        "numpy",
        "pandas",
        "python-dotenv",
        "httpx",
        "PyPDF2",
        "python-docx",
        "python-pptx"
    ]
    
    for requirement in basic_requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"  {requirement}")
        except subprocess.CalledProcessError:
            print(f"  Ошибка установки {requirement}")
            return False
    
    return True

def install_optional_requirements():
    """Устанавливает опциональные зависимости"""
    print("\nУстановка опциональных зависимостей...")
    
    optional_requirements = [
        "langchain",
        "langchain-community", 
        "langchain-text-splitters",
        "streamlit",
        "gradio",
        "pytest",
        "pytest-asyncio"
    ]
    
    for requirement in optional_requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"  {requirement}")
        except subprocess.CalledProcessError:
            print(f"  Пропущен {requirement} (опциональный)")

def setup_nltk_data():
    """Загружает необходимые данные NLTK"""
    print("\nНастройка NLTK...")
    
    try:
        import nltk
        
        # Загружаем необходимые корпуса
        datasets = ['punkt', 'stopwords', 'wordnet']
        
        for dataset in datasets:
            try:
                nltk.download(dataset, quiet=True)
                print(f"  {dataset}")
            except:
                print(f"  Не удалось загрузить {dataset}")
        
        return True
    except ImportError:
        print("  NLTK не установлен")
        return False

def create_directories():
    """Создает необходимые директории"""
    print("\nСоздание директорий...")
    
    directories = [
        'documents',
        'knowledge_bases', 
        'exports',
        'demo_documents',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  {directory}/")

def setup_env_file():
    """Создает .env файл если его нет"""
    if not os.path.exists('.env'):
        print("\n🔧 Создание .env файла...")
        
        env_content = """# API Keys for LLM services
MISTRAL_API_KEY=your_mistral_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Local LLM endpoint (if using local models)
LOCAL_LLM_URL=http://localhost:8000

# Embedding model settings
DEFAULT_EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Text processing settings
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
MAX_CONTEXT_DOCUMENTS=5

# Web interface settings
STREAMLIT_SERVER_PORT=8501
GRADIO_PORT=7860
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("  .env файл создан")
        print("  Не забудьте добавить ваши API ключи!")
    else:
        print("\n.env файл уже существует")

def test_installation():
    """Тестирует установку основных компонентов"""
    print("\nТестирование установки...")
    
    tests = [
        ("sentence_transformers", "SentenceTransformers"),
        ("faiss", "FAISS"),
        ("nltk", "NLTK"),
        ("PyPDF2", "PyPDF2"),
        ("docx", "python-docx"),
        ("pptx", "python-pptx")
    ]
    
    all_passed = True
    
    for module, name in tests:
        try:
            __import__(module)
            print(f"  {name}")
        except ImportError:
            print(f"  {name} не установлен")
            all_passed = False
    
    # Тест загрузки модели эмбеддингов
    try:
        print("\nТестирование загрузки модели эмбеддингов...")
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["Тестовое предложение"])
        
        print(f"  Модель загружена, размерность: {len(test_embedding[0])}")
        
    except Exception as e:
        print(f"  Ошибка загрузки модели: {e}")
        all_passed = False
    
    # Тест FAISS
    try:
        print("\nТестирование FAISS...")
        import faiss
        import numpy as np
        
        # Создаем простой индекс
        dimension = 384
        index = faiss.IndexFlatIP(dimension)
        
        # Добавляем тестовые векторы
        test_vectors = np.random.random((5, dimension)).astype('float32')
        index.add(test_vectors)
        
        # Тестируем поиск
        scores, indices = index.search(test_vectors[:1], k=3)
        
        print(f"  FAISS работает, найдено {len(indices[0])} результатов")
        
    except Exception as e:
        print(f"  Ошибка FAISS: {e}")
        all_passed = False
    
    return all_passed

def create_demo_launcher():
    """Создает скрипт для быстрого запуска демо"""
    print("\nСоздание скрипта запуска демо...")
    
    launcher_content = '''#!/usr/bin/env python3
"""
Быстрый запуск RAG Chatbot Demo
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("RAG Chatbot Demo Launcher")
    print("=" * 40)
    
    # Проверяем наличие основных файлов
    required_files = ['rag_chatbot.py', 'create_demo_docs.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print(f"Отсутствуют файлы: {', '.join(missing_files)}")
        return
    
    while True:
        print("\\nВыберите режим запуска:")
        print("1. Создать демо-документы")
        print("2. Командная строка (интерактивный)")
        print("3. Веб-интерфейс (Streamlit)")
        print("4. Веб-интерфейс (Gradio)")
        print("5. Запустить тесты")
        print("0. Выход")
        
        choice = input("\\nВаш выбор: ").strip()
        
        if choice == "0":
            break
        elif choice == "1":
            subprocess.run([sys.executable, "create_demo_docs.py"])
        elif choice == "2":
            if Path("demo_documents").exists():
                subprocess.run([
                    sys.executable, "rag_chatbot.py", 
                    "--documents", "demo_documents/*", 
                    "--interactive"
                ])
            else:
                print("Создайте сначала демо-документы (опция 1)")
        elif choice == "3":
            try:
                subprocess.run([sys.executable, "-m", "streamlit", "run", "web_interface.py"])
            except FileNotFoundError:
                print("Streamlit не установлен")
        elif choice == "4":
            try:
                subprocess.run([sys.executable, "gradio_interface.py"])
            except FileNotFoundError:
                print("gradio_interface.py не найден")
        elif choice == "5":
            try:
                subprocess.run([sys.executable, "test_rag.py", "--manual"])
            except FileNotFoundError:
                print("test_rag.py не найден")
        else:
            print("Неверный выбор")

if __name__ == "__main__":
    main()
'''
    
    with open('launch_demo.py', 'w') as f:
        f.write(launcher_content)
    
    # Делаем исполняемым (на Unix системах)
    try:
        os.chmod('launch_demo.py', 0o755)
    except:
        pass
    
    print("  Создан launch_demo.py")

def main():
    """Основная функция установки"""
    print("Установка RAG Chatbot Demo")
    print("=" * 50)
    
    # Проверка Python версии
    if sys.version_info < (3, 8):
        print("Требуется Python 3.8 или новее")
        return False
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Установка зависимостей
    if not install_requirements():
        print("\nОшибка установки основных зависимостей")
        return False
    
    install_optional_requirements()
    
    # Настройка NLTK
    setup_nltk_data()
    
    # Создание директорий
    create_directories()
    
    # Создание .env файла
    setup_env_file()
    
    # Создание скрипта запуска
    create_demo_launcher()
    
    # Тестирование
    if test_installation():
        print("\n" + "=" * 50)
        print("Установка завершена успешно!")
        print("\nСледующие шаги:")
        print("1. Добавьте API ключи в .env файл")
        print("2. Запустите: python launch_demo.py")
        print("3. Или создайте демо: python create_demo_docs.py")
        print("4. Запустите чат: python rag_chatbot.py --documents demo_documents/* --interactive")
        print("\nВеб-интерфейсы:")
        print("• Streamlit: streamlit run web_interface.py")
        print("• Gradio: python gradio_interface.py")
        return True
    else:
        print("\nНекоторые компоненты установлены с ошибками")
        print("Проверьте вывод выше и переустановите проблемные компоненты")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)