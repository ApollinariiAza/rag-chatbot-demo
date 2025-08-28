#!/usr/bin/env python3
"""
Тесты для RAG Chatbot системы
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
import numpy as np

from main import (
    DocumentProcessor, TextSplitter, VectorStore, 
    RAGChatbot, LLMClient
)

class TestDocumentProcessor:
    """Тесты обработки документов"""
    
    def test_extract_text_from_txt(self):
        """Тест извлечения текста из TXT файла"""
        test_content = "Это тестовый текст для проверки извлечения."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            extracted_text = DocumentProcessor.extract_text(temp_path)
            assert extracted_text.strip() == test_content
        finally:
            os.unlink(temp_path)
    
    def test_unsupported_file_type(self):
        """Тест неподдерживаемого типа файла"""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            extracted_text = DocumentProcessor.extract_text(temp_path)
            assert extracted_text == ""
        finally:
            os.unlink(temp_path)

class TestTextSplitter:
    """Тесты разбиения текста"""
    
    def test_basic_text_splitting(self):
        """Тест базового разбиения текста"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10, use_langchain=False)
        
        text = "Это первое предложение. Это второе предложение. Это третье предложение. Это четвертое предложение."
        chunks = splitter.split_text(text)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 100 for chunk in chunks)  # Учитываем overlap
    
    def test_empty_text(self):
        """Тест пустого текста"""
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert len(chunks) == 0
    
    def test_short_text(self):
        """Тест короткого текста"""
        splitter = TextSplitter(chunk_size=100)
        text = "Короткий текст."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0].strip() == text

class TestVectorStore:
    """Тесты векторного хранилища"""
    
    def test_add_and_search_documents(self):
        """Тест добавления и поиска документов"""
        vector_store = VectorStore("all-MiniLM-L6-v2")
        
        # Добавляем тестовые документы
        test_docs = [
            "Python - это язык программирования.",
            "Машинное обучение использует алгоритмы для анализа данных.",
            "FAISS - это библиотека для быстрого поиска векторов."
        ]
        
        vector_store.add_documents(test_docs)
        
        # Проверяем, что документы добавлены
        assert len(vector_store.documents) == 3
        
        # Тестируем поиск
        results = vector_store.search("Python программирование", k=2)
        
        assert len(results) == 2
        assert all(0 <= result['score'] <= 1 for result in results)
        assert results[0]['score'] >= results[1]['score']  # Сортировка по релевантности
    
    def test_empty_search(self):
        """Тест поиска в пустом хранилище"""
        vector_store = VectorStore()
        results = vector_store.search("тест", k=5)
        assert len(results) == 0
    
    def test_save_and_load(self):
        """Тест сохранения и загрузки векторного хранилища"""
        vector_store = VectorStore()
        
        # Добавляем документы
        test_docs = ["Документ 1", "Документ 2"]
        vector_store.add_documents(test_docs)
        
        # Сохраняем
        with tempfile.NamedTemporaryFile(delete=False) as f:
            base_path = f.name
        
        try:
            vector_store.save(base_path)
            
            # Создаем новый экземпляр и загружаем
            new_vector_store = VectorStore()
            new_vector_store.load(base_path)
            
            # Проверяем, что данные загрузились
            assert len(new_vector_store.documents) == 2
            assert new_vector_store.documents == test_docs
            
        finally:
            # Удаляем временные файлы
            for ext in ['.faiss', '.pkl']:
                try:
                    os.unlink(base_path + ext)
                except:
                    pass

class MockLLMClient:
    """Мок-клиент для тестирования без реальных API вызовов"""
    
    def __init__(self, provider="mock", model_name="mock-model"):
        self.provider = provider
        self.model_name = model_name
    
    async def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Генерирует мок-ответ на основе промпта"""
        if "Python" in prompt:
            return "Python - это высокоуровневый язык программирования, известный своей простотой и читаемостью."
        elif "машинное обучение" in prompt.lower():
            return "Машинное обучение - это раздел искусственного интеллекта, который позволяет компьютерам обучаться на данных."
        else:
            return "Это тестовый ответ от мок-клиента LLM."

class TestRAGChatbot:
    """Тесты RAG чатбота"""
    
    @pytest.fixture
    def mock_chatbot(self):
        """Создает RAG чатбот с мок-клиентом"""
        chatbot = RAGChatbot()
        chatbot.llm_client = MockLLMClient()
        return chatbot
    
    def test_load_documents(self, mock_chatbot):
        """Тест загрузки документов"""
        # Создаем временный файл
        test_content = "Это тестовый документ для загрузки в RAG систему. Python - отличный язык программирования."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Загружаем документ
            num_chunks = mock_chatbot.load_documents([temp_path])
            
            assert num_chunks > 0
            assert len(mock_chatbot.vector_store.documents) == num_chunks
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_ask_question(self, mock_chatbot):
        """Тест задания вопроса"""
        # Добавляем тестовые документы
        test_docs = [
            "Python - это интерпретируемый язык программирования высокого уровня.",
            "Машинное обучение использует статистические алгоритмы для анализа данных."
        ]
        
        mock_chatbot.vector_store.add_documents(test_docs)
        
        # Задаем вопрос
        response = await mock_chatbot.ask("Что такое Python?")
        
        assert 'answer' in response
        assert 'sources' in response
        assert 'query' in response
        assert len(response['answer']) > 0
        assert len(response['sources']) > 0
    
    @pytest.mark.asyncio
    async def test_ask_empty_knowledge_base(self, mock_chatbot):
        """Тест вопроса к пустой базе знаний"""
        response = await mock_chatbot.ask("Любой вопрос")
        
        assert "База знаний пуста" in response['answer']
        assert len(response['sources']) == 0
    
    def test_create_prompt(self, mock_chatbot):
        """Тест создания промпта"""
        context_docs = [
            {
                'content': 'Python - это язык программирования.',
                'score': 0.95,
                'metadata': {'source': 'test.txt'}
            }
        ]
        
        prompt = mock_chatbot.create_prompt("Что такое Python?", context_docs)
        
        assert "КОНТЕКСТ:" in prompt
        assert "ВОПРОС:" in prompt
        assert "Python - это язык программирования." in prompt
        assert "Что такое Python?" in prompt

class TestIntegration:
    """Интеграционные тесты"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Тест полного пайплайна RAG"""
        # Создаем чатбот с мок-клиентом
        chatbot = RAGChatbot(chunk_size=100)
        chatbot.llm_client = MockLLMClient()
        
        # Создаем тестовый документ
        test_content = """
        Python - это высокоуровневый язык программирования.
        Он был создан Гвидо ван Россумом в 1991 году.
        Python известен своей простотой и читаемостью кода.
        Машинное обучение активно использует Python для анализа данных.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # 1. Загружаем документ
            num_chunks = chatbot.load_documents([temp_path])
            assert num_chunks > 0
            
            # 2. Задаем вопрос
            response = await chatbot.ask("Кто создал Python и когда?")
            
            # 3. Проверяем ответ
            assert len(response['answer']) > 0
            assert len(response['sources']) > 0
            assert response['sources'][0]['score'] > 0.5
            
            # 4. Тестируем сохранение/загрузку
            with tempfile.NamedTemporaryFile(delete=False) as kb_file:
                kb_path = kb_file.name
            
            try:
                chatbot.save_knowledge_base(kb_path)
                
                # Создаем новый чатбот и загружаем базу знаний
                new_chatbot = RAGChatbot()
                new_chatbot.llm_client = MockLLMClient()
                new_chatbot.load_knowledge_base(kb_path)
                
                # Проверяем, что данные загрузились
                assert len(new_chatbot.vector_store.documents) == num_chunks
                
                # Задаем вопрос новому чатботу
                response2 = await new_chatbot.ask("Что такое Python?")
                assert len(response2['answer']) > 0
                
            finally:
                # Удаляем файлы базы знаний
                for ext in ['.faiss', '.pkl']:
                    try:
                        os.unlink(kb_path + ext)
                    except:
                        pass
        
        finally:
            os.unlink(temp_path)

def run_manual_tests():
    """Запуск ручных тестов для проверки основной функциональности"""
    
    print("Запуск ручных тестов RAG системы...")
    print("=" * 50)
    
    # Тест 1: Обработка документов
    print("Тест 1: Обработка документов")
    processor = DocumentProcessor()
    
    # Создаем тестовый файл
    test_content = "Тестовое содержимое для проверки обработки документов."
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    try:
        extracted = processor.extract_text(temp_path)
        assert extracted.strip() == test_content
        print("Обработка документов работает")
    except Exception as e:
        print(f"Ошибка в обработке документов: {e}")
    finally:
        os.unlink(temp_path)
    
    # Тест 2: Разбиение текста
    print("\nТест 2: Разбиение текста")
    try:
        splitter = TextSplitter(chunk_size=50, use_langchain=False)
        text = "Первое предложение. Второе предложение. Третье предложение."
        chunks = splitter.split_text(text)
        assert len(chunks) > 0
        print(f"Создано {len(chunks)} фрагментов текста")
    except Exception as e:
        print(f"Ошибка в разбиении текста: {e}")
    
    # Тест 3: Векторное хранилище
    print("\nТест 3: Векторное хранилище")
    try:
        vector_store = VectorStore("all-MiniLM-L6-v2")
        test_docs = ["Первый документ", "Второй документ", "Третий документ"]
        vector_store.add_documents(test_docs)
        
        results = vector_store.search("документ", k=2)
        assert len(results) == 2
        print(f"Поиск вернул {len(results)} результатов")
        print(f"   Лучший результат: score={results[0]['score']:.3f}")
    except Exception as e:
        print(f"Ошибка в векторном хранилище: {e}")
    
    print("\n" + "=" * 50)
    print("Ручные тесты завершены!")

if __name__ == "__main__":
    # Запуск ручных тестов если не используется pytest
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--manual":
        run_manual_tests()
    else:
        print("Для запуска автоматических тестов используйте: pytest test_rag.py")
        print("Для запуска ручных тестов используйте: python test_rag.py --manual")
        print("\nДоступные тесты:")
        print("• test_document_processor - тестирование обработки документов")
        print("• test_text_splitter - тестирование разбиения текста")
        print("• test_vector_store - тестирование векторного хранилища")
        print("• test_rag_chatbot - тестирование RAG чатбота")
        print("• test_integration - интеграционные тесты")