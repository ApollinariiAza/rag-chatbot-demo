import streamlit as st
import asyncio
import os
from pathlib import Path
import tempfile
import time

from main import RAGChatbot, DocumentProcessor
import logging

# Настройка страницы
st.set_page_config(
    page_title="RAG Chatbot Demo",
    page_icon="🤖",
    layout="wide"
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def init_chatbot():
    """Инициализация чатбота (кэшируется)"""
    return RAGChatbot(
        embedding_model=st.session_state.get('embedding_model', 'all-MiniLM-L6-v2'),
        llm_provider=st.session_state.get('llm_provider', 'mistral'),
        llm_model=st.session_state.get('llm_model', 'mistral-small'),
        chunk_size=st.session_state.get('chunk_size', 500)
    )

def save_uploaded_file(uploaded_file):
    """Сохраняет загруженный файл во временную директорию"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

async def process_query(chatbot, query):
    """Обрабатывает запрос пользователя"""
    return await chatbot.ask(query)

def main():
    st.title("🤖 RAG Chatbot Demo")
    st.markdown("*Retrieval-Augmented Generation чатбот с поддержкой документов*")
    
    # Боковая панель с настройками
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Настройки модели
        embedding_model = st.selectbox(
            "Модель эмбеддингов:",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
            index=0
        )
        
        llm_provider = st.selectbox(
            "Провайдер LLM:",
            ["mistral", "local"],
            index=0
        )
        
        llm_model = st.text_input(
            "Модель LLM:",
            value="mistral-small"
        )
        
        chunk_size = st.slider(
            "Размер чанка:",
            min_value=200,
            max_value=1000,
            value=500,
            step=50
        )
        
        # Сохраняем настройки в session state
        st.session_state.embedding_model = embedding_model
        st.session_state.llm_provider = llm_provider
        st.session_state.llm_model = llm_model
        st.session_state.chunk_size = chunk_size
        
        st.divider()
        
        # Управление базой знаний
        st.header("📚 База знаний")
        
        # Загрузка базы знаний
        kb_file = st.file_uploader(
            "Загрузить базу знаний (.pkl):",
            type=['pkl'],
            key="kb_upload"
        )
        
        if kb_file and st.button("Загрузить базу знаний"):
            try:
                # Сохраняем файл временно
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(kb_file.getvalue())
                    kb_path = tmp_file.name[:-4]  # Убираем .pkl для функции load
                
                # Загружаем в чатбот
                chatbot = init_chatbot()
                chatbot.load_knowledge_base(kb_path)
                st.success("База знаний загружена!")
                os.unlink(tmp_file.name)  # Удаляем временный файл
                
            except Exception as e:
                st.error(f"Ошибка загрузки базы знаний: {e}")
        
        # Сохранение базы знаний
        if st.button("Сохранить базу знаний"):
            try:
                chatbot = init_chatbot()
                kb_path = "knowledge_base"
                chatbot.save_knowledge_base(kb_path)
                
                # Предоставляем файлы для скачивания
                with open(f"{kb_path}.pkl", "rb") as f:
                    st.download_button(
                        label="Скачать .pkl файл",
                        data=f.read(),
                        file_name="knowledge_base.pkl",
                        mime="application/octet-stream"
                    )
                
                st.success("База знаний сохранена!")
                
            except Exception as e:
                st.error(f"Ошибка сохранения базы знаний: {e}")
    
    # Основной интерфейс
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📄 Загрузка документов")
        
        # Загрузка файлов
        uploaded_files = st.file_uploader(
            "Выберите документы для добавления в базу знаний:",
            type=['pdf', 'docx', 'doc', 'pptx', 'ppt', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Обработать документы"):
            chatbot = init_chatbot()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            temp_files = []
            try:
                # Сохраняем загруженные файлы
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Сохранение файла {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                    temp_file_path = save_uploaded_file(uploaded_file)
                    temp_files.append(temp_file_path)
                    progress_bar.progress((i + 1) * 0.3 / len(uploaded_files))
                
                # Обрабатываем документы
                status_text.text("Обработка документов и создание эмбеддингов...")
                num_chunks = chatbot.load_documents(temp_files)
                progress_bar.progress(1.0)
                
                status_text.text(f"Обработка завершена! Создано {num_chunks} текстовых фрагментов.")
                st.success(f"Успешно обработано {len(uploaded_files)} документов и создано {num_chunks} фрагментов!")
                
            except Exception as e:
                st.error(f"Ошибка при обработке документов: {e}")
                logger.error(f"Ошибка обработки документов: {e}")
            
            finally:
                # Удаляем временные файлы
                for temp_file in temp_files:
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        st.divider()
        
        # Чат интерфейс
        st.header("💬 Чат с документами")
        
        # История чата
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Отображаем историю чата
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Показываем источники для ответов бота
                if message["role"] == "assistant" and "sources" in message:
                    if message["sources"]:
                        with st.expander("Источники информации"):
                            for i, source in enumerate(message["sources"], 1):
                                st.write(f"**{i}. {Path(source['source']).name}** (релевантность: {source['score']:.3f})")
                                st.write(source['content'])
                                st.divider()
        
        # Поле ввода для нового сообщения
        if user_query := st.chat_input("Задайте вопрос по документам..."):
            # Добавляем сообщение пользователя
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_query
            })
            
            with st.chat_message("user"):
                st.write(user_query)
            
            # Получаем ответ от чатбота
            with st.chat_message("assistant"):
                with st.spinner("Думаю..."):
                    try:
                        chatbot = init_chatbot()
                        
                        # Асинхронный вызов
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(process_query(chatbot, user_query))
                        loop.close()
                        
                        st.write(response['answer'])
                        
                        # Добавляем ответ в историю
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": response['sources']
                        })
                        
                        # Показываем источники
                        if response['sources']:
                            with st.expander("Источники информации"):
                                for i, source in enumerate(response['sources'], 1):
                                    st.write(f"**{i}. {Path(source['source']).name}** (релевантность: {source['score']:.3f})")
                                    st.write(source['content'])
                                    st.divider()
                    
                    except Exception as e:
                        error_msg = f"Ошибка при обработке запроса: {e}"
                        st.error(error_msg)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                            "sources": []
                        })
        
        # Кнопка очистки истории чата
        if st.button("Очистить историю чата"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        st.header("📊 Статистика")
        
        try:
            chatbot = init_chatbot()
            num_docs = len(chatbot.vector_store.documents)
            
            st.metric("Документов в базе", num_docs)
            
            if num_docs > 0:
                # Показываем информацию о документах
                st.subheader("Загруженные документы:")
                
                sources = set()
                for metadata in chatbot.vector_store.metadata:
                    if 'source' in metadata:
                        sources.add(Path(metadata['source']).name)
                
                for source in sorted(sources):
                    st.write(f"• {source}")
                
                # Статистика по типам файлов
                file_types = {}
                for metadata in chatbot.vector_store.metadata:
                    if 'file_type' in metadata:
                        file_type = metadata['file_type']
                        file_types[file_type] = file_types.get(file_type, 0) + 1
                
                if file_types:
                    st.subheader("Типы файлов:")
                    for file_type, count in file_types.items():
                        st.write(f"• {file_type}: {count} фрагментов")
        
        except Exception as e:
            st.error(f"Ошибка получения статистики: {e}")
        
        st.divider()
        
        st.header("ℹ️ Информация")
        st.write("**Поддерживаемые форматы:**")
        st.write("• PDF (.pdf)")
        st.write("• Word (.docx, .doc)")
        st.write("• PowerPoint (.pptx, .ppt)")
        st.write("• Текстовые файлы (.txt, .md)")
        
        st.write("**Как использовать:**")
        st.write("1. Загрузите документы")
        st.write("2. Дождитесь обработки")
        st.write("3. Задавайте вопросы в чате")
        
        st.write("**Настройки:**")
        st.write("• Измените параметры в боковой панели")
        st.write("• Сохраните/загрузите базу знаний")

if __name__ == "__main__":
    main()