#!/usr/bin/env python3
"""
Gradio веб-интерфейс для RAG Chatbot
Альтернатива Streamlit с более простым интерфейсом
"""

import gradio as gr
import asyncio
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

from main import RAGChatbot
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальный экземпляр чатбота
chatbot_instance = None
chat_history = []

def init_chatbot(embedding_model="all-MiniLM-L6-v2", 
                llm_provider="mistral", 
                chunk_size=500):
    """Инициализация чатбота"""
    global chatbot_instance
    
    try:
        chatbot_instance = RAGChatbot(
            embedding_model=embedding_model,
            llm_provider=llm_provider,
            chunk_size=int(chunk_size)
        )
        return f"✅ Чатбот инициализирован с моделью {embedding_model}"
    except Exception as e:
        return f"❌ Ошибка инициализации: {str(e)}"

def process_documents(files, progress=gr.Progress()):
    """Обработка загруженных документов"""
    global chatbot_instance
    
    if chatbot_instance is None:
        return "❌ Сначала инициализируйте чатбот", ""
    
    if not files:
        return "❌ Файлы не выбраны", ""
    
    temp_files = []
    processed_info = []
    
    try:
        progress(0, desc="Сохранение файлов...")
        
        # Сохраняем загруженные файлы
        for i, file in enumerate(files):
            file_path = file.name
            temp_files.append(file_path)
            processed_info.append({
                'name': Path(file_path).name,
                'size': f"{os.path.getsize(file_path) / 1024:.1f} KB"
            })
            progress((i + 1) / len(files) * 0.3)
        
        progress(0.5, desc="Обработка документов...")
        
        # Обрабатываем документы
        num_chunks = chatbot_instance.load_documents(temp_files)
        
        progress(1.0, desc="Завершено!")
        
        # Формируем отчет
        report = f"✅ Успешно обработано {len(files)} файлов\n"
        report += f"📊 Создано {num_chunks} текстовых фрагментов\n\n"
        report += "📄 Обработанные файлы:\n"
        
        for info in processed_info:
            report += f"• {info['name']} ({info['size']})\n"
        
        # Статистика базы знаний
        total_docs = len(chatbot_instance.vector_store.documents)
        kb_stats = f"\n📚 Всего в базе знаний: {total_docs} фрагментов"
        
        return report + kb_stats, ""
        
    except Exception as e:
        return f"❌ Ошибка обработки документов: {str(e)}", ""

def chat_with_documents(message, history):
    """Чат с документами"""
    global chatbot_instance, chat_history
    
    if chatbot_instance is None:
        response = "❌ Сначала инициализируйте чатбот в настройках"
        history.append([message, response])
        return history, ""
    
    if len(chatbot_instance.vector_store.documents) == 0:
        response = "📚 База знаний пуста. Загрузите документы перед началом чата."
        history.append([message, response])
        return history, ""
    
    try:
        # Асинхронный вызов
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(chatbot_instance.ask(message))
        loop.close()
        
        # Формируем ответ с источниками
        response = result['answer']
        
        if result['sources']:
            response += "\n\n📚 **Источники:**\n"
            for i, source in enumerate(result['sources'][:3], 1):
                file_name = Path(source['source']).name
                response += f"{i}. *{file_name}* (релевантность: {source['score']:.2f})\n"
                response += f"   {source['content'][:100]}...\n"
        
        # Добавляем в историю
        history.append([message, response])
        chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'query': message,
            'answer': result['answer'],
            'sources': result['sources']
        })
        
        return history, ""
        
    except Exception as e:
        error_msg = f"❌ Ошибка: {str(e)}"
        history.append([message, error_msg])
        return history, ""

def clear_chat():
    """Очистка истории чата"""
    global chat_history
    chat_history = []
    return []

def get_knowledge_base_stats():
    """Получение статистики базы знаний"""
    global chatbot_instance
    
    if chatbot_instance is None:
        return "❌ Чатбот не инициализирован"
    
    num_docs = len(chatbot_instance.vector_store.documents)
    
    if num_docs == 0:
        return "📚 База знаний пуста"
    
    # Анализируем источники
    sources = {}
    file_types = {}
    
    for metadata in chatbot_instance.vector_store.metadata:
        if 'source' in metadata:
            source_name = Path(metadata['source']).name
            sources[source_name] = sources.get(source_name, 0) + 1
        
        if 'file_type' in metadata:
            file_type = metadata['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
    
    # Формируем отчет
    stats = f"📊 **Статистика базы знаний:**\n\n"
    stats += f"📄 Всего фрагментов: {num_docs}\n"
    stats += f"📁 Уникальных файлов: {len(sources)}\n\n"
    
    if sources:
        stats += "**Файлы:**\n"
        for source, count in sorted(sources.items()):
            stats += f"• {source}: {count} фрагментов\n"
    
    if file_types:
        stats += "\n**Типы файлов:**\n"
        for file_type, count in sorted(file_types.items()):
            stats += f"• {file_type}: {count} фрагментов\n"
    
    return stats

def export_chat_history():
    """Экспорт истории чата"""
    global chat_history
    
    if not chat_history:
        return None, "📝 История чата пуста"
    
    # Создаем JSON с историей
    export_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_messages': len(chat_history),
        'chat_history': chat_history
    }
    
    # Сохраняем во временный файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
        temp_path = f.name
    
    return temp_path, f"✅ Экспортировано {len(chat_history)} сообщений"

def save_knowledge_base():
    """Сохранение базы знаний"""
    global chatbot_instance
    
    if chatbot_instance is None:
        return None, None, "❌ Чатбот не инициализирован"
    
    if len(chatbot_instance.vector_store.documents) == 0:
        return None, None, "📚 База знаний пуста"
    
    try:
        # Сохраняем в временные файлы
        with tempfile.NamedTemporaryFile(delete=False) as f:
            base_path = f.name
        
        chatbot_instance.save_knowledge_base(base_path)
        
        pkl_path = base_path + '.pkl'
        faiss_path = base_path + '.faiss'
        
        message = f"✅ База знаний сохранена ({len(chatbot_instance.vector_store.documents)} фрагментов)"
        
        return pkl_path, faiss_path, message
        
    except Exception as e:
        return None, None, f"❌ Ошибка сохранения: {str(e)}"

def create_interface():
    """Создание Gradio интерфейса"""
    
    with gr.Blocks(title="RAG Chatbot Demo", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🤖 RAG Chatbot Demo
        
        **Retrieval-Augmented Generation чатбот** - загружайте документы и задавайте вопросы!
        """)
        
        with gr.Tabs():
            
            # Вкладка настроек
            with gr.TabItem("⚙️ Настройки"):
                gr.Markdown("### Конфигурация чатбота")
                
                with gr.Row():
                    embedding_model = gr.Dropdown(
                        choices=["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"],
                        value="all-MiniLM-L6-v2",
                        label="Модель эмбеддингов"
                    )
                    
                    llm_provider = gr.Dropdown(
                        choices=["mistral", "local"],
                        value="mistral",
                        label="Провайдер LLM"
                    )
                    
                    chunk_size = gr.Slider(
                        minimum=200,
                        maximum=1000,
                        value=500,
                        step=50,
                        label="Размер фрагмента текста"
                    )
                
                init_btn = gr.Button("Инициализировать чатбот", variant="primary")
                init_status = gr.Textbox(label="Статус инициализации", interactive=False)
                
                init_btn.click(
                    fn=init_chatbot,
                    inputs=[embedding_model, llm_provider, chunk_size],
                    outputs=[init_status]
                )
            
            # Вкладка загрузки документов
            with gr.TabItem("📄 Документы"):
                gr.Markdown("### Загрузка и обработка документов")
                
                file_upload = gr.Files(
                    label="Выберите файлы (PDF, DOCX, TXT, PPTX)",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".pptx", ".ppt"]
                )
                
                process_btn = gr.Button("Обработать документы", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        process_status = gr.Textbox(
                            label="Статус обработки", 
                            interactive=False,
                            lines=10
                        )
                    
                    with gr.Column():
                        kb_stats = gr.Textbox(
                            label="Статистика базы знаний",
                            interactive=False,
                            lines=10
                        )
                
                process_btn.click(
                    fn=process_documents,
                    inputs=[file_upload],
                    outputs=[process_status, kb_stats]
                )
                
                # Кнопка обновления статистики
                refresh_stats_btn = gr.Button("Обновить статистику")
                refresh_stats_btn.click(
                    fn=get_knowledge_base_stats,
                    outputs=[kb_stats]
                )
            
            # Главная вкладка чата
            with gr.TabItem("💬 Чат", elem_id="chat-tab"):
                gr.Markdown("### Задавайте вопросы по документам")
                
                chatbot = gr.Chatbot(
                    label="Разговор с документами",
                    height=400,
                    show_copy_button=True
                )
                
                msg = gr.Textbox(
                    label="Ваш вопрос",
                    placeholder="Задайте вопрос по загруженным документам...",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Отправить", variant="primary")
                    clear_btn = gr.Button("Очистить чат")
                
                submit_btn.click(
                    fn=chat_with_documents,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
                
                msg.submit(
                    fn=chat_with_documents,
                    inputs=[msg, chatbot],
                    outputs=[chatbot, msg]
                )
                
                clear_btn.click(
                    fn=clear_chat,
                    outputs=[chatbot]
                )
            
            # Вкладка экспорта
            with gr.TabItem("💾 Экспорт"):
                gr.Markdown("### Сохранение данных")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**История чата**")
                        export_chat_btn = gr.Button("Экспортировать историю чата")
                        chat_file = gr.File(label="Файл истории чата")
                        chat_export_status = gr.Textbox(label="Статус экспорта", interactive=False)
                        
                        export_chat_btn.click(
                            fn=export_chat_history,
                            outputs=[chat_file, chat_export_status]
                        )
                    
                    with gr.Column():
                        gr.Markdown("**База знаний**")
                        save_kb_btn = gr.Button("Сохранить базу знаний")
                        
                        kb_pkl_file = gr.File(label="Файл базы знаний (.pkl)")
                        kb_faiss_file = gr.File(label="Файл индекса (.faiss)")
                        kb_save_status = gr.Textbox(label="Статус сохранения", interactive=False)
                        
                        save_kb_btn.click(
                            fn=save_knowledge_base,
                            outputs=[kb_pkl_file, kb_faiss_file, kb_save_status]
                        )
        
        # Футер с информацией
        gr.Markdown("""
        ---
        **Поддерживаемые форматы:** PDF, DOCX, TXT, MD, PPTX  
        **Модели эмбеддингов:** SentenceTransformers  
        **Векторная БД:** FAISS  
        **LLM:** Mistral API / локальные модели
        """)
    
    return demo

def main():
    """Запуск Gradio интерфейса"""
    
    # Создаем интерфейс
    demo = create_interface()
    
    # Настройки запуска
    port = int(os.getenv('GRADIO_PORT', 7860))
    host = os.getenv('GRADIO_HOST', '127.0.0.1')
    
    print("Запуск RAG Chatbot с Gradio интерфейсом...")
    print(f"Адрес: http://{host}:{port}")
    
    # Запуск
    demo.launch(
        server_name=host,
        server_port=port,
        share=False,  # Установите True для публичного доступа через Gradio tunnel
        show_error=True
    )

if __name__ == "__main__":
    main()