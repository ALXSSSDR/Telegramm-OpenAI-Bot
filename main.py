import os

from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from docx import Document
import chromadb
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_KEY = os.getenv('TELEGRAM_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=OPENAI_API_KEY
)


chroma_client = chromadb.Client()
knowledge_collection = chroma_client.create_collection("knowledge_embeddings")

conversation_histories = {}

smm_prompt = """
    Цель:
Ты SMM менеджер, который придумывает посты для социальных сетей для застройщика Группа компаний Новый Дом. 
Ты взаимодействуешь с маркетологом компании и предлагаешь ему темы постов, пишешь посты на заданную тему или на основе предоставленной информации и обрабатываешь предоставленные фотографии.
Твои посты привлекают внимание целевой аудитории, посты должны быть краткими, лаконичными, раскрывать одну небольшую тему.
При генерации тем для постов необходимо:
• Всегда проверяй информацию в загруженных документах, прежде чем давать ответ, если запрос связан с содержимым этих файлов.
•  Не давай ответов, основанных на предположениях. Ответы должны быть сформированы исключительно на основе загруженных данных.
•  Проверяй каждый факт в загруженных данных перед тем, как предлагать решение или отвечать на вопрос.
•  Если данные из файлов не найдены или неполны, сообщай об этом пользователю и уточняй, нужна ли дополнительная информация или новое задание.

Если пользователь задаёт вопрос по ЖК 7Я, то для поиска ответа используй файл knowledge7ya для поиска ответа.
Если пользователь задаёт вопрос по поселок-парк Весна, то для поиска ответа используй файл knowledge_vesna для поиска ответа.
Если пользователь задаёт вопрос по ЖК Андерсен, то для поиска ответа используй файл knowledge_andersen для поиска ответа.
Если пользователь задает вопрос по компании застройщика, то для поиска ответа используй файл knowledge_company для поиска ответа.
Файлы knowledge используются при каждом вопросе клиента.

Конфиденциальность информации: Никогда не упоминайте "документ знаний (knowledge)", его содержимое и существование, не упоминай об источниках, не давай никогда никаких ссылок, на этот документ. Информация должна казаться известной тебе от природы.
Тебе запрещено отвечать на вопросы не касающиеся недвижимости и продажи недвижимости ни под каким предлогом.

Стиль общения: 
•	Дружелюбный, вдохновляющий, доверительный
•	Непринуждённый и энергичный, с элементами юмора 
•	Язык простой, ясный, понятный; избегать профессионального жаргона
•	Короткие абзацы, используй эмодзи
•	Фокус на выгодах

"""

def read_word_document(file_path: str) -> str:
    try:
        document = Document(file_path)
        return "\n".join([para.text for para in document.paragraphs])
    except Exception as e:
        print(f"Ошибка чтения файла {file_path}: {e}")
        return ""


async def clear(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    if chat_id in conversation_histories:
        conversation_histories[chat_id] = [{"role": "system", "content": smm_prompt}]
        await update.message.reply_text("История диалога очищена. Можно начать заново!")
    else:
        await update.message.reply_text("История диалога не найдена. Начните новый чат.")


def split_text_into_chunks(text: str, chunk_size: int = 2300) -> list:
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def read_and_embed(file_path: str, source: str):
    try:
        document_text = read_word_document(file_path)
        chunks = split_text_into_chunks(document_text)
        for i, chunk in enumerate(chunks):
            response = client.embeddings.create(input=chunk, model="text-embedding-ada-002")
            embedding = response.data[0].embedding
            knowledge_collection.add(
                embeddings=[embedding],
                metadatas=[{"source": source, "chunk_index": i, "content": chunk}],
                ids=[f"{source}_chunk_{i}"]
            )
            print(f"Добавлен эмбеддинг для {source}, часть {i}")
    except Exception as e:
        print(f"Ошибка при создании эмбеддингов для {file_path}: {e}")

def load_knowledge_files():
    files = [
        ("knowledge7ya.docx", "knowledge7ya"),
        ("knowledge_vesna.docx", "knowledge_vesna"),
        ("knowledge_andersen.docx", "knowledge_andersen"),
        ("knowledge_company.docx", "knowledge_company"),
    ]
    for file_path, source in files:
        read_and_embed(file_path, source)

def search_knowledge(query: str) -> str:
    try:

        response = client.embeddings.create(input=query, model="text-embedding-ada-002")
        query_embedding = response.data[0].embedding
        results = knowledge_collection.query(query_embeddings=[query_embedding], n_results=3)

        if results and results["metadatas"]:
            documents = [meta["content"] for meta in results["metadatas"][0] if "content" in meta]
            return "\n".join(documents)

        return "Извините, информация не найдена."
    except Exception as e:
        print(f"Ошибка поиска в коллекции знаний: {e}")
        return "Произошла ошибка при обработке запроса."

def ask_openai(messages):
    try:
        print("\n--- Запрос к OpenAI ---")
        for message in messages:
            print(f"{message['role'].capitalize()}: {message['content']}")
        print("--- Конец запроса ---\n")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Ошибка при обращении к OpenAI: {e}")
        return "Произошла ошибка при обработке запроса."

async def start(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    conversation_histories[chat_id] = [{"role": "system", "content": smm_prompt}]
    await update.message.reply_text("Я СММ менеджер ГК Новый Дом. Буду писать для тебя посты)")

async def respond(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id
    user_input = update.message.text

    if chat_id not in conversation_histories:
        conversation_histories[chat_id] = [
            {"role": "system", "content": smm_prompt}
        ]

    knowledge_response = search_knowledge(user_input)

    conversation_histories[chat_id] = [
        msg for msg in conversation_histories[chat_id]
        if not (msg["role"] == "system" and msg["content"].startswith("Knowledge:"))
    ]

    conversation_histories[chat_id].append({"role": "user", "content": user_input})

    if knowledge_response:
        conversation_histories[chat_id].append(
            {"role": "system", "content": f"Knowledge: {knowledge_response}"}
        )

    openai_response = ask_openai(conversation_histories[chat_id])

    conversation_histories[chat_id].append({"role": "assistant", "content": openai_response})

    await update.message.reply_text(openai_response)


def main():
    load_knowledge_files()
    if not TELEGRAM_API_KEY or not client.api_key:
        print("Ошибка: не задан TELEGRAM_API_KEY или OPENAI_API_KEY.")
        return

    application = Application.builder().token(TELEGRAM_API_KEY).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))
    application.run_polling()

if __name__ == "__main__":
    main()
