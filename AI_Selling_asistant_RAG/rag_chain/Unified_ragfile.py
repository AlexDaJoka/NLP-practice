import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from sentence_transformers import CrossEncoder
import redis
import pickle
import hashlib
from celery import Celery

# ================================
# CELERY CONFIG
# ================================
app = Celery(
    "rag_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1"
)

# ================================
# REDIS CACHE
# ================================
cache = redis.Redis(host='localhost', port=6379, db=2)

def make_cache_key(query: str, filters: dict, prompt_version: str):
    key_str = f"{query}-{filters}-{prompt_version}"
    return hashlib.sha256(key_str.encode()).hexdigest()




# ================================
# LLM
# ================================
hf_token = "Your HuggingFace token"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.float16,
    use_auth_token=hf_token
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ================================
# EMBEDDINGS
# ================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ================================
# ELASTICSEARCH VECTOR STORE
# ================================
vectorstore = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="rag_reviews",
    embedding=embeddings,
    query_field="review"
)

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)



# ================================
# PROMPT
# ================================
prompt = ChatPromptTemplate.from_template("""
Ты аналитический AI-ассистент.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Если данных недостаточно — прямо укажи это.

Контекст (выдержки из отзывов):
{context}

Статистика по всей выборке:
- всего отзывов:
- рекомендовали: %
- не рекомендовали: %

Вопрос:
{question}

Формат ответа СТРОГО такой:

1. Общее мнение пользователей (2–3 предложения, ТОЛЬКО по контексту)
2. Доля рекомендаций:
   - рекомендовали: X%
   - не рекомендовали: Y%
3. Самые частотные темы отзывов (короткие bullet points)
4. Основная проблема игры (1–2 предложения)

Запрещено:
- добавлять советы или рекомендации
- делать выводы, не подтверждённые контекстом
- использовать общие фразы без конкретных наблюдений
""")


# ================================
# RAG CHAIN
# ================================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # "stuff" объединяет все retrieved docs в один prompt
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# ================================
# ASYNC CELERY TASK
# ================================
@app.task(bind=True)
def generate_rag_answer(self, query: str, filters: dict = {}, prompt_version: str = "v1.0"):
    """
    Асинхронная задача RAG с семантическим кэшем и reranking.
    """

    # -------------------------------
    # 1️⃣ Проверяем кэш
    # -------------------------------
    cache_key = make_cache_key(query, filters, prompt_version)
    cached_result = cache.get(cache_key)
    if cached_result:
        return pickle.loads(cached_result)

    # -------------------------------
    # 2️⃣ Retrieval + reranking
    # -------------------------------
    docs = retriever.get_relevant_documents(query)
    docs = rerank_docs(query, docs, top_k=5)

    # -------------------------------
    # 3️⃣ Dedup & truncate (по необходимости)
    # -------------------------------
    # Можно обрезать контекст до max_tokens_context
    # И убрать дублирующиеся чанки (если есть)
    # Для простоты здесь не добавляем сложный токен-триммер

    # -------------------------------
    # 4️⃣ Генерация LLM
    # -------------------------------
    result = qa_chain.invoke({"query": query})

    # -------------------------------
    # 5️⃣ Кэшируем результат
    # -------------------------------
    cache.set(cache_key, pickle.dumps(result), ex=3600)

    return result

# ================================
# EXAMPLE USAGE
# ================================
if __name__ == "__main__":
    query = "Сделай анализ по игре Warhammer 40000"
    filters = {"category": "reviews"}
    task = generate_rag_answer.delay(query, filters, "v1.0")
    print(f"Task ID: {task.id}")

    # Получение результата (blocking, можно сделать через webhook/polling)
    result = qa_chain.invoke({"query": "Сделай анализ по игре Warhammer 40000"})
    print(result["result"])