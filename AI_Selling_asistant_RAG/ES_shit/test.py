import numpy as np
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

# ================================
# ИНИЦИАЛИЗАЦИЯ
# ================================
# Подключение к Elasticsearch
es = Elasticsearch("http://localhost:9200")
index_name = "rag_reviews"

# Модель для генерации embedding запроса
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


query = "Как вернуть деньги за игру?"  # пример запроса
query_emb = model.encode(query, normalize_embeddings=True).tolist()

search_body = {
    "size": 5,  # топ-5 ближайших документов
    "query": {
        "knn": {
            "vector": {
                "vector": query_emb,
                "k": 5
            }
        }
    }
}

res = es.search(
    index=index_name,
    knn={
        "field": "vector",
        "query_vector": query_emb,
        "k": 5,
        "num_candidates": 50,
    },
    _source=["review", "hours_played", "recommendation"]
)


for hit in res["hits"]["hits"]:
    print("SCORE:", hit["_score"])
    print("REVIEW:", hit["_source"]["review"])
    print("HOURS PLAYED:", hit["_source"]["hours_played"])
    print("RECOMMENDATION:", hit["_source"]["recommendation"])