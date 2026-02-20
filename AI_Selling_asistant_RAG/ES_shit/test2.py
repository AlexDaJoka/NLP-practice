from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

embedding = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
es = Elasticsearch("http://your-es-host:9200")

# Пример документов
documents = [
    {"review": "This product is great!", "embedding": embedding.embed_query("This product is great!"), "date": "2026-01-01"},
    {"review": "Not satisfied", "embedding": embedding.embed_query("Not satisfied"), "date": "2026-01-02"},
    # ... добавьте все остальные документы
]

# Добавляем документы через Bulk API
bulk_data = []
for i, doc in enumerate(documents):
    bulk_data.append({"index": {"_index": "rag_reviews", "_id": i}})
    bulk_data.append(doc)

from elasticsearch.helpers import bulk
bulk(es, bulk_data)
