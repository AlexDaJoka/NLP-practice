from elasticsearch import Elasticsearch

es = Elasticsearch("http://your-es-host:9200")

# Создаём индекс
index_name = "rag_reviews"

mapping = {
    "mappings": {
        "properties": {
            "review": {"type": "text"},
            "hours_played": {"type": "float"},
            "helpful": {"type": "float"},
            "recommendation": {"type": "keyword"},
            "date": {"type": "date"},
            "vector": {"type": "dense_vector", "dims": 384, "index": True, "similarity": "cosine"},
        }
    }
}

es.options(ignore_status=400).indices.create(index=index_name, body=mapping)