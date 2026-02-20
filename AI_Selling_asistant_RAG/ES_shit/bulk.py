import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch, helpers
from datetime import datetime

# ================================
# ПОДКЛЮЧЕНИЕ К ES
# ================================
es = Elasticsearch("http://your-es-host:9200")
index_name = "rag_reviews"

# ================================
# ЗАГРУЗКА ДАННЫХ
# ================================
df = pd.read_parquet("path/to/your/file.parquet")

# ================================
# ФУНКЦИИ ДЛЯ БЕЗОПАСНОЙ ПОДГОТОВКИ ДАННЫХ
# ================================
def safe_float(x):
    try:
        f = float(x)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except:
        return None

def safe_date(x):
    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.isna(dt):
            return None
        return dt.isoformat()
    except:
        return None

def safe_embedding(x):
    try:
        emb = np.array(x, dtype=np.float32)
        if emb.shape[0] != 384:
            return None
        if np.isnan(emb).any() or np.isinf(emb).any():
            return None
        return emb.tolist()
    except:
        return None

# ================================
# ПОДГОТОВКА ДОКУМЕНТОВ
# ================================
docs = []

for i, row in df.iterrows():
    emb_list = safe_embedding(row["embeddings"])
    if not emb_list:
        print(f"Skipping {i}: bad embedding")
        continue

    doc = {
        "_index": index_name,
        "_id": i,
        "_source": {
            "review": str(row["review"]) if pd.notna(row["review"]) else "",
            "hours_played": safe_float(row.get("hours_played", None)),
            "helpful": safe_float(row.get("helpful", None)),
            "recommendation": str(row["recommendation"]) if pd.notna(row["recommendation"]) else None,
            "date": safe_date(row.get("date", None)),
            "vector": emb_list
        }
    }
    docs.append(doc)

print(f"Prepared {len(docs)} documents for bulk upload")

# ================================
# BULK ЗАГРУЗКА ПО ЧАСТЯМ
# ================================
batch_size = 200
errors = []

for i in range(0, len(docs), batch_size):
    chunk = docs[i:i+batch_size]
    try:
        success, failed = helpers.bulk(es, chunk, raise_on_error=False)
        if failed:
            errors.extend(failed)
    except Exception as e:
        print(f"Bulk chunk {i}-{i+batch_size} failed: {e}")

print(f"Bulk finished. Total docs prepared: {len(docs)}")
print(f"Total errors: {len(errors)}")
if errors:
    print(errors[:10])  # печатаем первые 10 ошибок
else:
    print("No errors! All documents indexed successfully.")

# ================================
# ПРОВЕРКА КОЛИЧЕСТВА ДОКУМЕНТОВ В ИНДЕКСЕ
# ================================
count_res = es.count(index=index_name)
print("Total documents in ES index:", count_res['count'])
