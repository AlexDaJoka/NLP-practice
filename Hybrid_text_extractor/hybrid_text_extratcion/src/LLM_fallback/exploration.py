import pandas as pd
from bs4 import BeautifulSoup
import re
from openai import OpenAI
import json


qwen_client = OpenAI(base_url="http://your-qwen-host", api_key="ignored")


# ---------------------------
# 4. LLM fallback на Qwen
# ---------------------------
def llm_fallback_qwen(text, blocks):
    prompt = f"""
Extract the following fields as strict JSON:
company_name, tax_id, iban, total_amount, currency, invoice_date, due_date, signed_by

Document text:
{text}

Detected blocks:
{blocks}

Return ONLY JSON.
"""
    response = qwen_client.chat.completions.create(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    content = response.choices[0].message.content
    try:
        return json.loads(content)
    except:
        # Если LLM упал, возвращаем None
        return {f: None for f in ["company_name","tax_id","iban","total_amount","currency","invoice_date","due_date","signed_by"]}

# ---------------------------
# 5. Объединение rule + LLM
# ---------------------------
def merge_fields(rule_fields, llm_fields):
    merged = {}
    for k in llm_fields.keys():
        merged[k] = rule_fields.get(k) or llm_fields.get(k)
    return merged


# ---------------------------
# 7. Пример работы на датасете
# ---------------------------
df = pd.read_excel("path/to/your/file.xlsx")

# Предположим, текст документа в колонке 'case_text'
df["final_fields"] = df["case_text"].apply(process_document)

# Результат
print(df[["case_text","final_fields"]].head())
