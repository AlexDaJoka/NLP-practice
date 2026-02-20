import re
import pandas as pd
from bs4 import BeautifulSoup
import spacy
from spacy.matcher import Matcher
from typing import Dict, List

df = pd.read_csv("/Users/macbook/Desktop/Project3/hybrid_text_extratcion/data/raw/legal_text_classification.csv")

# ---------------------------
# 1. Preprocess text
# ---------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Remove HTML
    text = BeautifulSoup(text, "html.parser").get_text(" ")

    # Replace emails
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' <EMAIL> ', text)

    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)

    # Replace IPs
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' <IP> ', text)

    # Replace IDs
    text = re.sub(r'\b[A-ZА-Я]{2,}-?\d{3,}\b', ' <ID> ', text)

    # aaaaa -> aaa
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)

    # Remove very long tokens
    text = re.sub(r"\b\w{30,}\b", " ", text)

    # Normalize punctuation
    text = re.sub(r'([!?.,]){2,}', r'\1', text)
    text = re.sub(r"([_\-])\1{2,}", r"\1", text)
    text = re.sub(r"([!?]){2,}", r'\1', text)

    # Remove hashtags
    text = re.sub(r"#(\w+)", r"\1", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ---------------------------
# 2. Initialize spaCy and Matcher
# ---------------------------
nlp = spacy.load("en_core_web_sm")

def create_block_matcher(nlp):
    matcher = Matcher(nlp.vocab)

    # Requisites
    matcher.add("REQUISITES", [
        [{"LOWER": "company"}, {"LOWER": "name"}],
        [{"LOWER": "vat"}],
        [{"LOWER": "tax"}, {"LOWER": "id"}],
        [{"LOWER": "iban"}],
        [{"LOWER": "swift"}],
    ])

    # Payment
    matcher.add("PAYMENT", [
        [{"LOWER": "total"}, {"LOWER": "amount"}],
        [{"LOWER": "amount"}, {"LOWER": "due"}],
        [{"LOWER": "grand"}, {"LOWER": "total"}],
        [{"LOWER": "subtotal"}]
    ])

    # Dates
    matcher.add("DATES", [
        [{"LOWER": "invoice"}, {"LOWER": "date"}],
        [{"LOWER": "due"}, {"LOWER": "date"}]
    ])

    # Signatures
    matcher.add("SIGNATURES", [
        [{"LOWER": "authorized"}, {"LOWER": "signature"}],
        [{"LOWER": "signed"}, {"LOWER": "by"}],
        [{"LOWER": "on"}, {"LOWER": "behalf"}, {"LOWER": "of"}]
    ])

    return matcher

matcher = create_block_matcher(nlp)

# ---------------------------
# 3. Extract blocks with spaCy Matcher
# ---------------------------
def extract_blocks_spacy(text: str) -> dict:
    doc = nlp(text)
    matches = matcher(doc)
    blocks = {"requisites": "", "payment": "", "dates": "", "signatures": ""}

    for match_id, start, end in matches:
        span = doc[start:end].text
        match_name = nlp.vocab.strings[match_id].lower()
        if match_name in blocks:
            blocks[match_name] += span + " "

    # Clean whitespace
    for k in blocks:
        blocks[k] = blocks[k].strip()

    return blocks

# ---------------------------
# 4. Scoring blocks
# ---------------------------
def score_block(block_text: str) -> float:
    return 1.0 if block_text else 0.0

def compute_scores(text: str, blocks: dict) -> dict:
    return {
        "requisites_score": score_block(blocks["requisites"]),
        "payment_score": score_block(blocks["payment"]),
        "dates_score": score_block(blocks["dates"]),
        "signatures_score": score_block(blocks["signatures"])
    }

# ---------------------------
# 5. Router: decide if LLM needed
# ---------------------------
def should_call_llm(scores: dict, threshold: float = 0.5, min_blocks: int = 3) -> bool:
    """
    LLM fallback if less than min_blocks are found
    """
    found_blocks = sum(1 for s in scores.values() if s >= threshold)
    return found_blocks < min_blocks

# ---------------------------
# 6. Extract specific fields
# ---------------------------
def extract_fields_from_blocks(blocks: dict) -> dict:
    fields = {}

    # ---------- Requisites ----------
    requisites_text = blocks.get("requisites", "").lower()
    company_match = re.search(r"company\s*name[:\s]*([A-Za-z0-9 &.,-]+)", requisites_text, re.I)
    vat_match = re.search(r"\bvat[:\s]*([A-Z0-9-]+)", requisites_text, re.I)
    iban_match = re.search(r"\biban[:\s]*([A-Z0-9]+)\b", requisites_text, re.I)

    fields["company_name"] = company_match.group(1).strip() if company_match else None
    fields["tax_id"] = vat_match.group(1).strip() if vat_match else None
    fields["iban"] = iban_match.group(1).strip() if iban_match else None

    # ---------- Payment ----------
    payment_text = blocks.get("payment", "").lower()
    total_match = re.search(r"total\s*amount[:\s]*([\d,\.]+)", payment_text, re.I)
    currency_match = re.search(r"([\$€£])", payment_text)
    fields["total_amount"] = total_match.group(1) if total_match else None
    fields["currency"] = currency_match.group(1) if currency_match else None

    # ---------- Dates ----------
    dates_text = blocks.get("dates", "").lower()
    invoice_match = re.search(r"invoice\s*date[:\s]*([\d/.-]+)", dates_text, re.I)
    due_match = re.search(r"due\s*date[:\s]*([\d/.-]+)", dates_text, re.I)
    fields["invoice_date"] = invoice_match.group(1) if invoice_match else None
    fields["due_date"] = due_match.group(1) if due_match else None

    # ---------- Signatures ----------
    sig_text = blocks.get("signatures", "").lower()
    sig_match = re.search(r"signed\s*by[:\s]*([A-Za-z\s]+)", sig_text, re.I)
    fields["signed_by"] = sig_match.group(1).strip() if sig_match else None

    return fields

# ---------------------------
# 7. Full document processing
# ---------------------------
def process_document(text: str) -> dict:
    clean_text = preprocess_text(text)
    blocks = extract_blocks_spacy(clean_text)
    scores = compute_scores(clean_text, blocks)
    call_llm = should_call_llm(scores)
    fields = extract_fields_from_blocks(blocks)
    return {
        "clean_text": clean_text,
        "blocks": blocks,
        "scores": scores,
        "call_llm": call_llm,
        "fields": fields
    }

# ---------------------------
# 8. Process DataFrame
# ---------------------------
def process_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    results = df[text_col].apply(process_document).apply(pd.Series)
    return pd.concat([df, results], axis=1)

# ---------------------------
# 9. Example usage
# ---------------------------

df_result = process_dataframe(df, "case_text")
df_result.head()