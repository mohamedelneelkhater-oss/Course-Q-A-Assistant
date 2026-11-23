import gc
import os
import time

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = "/kaggle/working"
DB_DIR = f"{BASE_DIR}/chroma_db"
APP_DIR = f"{BASE_DIR}/rag_app"
SAVED_LLM_DIR = f"{BASE_DIR}/best_llm"
CONFIG_PATH = f"{APP_DIR}/best_model.json"
COLLECTION = "course_qa"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK = "I don't know based on the course materials."
MIN_SCORE = 0.25

LLM_REGISTRY = {
    "flan-t5-small": {"repo": "google/flan-t5-small", "type": "seq2seq"},
    "flan-t5-base": {"repo": "google/flan-t5-base", "type": "seq2seq"},
    "qwen2.5-0.5b-instruct": {"repo": "Qwen/Qwen2.5-0.5B-Instruct", "type": "causal"},
}

SYSTEM_PROMPT = (
    "You are a course assistant. Answer the question using ONLY the context. "
    "Be concise and answer in one or two complete sentences. "
    f"If the answer is not in the context, reply exactly: {FALLBACK}"
)

_EMBEDDER = None
_LLMS = {}


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _EMBEDDER


def get_collection(reset=False):
    client = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def retrieve(query, k=4, course=None):
    vec = get_embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"course": course} if course and course != "All" else None
    res = get_collection().query(query_embeddings=vec, n_results=k, where=where)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": text, "meta": meta, "score": 1 - dist})
    return hits


def load_llm(name):
    if name in _LLMS:
        return _LLMS[name]
    cfg = LLM_REGISTRY[name]
    path = cfg["repo"]
    marker = f"{SAVED_LLM_DIR}/model_name.txt"
    if os.path.exists(marker) and open(marker).read().strip() == name:
        path = SAVED_LLM_DIR
    tok = AutoTokenizer.from_pretrained(path)
    model_cls = AutoModelForSeq2SeqLM if cfg["type"] == "seq2seq" else AutoModelForCausalLM
    model = model_cls.from_pretrained(path, torch_dtype=torch.float32).eval()
    _LLMS[name] = (tok, model)
    return _LLMS[name]


def unload_llm(name):
    _LLMS.pop(name, None)
    gc.collect()


def build_prompt(question, hits, name, tok):
    context = "\n\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))[:3000]
    if LLM_REGISTRY[name]["type"] == "seq2seq":
        return f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(name, prompt, max_new_tokens=120):
    tok, model = load_llm(name)
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1536)
    with torch.no_grad():
        if LLM_REGISTRY[name]["type"] == "seq2seq":
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=2, do_sample=False)
            return tok.decode(out[0], skip_special_tokens=True).strip()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def answer_question(question, model_name, k=4, course=None):
    start = time.time()
    hits = retrieve(question, k=k, course=course)
    if not hits or hits[0]["score"] < MIN_SCORE:
        return {"answer": FALLBACK, "sources": [], "hits": hits, "latency": time.time() - start}
    tok, _ = load_llm(model_name)
    answer = generate(model_name, build_prompt(question, hits, model_name, tok))
    sources, seen = [], set()
    for h in hits:
        m = h["meta"]
        label = f"{m['file']} (course: {m['course']}, {m['type']}, part {m['page']})"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return {"answer": answer or FALLBACK, "sources": sources, "hits": hits, "latency": time.time() - start}
