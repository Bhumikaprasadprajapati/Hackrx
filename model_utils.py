from llama_cpp import Llama
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

# -------------------------------
# Load LLaMA model
# -------------------------------
MODEL_PATH = r"C:\Users\HONER\Downloads\llama-2-7b.Q2_K.gguf"
llm = Llama(model_path=MODEL_PATH)

# Detect model's context length (n_ctx) safely
MODEL_MAX_CONTEXT = llm.n_ctx() if callable(llm.n_ctx) else llm.n_ctx

# -------------------------------
# Load spaCy model
# -------------------------------
nlp = spacy.load("en_core_web_md")


# -------------------------------
# Load embedded chunks from JSON
# -------------------------------
def load_embedded_chunks(json_path="embedded_chunks.json"):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Please place it in the project folder.")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


# -------------------------------
# Find top-k relevant chunks
# -------------------------------
def find_relevant_chunks(query, embedded_chunks, top_k=5):
    query_vector = nlp(query).vector.reshape(1, -1)
    chunk_vectors = np.array([chunk['embedding'] for chunk in embedded_chunks])
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        chunk = embedded_chunks[idx]
        results.append({
            "pdf_name": chunk["pdf_name"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "similarity": round(similarities[idx], 3),
            "text": chunk["text"]
        })
    return results


# -------------------------------
# Build prompt with token safety
# -------------------------------
def build_prompt(query, relevant_chunks, max_context_tokens=None):
    if max_context_tokens is None:
        # Reserve ~80% of context for prompt
        max_context_tokens = int(MODEL_MAX_CONTEXT * 0.8)

    context = ""
    for chunk in relevant_chunks:
        candidate = context + "\n\n" + chunk["text"]
        token_count = len(llm.tokenize(candidate.encode("utf-8")))
        if token_count > max_context_tokens:
            break
        context = candidate

    prompt = f"""You are an expert assistant for policy documents.
Answer clearly and directly based on the provided context.
If possible, start with "Yes" or "No" and then give a short reason.
If the answer is not found, say "The policy document does not provide this information."

Context:
{context}

Question:
{query}

Answer:"""
    return prompt


# -------------------------------
# Generate answer
# -------------------------------
def generate_answer_llama_cpp(prompt):
    prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
    available_tokens = MODEL_MAX_CONTEXT - prompt_tokens

    if available_tokens < 10:
        available_tokens = 10  # minimum safety

    response = llm(
        prompt=prompt,
        max_tokens=available_tokens,
        temperature=0.3,  # factual
        top_p=0.9,
        stop=["\n\n"]
    )
    return response['choices'][0]['text'].strip()