import streamlit as st
import pandas as pd
import os
import json
import hashlib
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, SearchRequest
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inventory RAG", page_icon="📦", layout="wide")

# ── Sidebar: API keys & settings ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    openai_api_key = st.text_input("OpenAI API Key", type="password",
                                   placeholder="sk-...")
    excel_file = st.file_uploader("Upload Inventory Excel", type=["xlsx", "xls"])
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "This app uses **OpenAI embeddings** + **Qdrant** (in-memory) "
        "to answer questions about your inventory data."
    )

COLLECTION = "inventory"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
EMBED_DIM   = 1536
CHUNK_SIZE  = 50          # rows per chunk

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_qdrant():
    return QdrantClient(":memory:")


def row_to_text(row: dict) -> str:
    """Convert a DataFrame row dict to a readable text chunk."""
    parts = []
    for k, v in row.items():
        if pd.notna(v) and v != "":
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


def chunk_dataframe(df: pd.DataFrame, size: int = CHUNK_SIZE):
    """Yield (chunk_id, text) for overlapping row windows."""
    texts, metas = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        text = row_to_text(row.to_dict())
        texts.append(text)
        metas.append({"row_index": i, "material": str(row.get("Material", "")),
                       "plant": str(row.get("Plant", "")),
                       "material_name": str(row.get("Material Name", ""))})
    # group into chunks of `size` rows
    chunks, chunk_metas = [], []
    for start in range(0, len(texts), size):
        chunk_text = "\n".join(texts[start:start + size])
        chunks.append(chunk_text)
        chunk_metas.append({"start_row": start, "end_row": start + len(texts[start:start+size]) - 1,
                             "sample_material": metas[start]["material"]})
    return chunks, chunk_metas


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([r.embedding for r in resp.data])
    return all_embeddings


def build_index(df: pd.DataFrame, openai_client: OpenAI, qdrant: QdrantClient,
                progress_bar, status_text):
    status_text.text("Chunking data...")
    chunks, metas = chunk_dataframe(df)

    # (Re)create collection
    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

    status_text.text(f"Embedding {len(chunks)} chunks…")
    embeddings = get_embeddings(openai_client, chunks)

    status_text.text("Uploading to Qdrant…")
    points = [
        PointStruct(id=i, vector=embeddings[i],
                    payload={"text": chunks[i], **metas[i]})
        for i in range(len(chunks))
    ]
    qdrant.upsert(collection_name=COLLECTION, points=points)
    progress_bar.progress(1.0)
    status_text.text(f"✅ Index built — {len(chunks)} chunks, {len(df)} rows")


def retrieve(query: str, openai_client: OpenAI, qdrant: QdrantClient, top_k: int = 6):
    q_embed = openai_client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    results = qdrant.search(collection_name=COLLECTION, query_vector=q_embed, limit=top_k)
    return [r.payload["text"] for r in results]


def answer(query: str, context_chunks: list[str], openai_client: OpenAI) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    system = (
        "You are an expert inventory analyst. "
        "Answer the user's question strictly based on the inventory data provided in the context. "
        "If the answer isn't in the context, say so. "
        "Be concise but thorough. Use bullet points or tables when helpful."
    )
    user_msg = f"Context (inventory records):\n{context}\n\nQuestion: {query}"
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user_msg}],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📦 Inventory RAG Assistant")
st.markdown("Ask anything about your inventory — materials, stock levels, plants, suppliers, and more.")

if not openai_api_key:
    st.info("👈 Enter your **OpenAI API Key** in the sidebar to get started.")
    st.stop()

if not excel_file:
    st.info("👈 Upload your **Inventory Excel** file in the sidebar.")
    st.stop()

openai_client = OpenAI(api_key=openai_api_key)
qdrant = get_qdrant()

# ── Load & index data ─────────────────────────────────────────────────────────
file_hash = hashlib.md5(excel_file.read()).hexdigest()
excel_file.seek(0)

if st.session_state.get("indexed_hash") != file_hash:
    with st.spinner("Loading Excel file…"):
        df = pd.read_excel(excel_file, sheet_name=0)
        df = df.fillna("")
        st.session_state["df"] = df
        st.session_state["indexed_hash"] = None  # reset

    st.success(f"Loaded **{len(df):,} rows × {len(df.columns)} columns**")
    st.dataframe(df.head(5), use_container_width=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        prog = st.progress(0)
    with col2:
        status = st.empty()

    build_index(df, openai_client, qdrant, prog, status)
    st.session_state["indexed_hash"] = file_hash
else:
    df = st.session_state["df"]
    st.success(f"✅ Index ready — **{len(df):,} rows** loaded. Ask your question below.")

# ── Chat interface ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Example prompts
st.markdown("#### 💡 Example questions")
example_cols = st.columns(3)
examples = [
    "Which materials have zero shelf stock?",
    "Show top 10 materials by shelf stock value",
    "List all plants and their material counts",
    "What materials belong to product family X?",
    "Which materials have safety stock > demand?",
    "Summarize WIP by product category",
]
for i, ex in enumerate(examples):
    if example_cols[i % 3].button(ex, key=f"ex_{i}"):
        st.session_state["messages"].append({"role": "user", "content": ex})

st.markdown("---")

# Chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about your inventory…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching & answering…"):
            chunks = retrieve(prompt, openai_client, qdrant)
            response = answer(prompt, chunks, openai_client)
        st.markdown(response)
        st.session_state["messages"].append({"role": "assistant", "content": response})
