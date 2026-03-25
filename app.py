import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inventory RAG Assistant",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports with lazy install guard ────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct, Filter,
        FieldCondition, MatchValue, Range
    )
    from sentence_transformers import SentenceTransformer
    import anthropic
except ImportError as e:
    st.error(f"Missing dependency: {e}. Run `pip install -r requirements.txt`")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
EXCEL_PATH   = Path(__file__).parent / "Current_Inventory_.xlsx"
COLLECTION   = "inventory"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, free, fast
VECTOR_DIM   = 384
TOP_K        = 8

# Columns used to build the text passage for embedding
TEXT_COLS = [
    "Material", "Material Name", "Material Type", "Plant",
    "UOM", "Product Category", "Material Group",
    "Material Application", "Sub Application",
    "Product Family ", "SOP Family", "Product Group",
    "MRP Controller Text", "Purchasing Group Text", "ABC",
]
# Numeric columns stored as payload for filtering
NUM_COLS = ["Shelf Stock", "Shelf Stock ($)", "GIT", "GIT ($)",
            "WIP", "WIP($)", "DOH", "Safety Stock", "Demand"]

# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner="Connecting to Qdrant…")
def get_qdrant():
    return QdrantClient(":memory:")   # in-memory; swap for host= for persistent

@st.cache_data(show_spinner="Reading inventory file…")
def load_dataframe():
    df = pd.read_excel(EXCEL_PATH, dtype=str)
    # Coerce numeric columns back to float
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna("")
    return df

def build_passage(row: pd.Series) -> str:
    """Convert a DataFrame row into a descriptive text passage for embedding."""
    parts = []
    for col in TEXT_COLS:
        val = str(row.get(col, "")).strip()
        if val and val != "nan":
            parts.append(f"{col.strip()}: {val}")
    # Add key numeric fields as human-readable text
    for col in ["Shelf Stock", "Shelf Stock ($)", "DOH", "Safety Stock", "Demand"]:
        val = row.get(col, "")
        if val != "" and str(val) != "nan":
            try:
                parts.append(f"{col}: {float(val):,.2f}")
            except Exception:
                pass
    return " | ".join(parts)

def build_payload(row: pd.Series, idx: int) -> dict:
    payload = {"_row_index": idx}
    for col in TEXT_COLS:
        val = str(row.get(col, "")).strip()
        payload[col.strip()] = val if val and val != "nan" else None
    for col in NUM_COLS:
        val = row.get(col, None)
        try:
            payload[col] = float(val) if val != "" and str(val) != "nan" else None
        except Exception:
            payload[col] = None
    return payload

@st.cache_resource(show_spinner="Building vector index (first run may take a few minutes)…")
def build_index(_df, _embedder, _client):
    """Embed all rows and upsert into Qdrant. Cached so it only runs once."""
    client   = _client
    embedder = _embedder
    df       = _df

    # Create collection
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    BATCH = 512
    total = len(df)
    progress = st.progress(0, text="Indexing inventory rows…")

    for start in range(0, total, BATCH):
        end   = min(start + BATCH, total)
        batch = df.iloc[start:end]

        passages = [build_passage(row) for _, row in batch.iterrows()]
        vectors  = embedder.encode(passages, batch_size=64, show_progress_bar=False)

        points = [
            PointStruct(
                id=int(start + i),
                vector=vectors[i].tolist(),
                payload=build_payload(row, start + i),
            )
            for i, (_, row) in enumerate(batch.iterrows())
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        progress.progress(end / total, text=f"Indexed {end:,} / {total:,} rows…")

    progress.empty()
    return True

def search_inventory(query: str, embedder, client, top_k=TOP_K) -> list[dict]:
    vec = embedder.encode([query])[0].tolist()
    results = client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {"score": round(r.score, 4), **r.payload}
        for r in results
    ]

def format_context(hits: list[dict]) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[Row {i}]")
        for k, v in h.items():
            if k.startswith("_") or v is None or v == "":
                continue
            lines.append(f"  {k}: {v}")
        lines.append("")
    return "\n".join(lines)

def ask_claude(question: str, context: str, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    system = (
        "You are an expert inventory analyst assistant. "
        "You have been given retrieved inventory records as context. "
        "Answer the user's question accurately and concisely using only the provided context. "
        "If the answer is not in the context, say so clearly. "
        "Format numbers with commas and 2 decimal places where appropriate. "
        "When listing items, use bullet points."
    )
    user_msg = f"""Context (retrieved inventory records):
{context}

Question: {question}

Answer based on the context above:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/box.png", width=64)
    st.title("⚙️ Configuration")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-…",
        help="Get your key from console.anthropic.com",
    )

    st.divider()
    st.markdown("**About this dataset**")
    st.caption(f"📄 File: `Current_Inventory_.xlsx`")

    if EXCEL_PATH.exists():
        st.success("✅ Inventory file found")
    else:
        st.error("❌ Inventory file not found")
        st.stop()

    top_k = st.slider("Top-K results to retrieve", 3, 20, TOP_K)
    show_raw = st.checkbox("Show retrieved rows", value=True)

    st.divider()
    st.markdown("**Sample questions**")
    sample_qs = [
        "Which materials have DOH greater than 90?",
        "Show me all Raw materials with high shelf stock",
        "What are the top materials by demand?",
        "Find fiber related materials in plant 2001",
        "Which items have no safety stock?",
        "List semifinished products managed by Alex Bernstein",
    ]
    for q in sample_qs:
        if st.button(q, use_container_width=True, key=q):
            st.session_state["prefill"] = q

# ── Initialize resources ────────────────────────────────────────────────────────
df       = load_dataframe()
embedder = load_embedder()
qdrant   = get_qdrant()
indexed  = build_index(df, embedder, qdrant)

# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("📦 Inventory RAG Assistant")
st.caption(f"Semantic search over **{len(df):,} inventory records** · Powered by Qdrant + Claude")

# Dashboard metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    plants = df["Plant"].nunique() if "Plant" in df.columns else "—"
    st.metric("Plants", plants)
with col3:
    mat_types = df["Material Type"].nunique() if "Material Type" in df.columns else "—"
    st.metric("Material Types", mat_types)
with col4:
    total_val = pd.to_numeric(df["Shelf Stock ($)"], errors="coerce").sum()
    st.metric("Total Shelf Value ($)", f"{total_val:,.0f}")

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits") and show_raw:
            with st.expander("🔍 Retrieved inventory rows", expanded=False):
                hits_df = pd.DataFrame(msg["hits"]).drop(columns=["_row_index"], errors="ignore")
                st.dataframe(hits_df, use_container_width=True)

# Chat input
prefill = st.session_state.pop("prefill", "")
query = st.chat_input("Ask anything about the inventory…") or prefill

if query:
    if not api_key:
        st.warning("⚠️ Please enter your Anthropic API key in the sidebar.")
        st.stop()

    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching inventory…"):
            hits = search_inventory(query, embedder, qdrant, top_k=top_k)
            context = format_context(hits)

        with st.spinner("Generating answer…"):
            answer = ask_claude(query, context, api_key)

        st.markdown(answer)

        if show_raw and hits:
            with st.expander("🔍 Retrieved inventory rows", expanded=False):
                hits_df = pd.DataFrame(hits).drop(columns=["_row_index"], errors="ignore")
                st.dataframe(hits_df, use_container_width=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "hits": hits,
    })
# import streamlit as st
# import pandas as pd
# from pathlib import Path

# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance, VectorParams, PointStruct,
#     Filter, FieldCondition, Range
# )

# from openai import OpenAI

# # ---------------- CONFIG ----------------
# st.set_page_config(page_title="Inventory RAG Assistant", layout="wide")

# EXCEL_PATH = Path(__file__).parent / "Current_Inventory_.xlsx"
# COLLECTION = "inventory"

# # ---------------- LOAD DATA ----------------
# @st.cache_data
# def load_data():
#     df = pd.read_excel(EXCEL_PATH)
#     return df.fillna("")

# # ---------------- OPENAI ----------------
# @st.cache_resource
# def get_openai_client():
#     return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# def get_embedding(text):
#     client = get_openai_client()
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         input=text
#     )
#     return response.data[0].embedding

# # ---------------- QDRANT ----------------
# @st.cache_resource
# def get_qdrant():
#     return QdrantClient(":memory:")

# # ---------------- DOCUMENT BUILD ----------------
# def build_passage(row):
#     return f"""
#     Material {row.get('Material Name', '')} is a {row.get('Material Type', '')}
#     located in plant {row.get('Plant', '')}, under category {row.get('Product Category', '')}.

#     It currently has stock of {row.get('Shelf Stock', '')} units,
#     demand of {row.get('Demand', '')} units,
#     and safety stock of {row.get('Safety Stock', '')} units.

#     Additional details:
#     MRP Controller: {row.get('MRP Controller Text', '')}
#     Purchasing Group: {row.get('Purchasing Group Text', '')}
#     """

# # ---------------- FILTER ----------------
# def build_filter(query: str):
#     query = query.lower()
#     conditions = []

#     if "doh" in query:
#         try:
#             value = float(query.split()[-1])
#             conditions.append(FieldCondition(key="DOH", range=Range(gt=value)))
#         except:
#             pass

#     return Filter(must=conditions) if conditions else None

# # ---------------- INDEX ----------------
# def collection_exists(client):
#     return COLLECTION in [c.name for c in client.get_collections().collections]

# def build_index(df, client):
#     client.recreate_collection(
#         collection_name=COLLECTION,
#         vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
#     )

#     # client.recreate_collection(
#     #     collection_name=COLLECTION,
#     #     vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
#     # )

#     texts = [build_passage(row) for _, row in df.iterrows()]
#     vectors = [get_embedding(text) for text in texts]

#     points = [
#         PointStruct(
#             id=i,
#             vector=vectors[i],
#             payload=df.iloc[i].to_dict()
#         )
#         for i in range(len(df))
#     ]

#     client.upsert(collection_name=COLLECTION, points=points)

# # ---------------- SEARCH ----------------
# def search_inventory(query, client, top_k=5):
#     vector = get_embedding(query)
#     query_filter = build_filter(query)

#     results = client.query_points(
#         collection_name=COLLECTION,
#         query=vector,
#         limit=top_k,
#         with_payload=True,
#         query_filter=query_filter
#     )

#     return [
#         {"score": round(r.score, 4), **r.payload}
#         for r in results.points
#     ]

# # ---------------- FORMAT ----------------
# def format_context(results):
#     context = ""
#     for i, r in enumerate(results, 1):
#         context += f"\n[Result {i}]\n"
#         for k, v in r.items():
#             if k != "score" and v:
#                 context += f"{k}: {v}\n"
#     return context

# # ---------------- LLM ----------------
# def ask_openai(query, context):
#     client = get_openai_client()

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an inventory analyst. Answer only from the given context."
#             },
#             {
#                 "role": "user",
#                 "content": f"Context:\n{context}\n\nQuestion: {query}"
#             }
#         ],
#         temperature=0.3
#     )

#     return response.choices[0].message.content

# # ---------------- UI ----------------
# st.title("📦 Inventory RAG Assistant")

# top_k = st.sidebar.slider("Top K Results", 3, 20, 8)

# df = load_data()
# qdrant = get_qdrant()

# with st.spinner("🔄 Building vector index..."):
#     build_index(df, qdrant)

# st.success(f"Loaded {len(df)} inventory records")

# query = st.text_input("Ask your question:")

# if query:
#     try:
#         with st.spinner("🔍 Retrieving relevant data..."):
#             results = search_inventory(query, qdrant, top_k)
#             context = format_context(results)

#         with st.spinner("🤖 Generating answer..."):
#             answer = ask_openai(query, context)

#         st.subheader("Answer")
#         st.write(answer)

#         with st.expander("🔍 Retrieved Data"):
#             st.dataframe(pd.DataFrame(results))

#     except Exception as e:
#         st.error(f"Error: {str(e)}")

