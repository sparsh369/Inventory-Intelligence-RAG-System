import streamlit as st
import pandas as pd
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, Range
)

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Inventory RAG Assistant", layout="wide")

EXCEL_PATH = Path(__file__).parent / "Current_Inventory_.xlsx"  # ✅ FIXED
COLLECTION = "inventory"
VECTOR_DIM = 384

# ---------------- LOADERS ----------------
@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_PATH)
    return df.fillna("")

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant():
    return QdrantClient(":memory:")

# ---------------- TEXT BUILD ----------------
def build_passage(row):
    return " | ".join([f"{k}: {v}" for k, v in row.items() if v])

# ---------------- FILTER ----------------
def build_filter(query: str):
    query = query.lower()
    conditions = []

    if "doh" in query:
        try:
            value = float(query.split()[-1])
            conditions.append(FieldCondition(key="DOH", range=Range(gt=value)))
        except:
            pass

    return Filter(must=conditions) if conditions else None

# ---------------- INDEX ----------------
def build_index(df, embedder, client):
    if client.get_collections().collections:
        return

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )

    texts = [build_passage(row) for _, row in df.iterrows()]
    vectors = embedder.encode(texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload=df.iloc[i].to_dict()
        )
        for i in range(len(df))
    ]

    client.upsert(collection_name=COLLECTION, points=points)

# ---------------- SEARCH (FIXED) ----------------
def search_inventory(query, embedder, client, top_k=5):
    vector = embedder.encode([query])[0].tolist()
    query_filter = build_filter(query)

    results = client.query_points(   # ✅ FIXED API
        collection_name=COLLECTION,
        query=vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    )

    return [
        {"score": round(r.score, 4), **r.payload}
        for r in results.points   # ✅ FIXED
    ]

# ---------------- FORMAT ----------------
def format_context(results):
    context = ""
    for i, r in enumerate(results, 1):
        context += f"\n[Result {i}]\n"
        for k, v in r.items():
            if k != "score" and v:
                context += f"{k}: {v}\n"
    return context

# ---------------- OPENAI ----------------
@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def ask_openai(query, context):
    client = get_openai_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer only using provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# ---------------- UI ----------------
st.title("📦 Inventory RAG Assistant")

top_k = st.sidebar.slider("Top K Results", 3, 15, 5)

# Load
df = load_data()
embedder = load_embedder()
qdrant = get_qdrant()

with st.spinner("Building index..."):
    build_index(df, embedder, qdrant)

st.success(f"Loaded {len(df)} records")

query = st.text_input("Ask your question:")

if query:
    try:
        with st.spinner("Searching..."):
            results = search_inventory(query, embedder, qdrant, top_k)
            context = format_context(results)

        with st.spinner("Generating answer..."):
            answer = ask_openai(query, context)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("🔍 Retrieved Data"):
            st.dataframe(pd.DataFrame(results))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# import streamlit as st
# import pandas as pd
# from pathlib import Path

# from qdrant_client import QdrantClient
# from qdrant_client.models import (
#     Distance, VectorParams, PointStruct,
#     Filter, FieldCondition, MatchValue, Range
# )

# from sentence_transformers import SentenceTransformer
# from openai import OpenAI

# # ---------------- CONFIG ----------------
# st.set_page_config(page_title="Inventory RAG Assistant", layout="wide")

# EXCEL_PATH = Path("Current Inventory .xlsx")
# COLLECTION = "inventory"
# VECTOR_DIM = 384

# # ---------------- LOADERS ----------------
# @st.cache_data
# def load_data():
#     df = pd.read_excel(EXCEL_PATH)
#     return df.fillna("")

# @st.cache_resource
# def load_embedder():
#     return SentenceTransformer("all-MiniLM-L6-v2")

# @st.cache_resource
# def get_qdrant():
#     return QdrantClient(":memory:")  # ✅ required for Streamlit Cloud

# # ---------------- COLUMN HANDLING ----------------
# def get_text_columns(df):
#     return [col for col in df.columns if df[col].dtype == "object"]

# def get_numeric_columns(df):
#     return [col for col in df.columns if df[col].dtype != "object"]

# # ---------------- TEXT BUILD ----------------
# def build_passage(row, text_cols, num_cols):
#     parts = []

#     for col in text_cols:
#         val = str(row.get(col, "")).strip()
#         if val:
#             parts.append(f"{col}: {val}")

#     for col in num_cols:
#         val = row.get(col, None)
#         if pd.notna(val):
#             try:
#                 parts.append(f"{col}: {float(val):,.2f}")
#             except:
#                 pass

#     return " | ".join(parts)

# # ---------------- FILTER ----------------
# def build_filter(query: str):
#     query = query.lower()
#     conditions = []

#     # Example: DOH > 90
#     if "doh" in query and ("greater" in query or ">" in query):
#         try:
#             value = float(query.split()[-1])
#             conditions.append(
#                 FieldCondition(key="DOH", range=Range(gt=value))
#             )
#         except:
#             pass

#     # Example: Plant 2001
#     for word in query.split():
#         if word.isdigit():
#             conditions.append(
#                 FieldCondition(
#                     key="Plant",
#                     match=MatchValue(value=word)
#                 )
#             )

#     return Filter(must=conditions) if conditions else None

# # ---------------- INDEX ----------------
# def collection_exists(client):
#     return COLLECTION in [c.name for c in client.get_collections().collections]

# # @st.cache_resource
# def build_index(df, embedder, client):
#     if collection_exists(client):
#         return

#     client.recreate_collection(
#         collection_name=COLLECTION,
#         vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
#     )

#     text_cols = get_text_columns(df)
#     num_cols = get_numeric_columns(df)

#     BATCH = 256

#     for i in range(0, len(df), BATCH):
#         batch = df.iloc[i:i+BATCH]

#         texts = [
#             build_passage(row, text_cols, num_cols)
#             for _, row in batch.iterrows()
#         ]

#         vectors = embedder.encode(texts)

#         points = [
#             PointStruct(
#                 id=i + j,
#                 vector=vectors[j].tolist(),
#                 payload=batch.iloc[j].to_dict()
#             )
#             for j in range(len(batch))
#         ]

#         client.upsert(collection_name=COLLECTION, points=points)

# # ---------------- SEARCH ----------------
# def search_inventory(query, embedder, client, top_k=5):
#     vector = embedder.encode([query])[0].tolist()
#     query_filter = build_filter(query)

#     results = client.search(
#         collection_name=COLLECTION,
#         query_vector=vector,
#         query_filter=query_filter,
#         limit=top_k,
#         with_payload=True
#     )

#     return [{"score": round(r.score, 4), **r.payload} for r in results]

# # ---------------- FORMAT ----------------
# def format_context(results):
#     context = ""
#     for i, r in enumerate(results, 1):
#         context += f"\n[Result {i}]\n"
#         for k, v in r.items():
#             if k != "score" and v:
#                 context += f"{k}: {v}\n"
#     return context

# # ---------------- OPENAI ----------------
# @st.cache_resource
# def get_openai_client():
#     return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ✅ secure

# def ask_openai(query, context):
#     client = get_openai_client()

#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an expert inventory analyst. Answer ONLY from provided context."
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

# # Sidebar
# st.sidebar.header("Settings")
# top_k = st.sidebar.slider("Top K Results", 3, 15, 5)

# # Load everything
# df = load_data()
# embedder = load_embedder()
# qdrant = get_qdrant()

# build_index(df, embedder, qdrant)

# st.success(f"Loaded {len(df)} inventory records")

# # Chat
# query = st.text_input("Ask your question:")

# if query:
#     with st.spinner("Searching..."):
#         results = search_inventory(query, embedder, qdrant, top_k)
#         context = format_context(results)

#     with st.spinner("Generating answer..."):
#         answer = ask_openai(query, context)

#     st.subheader("Answer")
#     st.write(answer)

#     with st.expander("🔍 Retrieved Data"):
#         st.dataframe(pd.DataFrame(results))
