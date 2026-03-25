import streamlit as st
import pandas as pd
import hashlib
import traceback
from typing import List
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── CONFIG ─────────────────────────────────────────────
st.set_page_config(page_title="Inventory RAG", page_icon="📦", layout="wide")

COLLECTION = "Current Inventory .xlsx"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
EMBED_DIM = 1536
CHUNK_SIZE = 10
BATCH_SIZE = 64

# 📁 DEFAULT FILE PATH (put your file here)
DEFAULT_FILE = "inventory.xlsx"

# ── LOAD API KEY FROM SECRETS ──────────────────────────
openai_api_key = st.secrets["OPENAI_API_KEY"]

# ── INIT CLIENTS ───────────────────────────────────────
@st.cache_resource
def get_qdrant():
    return QdrantClient(":memory:")

def get_openai_client(api_key):
    return OpenAI(api_key=api_key)

# ── HELPERS ────────────────────────────────────────────
def row_to_text(row: dict) -> str:
    parts = []
    for k, v in row.items():
        if pd.notna(v) and str(v).strip():
            parts.append(f"{k}: {v}")
    return " | ".join(parts)

def chunk_dataframe(df: pd.DataFrame):
    texts, metas = [], []

    for i, (_, row) in enumerate(df.iterrows()):
        text = row_to_text(row.to_dict())
        texts.append(text)

        metas.append({
            "row": i,
            "material": str(row.get("Material", "")),
            "plant": str(row.get("Plant", "")),
        })

    chunks, chunk_metas = [], []

    for i in range(0, len(texts), CHUNK_SIZE):
        chunk = "\n".join(texts[i:i + CHUNK_SIZE])[:8000]
        if chunk.strip():
            chunks.append(chunk)
            chunk_metas.append({
                "start": i,
                "end": i + CHUNK_SIZE
            })

    return chunks, chunk_metas

# ── EMBEDDINGS ─────────────────────────────────────────
def get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch = [t[:8000] for t in batch if t.strip()]

        if not batch:
            continue

        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch
            )
            all_embeddings.extend([r.embedding for r in resp.data])

        except Exception as e:
            st.error("❌ Embedding Error")
            st.code(str(e))
            st.code(traceback.format_exc())
            raise

    return all_embeddings

# ── BUILD INDEX ────────────────────────────────────────
def build_index(df, client, qdrant, prog, status):
    status.text("🔹 Chunking data...")
    chunks, metas = chunk_dataframe(df)

    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)

    qdrant.create_collection(
        COLLECTION,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE
        )
    )

    status.text(f"🔹 Embedding {len(chunks)} chunks...")
    embeddings = get_embeddings(client, chunks)

    status.text("🔹 Uploading to Qdrant...")

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": chunks[i], **metas[i]}
        )
        for i in range(len(embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION, points=points)

    prog.progress(1.0)
    status.text(f"✅ Index built: {len(chunks)} chunks")

# ── RETRIEVE ───────────────────────────────────────────
def retrieve(query, client, qdrant, k=5):
    try:
        q_emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=[query]
        ).data[0].embedding

        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=q_emb,
            limit=k
        )

        return [point.payload["text"] for point in results.points]

    except Exception as e:
        st.error("❌ Retrieval Error")
        st.code(str(e))
        return []

# ── ANSWER ─────────────────────────────────────────────
def answer(query, context, client):
    try:
        system = (
            "You are an inventory analyst.\n"
            "Answer ONLY using provided data.\n"
            "If not found, say 'Not found in data'.\n"
            "Be concise."
        )

        ctx = "\n\n---\n\n".join(context)

        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"{ctx}\n\nQ: {query}"}
            ],
            temperature=0.2
        )

        return resp.choices[0].message.content

    except Exception as e:
        st.error("❌ LLM Error")
        st.code(str(e))
        return "Error generating answer."

# ── UI ─────────────────────────────────────────────────
st.title("📦 Inventory RAG Assistant")

client = get_openai_client(openai_api_key)
qdrant = get_qdrant()

# ── LOAD DEFAULT FILE ──────────────────────────────────
file_path = Path(DEFAULT_FILE)

if not file_path.exists():
    st.error(f"❌ File not found: {DEFAULT_FILE}")
    st.stop()

file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

if st.session_state.get("hash") != file_hash:

    df = pd.read_excel(file_path).fillna("")
    st.session_state["df"] = df
    st.session_state["hash"] = None

    st.success(f"Loaded {len(df)} rows")
    st.dataframe(df.head(5))

    prog = st.progress(0)
    status = st.empty()

    build_index(df, client, qdrant, prog, status)

    st.session_state["hash"] = file_hash

else:
    df = st.session_state["df"]
    st.success("✅ Index Ready")

# ── CHAT ───────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about inventory..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ctx = retrieve(prompt, client, qdrant)
            ans = answer(prompt, ctx, client)

        st.markdown(ans)
        st.session_state["messages"].append({"role": "assistant", "content": ans})

# import streamlit as st
# import pandas as pd
# import hashlib
# import traceback
# from typing import List

# from openai import OpenAI
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# # ── CONFIG ─────────────────────────────────────────────
# st.set_page_config(page_title="Inventory RAG", page_icon="📦", layout="wide")

# COLLECTION = "inventory"
# EMBED_MODEL = "text-embedding-3-small"
# CHAT_MODEL = "gpt-4o-mini"
# EMBED_DIM = 1536
# CHUNK_SIZE = 10   # 🔥 smaller = safer
# BATCH_SIZE = 64   # 🔥 avoid API overload

# # ── SIDEBAR ────────────────────────────────────────────
# with st.sidebar:
#     st.title("⚙️ Configuration")

#     openai_api_key = st.text_input("OpenAI API Key", type="password")

#     excel_file = st.file_uploader(
#         "Upload Inventory Excel",
#         type=["xlsx", "xls"]
#     )

#     st.markdown("---")
#     st.markdown("### About")
#     st.markdown(
#         "Uses OpenAI embeddings + Qdrant (in-memory) "
#         "to answer inventory questions."
#     )

# # ── INIT CLIENTS ───────────────────────────────────────
# @st.cache_resource
# def get_qdrant():
#     return QdrantClient(":memory:")

# def get_openai_client(api_key):
#     return OpenAI(api_key=api_key)

# # ── HELPERS ────────────────────────────────────────────
# def row_to_text(row: dict) -> str:
#     parts = []
#     for k, v in row.items():
#         if pd.notna(v) and str(v).strip():
#             parts.append(f"{k}: {v}")
#     return " | ".join(parts)


# def chunk_dataframe(df: pd.DataFrame):
#     texts, metas = [], []

#     for i, (_, row) in enumerate(df.iterrows()):
#         text = row_to_text(row.to_dict())
#         texts.append(text)

#         metas.append({
#             "row": i,
#             "material": str(row.get("Material", "")),
#             "plant": str(row.get("Plant", "")),
#         })

#     chunks, chunk_metas = [], []

#     for i in range(0, len(texts), CHUNK_SIZE):
#         chunk = "\n".join(texts[i:i + CHUNK_SIZE])[:8000]  # 🔥 trim
#         if chunk.strip():
#             chunks.append(chunk)
#             chunk_metas.append({
#                 "start": i,
#                 "end": i + CHUNK_SIZE
#             })

#     return chunks, chunk_metas


# # ── SAFE EMBEDDINGS ────────────────────────────────────
# def get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
#     all_embeddings = []

#     for i in range(0, len(texts), BATCH_SIZE):
#         batch = texts[i:i + BATCH_SIZE]

#         # 🔥 clean batch
#         batch = [t[:8000] for t in batch if t.strip()]

#         if not batch:
#             continue

#         try:
#             resp = client.embeddings.create(
#                 model=EMBED_MODEL,
#                 input=batch
#             )
#             all_embeddings.extend([r.embedding for r in resp.data])

#         except Exception as e:
#             st.error("❌ Embedding Error")
#             st.code(str(e))
#             st.code(traceback.format_exc())
#             raise

#     return all_embeddings


# # ── BUILD INDEX ────────────────────────────────────────
# def build_index(df, client, qdrant, prog, status):
#     status.text("🔹 Chunking data...")
#     chunks, metas = chunk_dataframe(df)

#     if qdrant.collection_exists(COLLECTION):
#         qdrant.delete_collection(COLLECTION)

#     qdrant.create_collection(
#         COLLECTION,
#         vectors_config=VectorParams(
#             size=EMBED_DIM,
#             distance=Distance.COSINE
#         )
#     )

#     status.text(f"🔹 Embedding {len(chunks)} chunks...")
#     embeddings = get_embeddings(client, chunks)

#     status.text("🔹 Uploading to Qdrant...")

#     points = [
#         PointStruct(
#             id=i,
#             vector=embeddings[i],
#             payload={"text": chunks[i], **metas[i]}
#         )
#         for i in range(len(embeddings))
#     ]

#     qdrant.upsert(collection_name=COLLECTION, points=points)

#     prog.progress(1.0)
#     status.text(f"✅ Index built: {len(chunks)} chunks")


# # ── RETRIEVE ───────────────────────────────────────────


# def retrieve(query, client, qdrant, k=5):
#     try:
#         q_emb = client.embeddings.create(
#             model=EMBED_MODEL,
#             input=[query]
#         ).data[0].embedding

#         results = qdrant.query_points(
#             collection_name=COLLECTION,
#             query=q_emb,
#             limit=k
#         )

#         return [point.payload["text"] for point in results.points]

#     except Exception as e:
#         st.error("❌ Retrieval Error")
#         st.code(str(e))
#         return []

# # ── ANSWER ─────────────────────────────────────────────
# def answer(query, context, client):
#     try:
#         system = (
#             "You are an inventory analyst.\n"
#             "Answer ONLY using provided data.\n"
#             "If not found, say 'Not found in data'.\n"
#             "Be concise."
#         )

#         ctx = "\n\n---\n\n".join(context)

#         resp = client.chat.completions.create(
#             model=CHAT_MODEL,
#             messages=[
#                 {"role": "system", "content": system},
#                 {"role": "user", "content": f"{ctx}\n\nQ: {query}"}
#             ],
#             temperature=0.2
#         )

#         return resp.choices[0].message.content

#     except Exception as e:
#         st.error("❌ LLM Error")
#         st.code(str(e))
#         return "Error generating answer."


# # ── UI ─────────────────────────────────────────────────
# st.title("📦 Inventory RAG Assistant")

# if not openai_api_key:
#     st.warning("Enter OpenAI API Key")
#     st.stop()

# if not excel_file:
#     st.info("Upload Excel file")
#     st.stop()

# client = get_openai_client(openai_api_key)
# qdrant = get_qdrant()

# # ── LOAD FILE ──────────────────────────────────────────
# file_hash = hashlib.md5(excel_file.read()).hexdigest()
# excel_file.seek(0)

# if st.session_state.get("hash") != file_hash:

#     df = pd.read_excel(excel_file).fillna("")
#     st.session_state["df"] = df
#     st.session_state["hash"] = None

#     st.success(f"Loaded {len(df)} rows")
#     st.dataframe(df.head(5))

#     prog = st.progress(0)
#     status = st.empty()

#     build_index(df, client, qdrant, prog, status)

#     st.session_state["hash"] = file_hash

# else:
#     df = st.session_state["df"]
#     st.success("✅ Index Ready")

# # ── CHAT ───────────────────────────────────────────────
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# for m in st.session_state["messages"]:
#     with st.chat_message(m["role"]):
#         st.markdown(m["content"])

# if prompt := st.chat_input("Ask about inventory..."):
#     st.session_state["messages"].append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#             ctx = retrieve(prompt, client, qdrant)
#             ans = answer(prompt, ctx, client)

#         st.markdown(ans)
#         st.session_state["messages"].append({"role": "assistant", "content": ans})


