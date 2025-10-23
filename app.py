# app.py — RAG PDF (FAISS + LangChain OpenAI, citas y controles)
import os
import io
import hashlib
import platform
from typing import List, Tuple

import streamlit as st
import pandas as pd

# PDF & text
from pypdf import PdfReader

# LangChain moderno (v0.2+)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
    create_map_reduce_documents_chain,
)

# Token length (más realista que len())
try:
    import tiktoken
except Exception:
    tiktoken = None

# ───────────────────────────────────────────────────────────────
# CONFIG UI
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG PDF – FAISS", page_icon="📄", layout="wide")
st.title("💬 Generación Aumentada por Recuperación (RAG) — PDF → FAISS")
st.caption(f"Python: **{platform.python_version()}** · LangChain (OpenAI + FAISS)")

# ───────────────────────────────────────────────────────────────
# UTILIDADES
# ───────────────────────────────────────────────────────────────
def file_sha1(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

def pdf_to_pages_text(uploaded_file) -> List[Tuple[int, str]]:
    """Devuelve lista [(page_index_1, text_1), ...]"""
    reader = PdfReader(uploaded_file)
    pages = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i, txt.strip()))
    return pages

def token_len(s: str) -> int:
    if not tiktoken:
        return len(s)
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(s))

def make_documents(pages: List[Tuple[int, str]], source_name: str) -> List[Document]:
    docs = []
    for i, text in pages:
        if not text:
            continue
        meta = {"source": source_name, "page": i + 1}
        docs.append(Document(page_content=text, metadata=meta))
    return docs

@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str, api_key: str):
    return OpenAIEmbeddings(model=model_name, api_key=api_key)

@st.cache_resource(show_spinner=False)
def build_faiss_index(
    docs: List[Document],
    embed_model: str,
    api_key: str,
    chunk_size: int,
    chunk_overlap: int,
):
    """Chunkea docs y construye FAISS; se cachea por parámetros."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_len if tiktoken else len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings(embed_model, api_key)
    store = FAISS.from_documents(chunks, embeddings)
    return store, chunks

def annotate_citations(answer: str, docs: List[Document]) -> str:
    """Agrega [n] al final y lista de fuentes numeradas."""
    if not docs:
        return answer
    lines = ["", "##### Fuentes:"]
    for idx, d in enumerate(docs, 1):
        src = d.metadata.get("source", "PDF")
        pg = d.metadata.get("page", "?")
        preview = (d.page_content[:180] + "…") if len(d.page_content) > 180 else d.page_content
        lines.append(f"[{idx}] {src} — pág. {pg}\n> {preview}")
    return answer + "\n\n" + "\n\n".join(lines)

def guardrail_strict_mode(retrieved: List[Document], min_chars: int = 120) -> bool:
    """True si vale la pena responder (hay contexto suficiente)."""
    joined = " ".join(d.page_content for d in retrieved)
    return len(joined.strip()) >= min_chars

# ───────────────────────────────────────────────────────────────
# SIDEBAR — opciones
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuración")

    # API key (st.secrets preferido)
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-…")
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    st.subheader("Modelos")
    llm_model = st.selectbox("Modelo LLM", ["gpt-4o", "gpt-4o-mini"], index=1)
    embed_model = st.selectbox(
        "Embeddings",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="small = +rápido / barato; large = +calidad",
    )

    st.subheader("Chunking")
    chunk_size = st.slider("Tamaño de chunk", 256, 2000, 900, step=64)
    chunk_overlap = st.slider("Solapamiento", 0, 400, 120, step=16)

    st.subheader("Búsqueda")
    k = st.slider("Top-K (documentos)", 1, 12, 4)
    use_mmr = st.toggle("Usar MMR (diversidad)", value=True)
    lambda_mult = st.slider(
        "MMR lambda (relevancia↔diversidad)",
        0.0, 1.0, 0.5, 0.05,
        help="Solo aplica si MMR está activo"
    )

    st.subheader("Generación")
    temperature = st.slider("Temperatura", 0.0, 1.2, 0.1, 0.1)
    chain_type = st.selectbox("Chain Type", ["stuff", "map_reduce"], index=0)
    strict_mode = st.toggle("Modo estricto (responder solo con contexto)", value=True)

# ───────────────────────────────────────────────────────────────
# SUBIDA DE PDF
# ───────────────────────────────────────────────────────────────
st.markdown("### 📄 Carga tu PDF")
uploaded = st.file_uploader("Selecciona un PDF", type=["pdf"])

if uploaded is None:
    st.info("Sube un PDF para construir el índice y hacer preguntas.")
    st.stop()

if not api_key:
    st.warning("Ingresa tu OpenAI API key en la barra lateral.")
    st.stop()

# ───────────────────────────────────────────────────────────────
# PROCESAMIENTO DEL PDF → FAISS
# ───────────────────────────────────────────────────────────────
file_bytes = uploaded.read()
source_name = uploaded.name
file_hash = file_sha1(file_bytes)

with st.spinner("Extrayendo texto del PDF…"):
    pages = pdf_to_pages_text(io.BytesIO(file_bytes))

total_chars = sum(len(t) for _, t in pages)
valid_pages = sum(1 for _, t in pages if t)
st.success(f"✅ Páginas: {len(pages)} · Páginas con texto: {valid_pages} · Caracteres: {total_chars:,}")

if valid_pages == 0:
    st.error("No se extrajo texto. Verifica que el PDF no sea escaneado/imagen. (Tip: usa OCR).")
    st.stop()

docs = make_documents(pages, source_name)

# Cacheo del índice por hash + parámetros clave
cache_key = (file_hash, embed_model, chunk_size, chunk_overlap)

@st.cache_resource(show_spinner=False)
def _cached_build(docs, embed_model, api_key, chunk_size, chunk_overlap):
    return build_faiss_index(docs, embed_model, api_key, chunk_size, chunk_overlap)

with st.spinner("Construyendo índice FAISS (embeddings)…"):
    vectordb, chunk_docs = _cached_build(docs, embed_model, api_key, chunk_size, chunk_overlap)

st.success(f"Índice listo con **{len(chunk_docs)}** chunks")

# ───────────────────────────────────────────────────────────────
# INTERFAZ DE PREGUNTAS
# ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("❓ Pregunta al documento")

qcol, bcol = st.columns([4, 1])
question = qcol.text_area("Escribe tu pregunta", height=90, placeholder="Ej: ¿Cuál es el objetivo principal del documento?")
ask = bcol.button("Preguntar", type="primary", use_container_width=True)

# Opciones de recuperación
if use_mmr:
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 2, 8), "lambda_mult": lambda_mult},
    )
else:
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

if ask:
    if not question.strip():
        st.warning("Escribe una pregunta.")
        st.stop()

    with st.spinner("Buscando en el índice y generando respuesta…"):
        llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            api_key=api_key,
        )

        # Prompt seguro (LCEL): contexto + input
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un asistente que responde únicamente con información del contexto proporcionado. "
             "Si la respuesta no está en el contexto, responde de forma breve: "
             "\"No encontré esa información en el documento\". "
             "Cita páginas al final."),
            ("human",
             "Pregunta: {input}\n\n"
             "Contexto:\n{context}")
        ])

        # Elegir el tipo de combinación de documentos
        if chain_type == "map_reduce":
            doc_chain = create_map_reduce_documents_chain(llm, prompt)
        else:
            doc_chain = create_stuff_documents_chain(llm, prompt)

        # 1) Recuperar documentos (sources)
        sources: List[Document] = retriever.invoke(question)

        # 2) Generar respuesta con esos documentos
        answer = doc_chain.invoke({"input": question, "context": sources})

        # Modo estricto: si no hay contexto suficiente, corta
        if strict_mode and not guardrail_strict_mode(sources):
            st.warning("No encontré suficiente contexto relevante para responder con confianza.")
            st.stop()

        # Anotar citas bonitas
        final = annotate_citations(answer, sources)

    st.markdown("### 🧠 Respuesta")
    st.write(final)

    with st.expander("📚 Chunks citados (fuentes)"):
        if sources:
            rows = []
            for i, d in enumerate(sources, 1):
                rows.append({
                    "#": i,
                    "página": d.metadata.get("page", "?"),
                    "caracteres": len(d.page_content),
                    "preview": (d.page_content[:300] + "…") if len(d.page_content) > 300 else d.page_content
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Exportar fuentes (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="rag_fuentes.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No se devolvieron fuentes (revisa parámetros o pregunta distinta).")

# ───────────────────────────────────────────────────────────────
# EXTRA: Descargas útiles
# ───────────────────────────────────────────────────────────────
with st.expander("⬇️ Exportar texto del PDF (limpio)"):
    all_text = "\n\n".join([f"[Pág. {p}] {t}" for p, t in pages if t])
    st.download_button(
        "Descargar .txt",
        data=all_text.encode("utf-8"),
        file_name=f"{source_name.rsplit('.',1)[0]}_texto.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.markdown("---")
st.caption("RAG con FAISS · Embeddings OpenAI v3 · Citas por página · Controles de chunking y búsqueda (API moderna LCEL, sin create_retrieval_chain)")
