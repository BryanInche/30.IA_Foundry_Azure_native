"""
Indexa las políticas en Chroma (RAG local, sin Azure).

Uso (desde la raíz del proyecto):
    python scripts/index_local_chroma.py
"""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ======================
# CONFIG
# ======================
DOCS_FOLDER = Path("sample_policies_data")
CHROMA_DIR = Path("vectorstore/chroma_policies")
COLLECTION_NAME = "policies"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Modelo local de embeddings (ligero y bueno para español/inglés)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_and_split() -> List[Document]:
    """Carga PDFs y aplica el mismo chunking profesional que la ingesta."""
    if not DOCS_FOLDER.exists():
        raise FileNotFoundError(f"No existe: {DOCS_FOLDER.resolve()}")

    loader = DirectoryLoader(
        path=str(DOCS_FOLDER),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"Cargadas {len(documents)} páginas.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Generados {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """Crea / sobrescribe el índice Chroma local."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Si ya existe, lo borramos para reindexar limpio
    if any(CHROMA_DIR.iterdir()):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        print("Índice Chroma anterior eliminado (reindexación limpia).")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"Indexados {len(chunks)} chunks en: {CHROMA_DIR.resolve()}")
    return vectorstore


def smoke_test(vectorstore: Chroma):
    """Prueba rápida de búsqueda semántica."""
    query = "¿En cuánto tiempo debo reportar un incidente de seguridad?"
    results = vectorstore.similarity_search(query, k=3)
    print("\n--- Smoke test ---")
    print(f"Pregunta: {query}")
    for i, doc in enumerate(results, 1):
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"  [{i}] {preview}...")
    print("--- Fin smoke test ---\n")


def main():
    print("=" * 60)
    print("  INDEXACIÓN LOCAL → Chroma (RAG sin Azure)")
    print("=" * 60)

    chunks = load_and_split()
    if not chunks:
        print("No hay chunks. Abortando.")
        return

    vs = build_vectorstore(chunks)
    smoke_test(vs)

    print("Listo. Chroma disponible para el RAGService.")
    print(f"Carpeta: {CHROMA_DIR.resolve()}")


if __name__ == "__main__":
    main()