"""
Script de ingesta de políticas internas para base de conocimiento (RAG)
Caso de uso: Asistente Inteligente de Consultas de Políticas Internas

Uso local (sin Azure):
    python scripts/ingest_policies.py

En Azure:
    - Cambiar USE_AZURE = True
    - Configurar las variables de entorno de Azure AI Search + embeddings
"""

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import json

# ======================
# CONFIGURACIÓN
# ======================
USE_AZURE = False  # Cambiar a True cuando se tenga  suscripción de Azure

DOCS_FOLDER = Path("sample_policies_data")  # Carpeta donde están los PDFs de politicas internas
INDEX_NAME = "policies-index"

OUTPUT_CHUNKS_FILE = DOCS_FOLDER / "chunks_preview.json"

# Parámetros de chunking profesional
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Azure (solo se usan si USE_AZURE = True)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)


def load_documents(folder: Path) -> List[Document]:
    """
    Carga todos los PDFs de la carpeta usando LangChain DirectoryLoader + PyPDFLoader.
    Cada página se convierte en un Document con metadata.
    """
    if not folder.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {folder.resolve()}")

    loader = DirectoryLoader(
        path=str(folder),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )

    documents = loader.load()

    if not documents:
        print(f"No se encontraron PDFs en: {folder.resolve()}")
        return []

    print(f"Cargados {len(documents)} páginas desde PDFs.")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Divide los documentos con RecursiveCharacterTextSplitter (estándar profesional).
    Respeta párrafos, oraciones y palabras en ese orden de prioridad.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f" Generados {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    return chunks

def save_chunks_preview(chunks: List[Document], output_path: Path):
    """Guarda una vista previa de los chunks en JSON (útil para desarrollo y debug)."""
    preview = []
    for i, chunk in enumerate(chunks):
        preview.append({
            "id": f"chunk_{i}",
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "char_count": len(chunk.page_content),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)

    print(f"Vista previa guardada en: {output_path}")

def ingest_local(chunks: List[Document]):
    """Modo local: solo genera y guarda los chunks para inspección."""
    save_chunks_preview(chunks, OUTPUT_CHUNKS_FILE)
    print("   (En producción estos chunks se indexan en Azure AI Search + embeddings)")


def ingest_azure(chunks: List[Document]):
    """
    Modo Azure (listo para activar cuando tengas suscripción):
    1. Genera embeddings con Azure OpenAI
    2. Upsert a Azure AI Search
    """
    print("   Modo Azure activado.")
    print("   TODO: implementar AzureOpenAIEmbeddings + AzureSearch")
    # from langchain_openai import AzureOpenAIEmbeddings
    # from langchain_community.vectorstores.azuresearch import AzureSearch
    #
    # embeddings = AzureOpenAIEmbeddings(
    #     azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    #     azure_endpoint=AZURE_OPENAI_ENDPOINT,
    #     api_key=AZURE_OPENAI_API_KEY,
    # )
    # vector_store = AzureSearch(
    #     azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
    #     azure_search_key=AZURE_SEARCH_API_KEY,
    #     index_name=INDEX_NAME,
    #     embedding_function=embeddings.embed_query,
    # )
    # vector_store.add_documents(chunks)
    # print(f" {len(chunks)} chunks indexados en Azure AI Search ({INDEX_NAME})")


def main():
    print("=" * 60)
    print("  INGESTA DE POLÍTICAS INTERNAS - Enterprise Policy Assistant")
    print("  (LangChain: PyPDFLoader + RecursiveCharacterTextSplitter)")
    print("=" * 60)

    # 1. Carga profesional de PDFs
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        return

    # 2. Chunking profesional
    chunks = split_documents(documents)

    # 3. Ingesta (local o Azure)
    if USE_AZURE:
        ingest_azure(chunks)
    else:
        ingest_local(chunks)

    print("\n Ingesta finalizada.")
    print("   Siguiente paso: conectar estos chunks al RAGService del backend.")


if __name__ == "__main__":
    main()