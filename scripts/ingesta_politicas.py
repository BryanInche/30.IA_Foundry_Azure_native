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

# ======================
# CONFIGURACIÓN
# ======================
USE_AZURE = False  # Cambiar a True cuando se tenga  suscripción de Azure

DOCS_FOLDER = Path("C:\Freelance\Azure_Foundry_Native\sample_policies_data")  # Carpeta donde están los PDFs de politicas internas
INDEX_NAME = "policies-index"

# Azure (solo se usan si USE_AZURE = True)
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
)


def load_pdfs(folder: Path) -> List[dict]:
    """Carga todos los PDFs de la carpeta y extrae texto por página."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Instala pypdf: pip install pypdf")

    documents = []
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron PDFs en: {folder.resolve()}")
        return []

    print(f" Encontrados {len(pdf_files)} PDFs:")
    for pdf_path in pdf_files:
        print(f"   - {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "id": f"{pdf_path.stem}_p{page_num}",
                    "title": pdf_path.stem.replace("_", " "),
                    "source": pdf_path.name,
                    "page": page_num,
                    "content": text.strip(),
                })

    print(f"Extraídas {len(documents)} páginas con contenido.")
    return documents


def chunk_documents(documents: List[dict], chunk_size: int = 800, overlap: int = 150) -> List[dict]:
    """Divide el texto en chunks con solapamiento."""
    chunks = []
    for doc in documents:
        text = doc["content"]
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": f"{doc['id']}_c{chunk_id}",
                    "title": doc["title"],
                    "source": doc["source"],
                    "page": doc["page"],
                    "content": chunk_text,
                })
                chunk_id += 1
            start += chunk_size - overlap
    print(f" Generados {len(chunks)} chunks.")
    return chunks


def ingest_local(chunks: List[dict]):
    """Modo local: guarda los chunks en un JSON para desarrollo."""
    import json
    output_path = Path("sample_policies_data/chunks_preview.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Chunks guardados en: {output_path}")
    print("   (En producción estos chunks se indexan en Azure AI Search)")


def ingest_azure(chunks: List[dict]):
    """
    Modo Azure: genera embeddings y sube a Azure AI Search.
    Requiere: azure-search-documents, openai, variables de entorno configuradas.
    """
    print("   Modo Azure activado (aún no implementado completamente).")
    print("   Próximos pasos cuando tengas suscripción:")
    print("   1. Crear índice en Azure AI Search con campos: id, title, content, source, page, embedding")
    print("   2. Generar embeddings con text-embedding-3-large")
    print("   3. Upsert de los documentos")
    # Aquí irá el código real de Azure AI Search + embeddings


def main():
    print("=" * 60)
    print("  INGESTA DE POLÍTICAS INTERNAS - Enterprise Policy Assistant")
    print("=" * 60)

    if not DOCS_FOLDER.exists():
        print(f" Carpeta no encontrada: {DOCS_FOLDER.resolve()}")
        return

    documents = load_pdfs(DOCS_FOLDER)
    if not documents:
        return

    chunks = chunk_documents(documents)

    if USE_AZURE:
        ingest_azure(chunks)
    else:
        ingest_local(chunks)

    print("\n Ingesta finalizada.")
    print("   Siguiente paso: conectar estos chunks al RAG Service del backend.")


if __name__ == "__main__":
    main()