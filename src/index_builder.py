from pathlib import Path

from chromadb import PersistentClient
from langchain_chroma import Chroma

from src.config import Settings
from src.helper import (
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)


CHROMA_WRITE_BATCH_SIZE = 1000


def _batched(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def build_and_store_index(settings: Settings) -> None:
    extracted_data = load_pdf_file(data=settings.data_dir)
    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data)

    if not text_chunks:
        return

    embeddings = download_hugging_face_embeddings()
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(name=settings.chroma_collection)
    except Exception:
        pass

    vector_store = Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
    )
    for batch in _batched(text_chunks, CHROMA_WRITE_BATCH_SIZE):
        vector_store.add_documents(batch)


def is_index_ready(settings: Settings) -> bool:
    persist_dir = Path(settings.chroma_persist_dir)
    if not persist_dir.exists():
        return False

    client = PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=settings.chroma_collection)
    except Exception:
        return False

    return collection.count() > 0


def ensure_index_ready(settings: Settings) -> None:
    if not is_index_ready(settings):
        build_and_store_index(settings)
