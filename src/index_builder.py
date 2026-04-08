from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import Settings
from src.helper import (
    download_hugging_face_embeddings,
    filter_to_minimal_docs,
    load_pdf_file,
    text_split,
)


def build_and_store_index(settings: Settings) -> None:
    extracted_data = load_pdf_file(data=settings.data_dir)
    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data)

    embeddings = download_hugging_face_embeddings()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if not pc.has_index(settings.index_name):
        pc.create_index(
            name=settings.index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=settings.index_name,
        embedding=embeddings,
    )


def is_index_ready(settings: Settings) -> bool:
    pc = Pinecone(api_key=settings.pinecone_api_key)
    if not pc.has_index(settings.index_name):
        return False

    stats = pc.Index(settings.index_name).describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)
    return bool(vector_count and vector_count > 0)


def ensure_index_ready(settings: Settings) -> None:
    if not is_index_ready(settings):
        build_and_store_index(settings)
