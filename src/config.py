import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    pinecone_api_key: str
    openai_api_key: str
    index_name: str = "medical-chatbot"
    retriever_k: int = 3
    llm_model: str = "gpt-4o"
    data_dir: str = "data/"


def get_settings(require_openai: bool = True) -> Settings:
    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "")
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")

    if not pinecone_api_key:
        raise RuntimeError(
            "Missing PINECONE_API_KEY in environment or .env file."
        )
    if require_openai and not openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY in environment or .env file."
        )

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return Settings(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
    )
