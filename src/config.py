import os
from collections.abc import Mapping
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


def _find_value_in_mapping(data: Mapping, key: str) -> str:
    if key in data and data[key]:
        return str(data[key]).strip()

    for value in data.values():
        if isinstance(value, Mapping):
            nested = _find_value_in_mapping(value, key)
            if nested:
                return nested
    return ""


def _read_streamlit_secret(key: str) -> str:
    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        return _find_value_in_mapping(st.secrets, key)
    except Exception:
        return ""


def get_settings(require_openai: bool = True) -> Settings:
    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not pinecone_api_key:
        pinecone_api_key = _read_streamlit_secret("PINECONE_API_KEY")
    if not openai_api_key:
        openai_api_key = _read_streamlit_secret("OPENAI_API_KEY")

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
