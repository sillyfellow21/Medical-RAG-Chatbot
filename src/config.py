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


KEY_ALIASES = {
    "PINECONE_API_KEY": [
        "PINECONE_API_KEY",
        "pinecone_api_key",
        "PINECONE_KEY",
        "pinecone_key",
    ],
    "OPENAI_API_KEY": [
        "OPENAI_API_KEY",
        "openai_api_key",
        "OPENAI_KEY",
        "openai_key",
    ],
}


def _norm_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _keys_match(left: str, right: str) -> bool:
    return _norm_key(left) == _norm_key(right)


def _find_value_in_mapping(data: Mapping, keys: list[str]) -> str:
    for key in keys:
        if key in data and data[key]:
            return str(data[key]).strip()

    for mapping_key, mapping_value in data.items():
        for key in keys:
            if _keys_match(str(mapping_key), key) and mapping_value:
                return str(mapping_value).strip()

    for value in data.values():
        if isinstance(value, Mapping):
            nested = _find_value_in_mapping(value, keys)
            if nested:
                return nested
    return ""


def _read_streamlit_secret(keys: list[str]) -> str:
    try:
        import streamlit as st
    except Exception:
        return ""

    try:
        return _find_value_in_mapping(st.secrets, keys)
    except Exception:
        return ""


def _read_env_key(keys: list[str]) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value

    for env_key, env_value in os.environ.items():
        for key in keys:
            if _keys_match(env_key, key) and str(env_value).strip():
                return str(env_value).strip()

    return ""


def get_settings(require_openai: bool = True) -> Settings:
    load_dotenv()

    pinecone_keys = KEY_ALIASES["PINECONE_API_KEY"]
    openai_keys = KEY_ALIASES["OPENAI_API_KEY"]

    pinecone_api_key = _read_env_key(pinecone_keys)
    openai_api_key = _read_env_key(openai_keys)

    if not pinecone_api_key:
        pinecone_api_key = _read_streamlit_secret(pinecone_keys)
    if not openai_api_key:
        openai_api_key = _read_streamlit_secret(openai_keys)

    if not pinecone_api_key:
        raise RuntimeError(
            "Missing PINECONE_API_KEY in environment/.env/Streamlit secrets."
        )
    if require_openai and not openai_api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY in environment/.env/Streamlit secrets."
        )

    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    return Settings(
        pinecone_api_key=pinecone_api_key,
        openai_api_key=openai_api_key,
    )
