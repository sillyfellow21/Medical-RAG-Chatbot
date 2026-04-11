import os
from collections.abc import Mapping
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = ""
    chroma_collection: str = "medical-chatbot"
    chroma_persist_dir: str = "chroma_db"
    retriever_k: int = 3
    llm_model: str = "llama-3.1-8b-instant"
    data_dir: str = "data/"


KEY_ALIASES = {
    "GROQ_API_KEY": [
        "GROQ_API_KEY",
        "groq_api_key",
        "GROQ_KEY",
        "groq_key",
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


def _read_optional_setting(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def get_settings(require_groq: bool = True) -> Settings:
    load_dotenv()

    groq_keys = KEY_ALIASES["GROQ_API_KEY"]

    groq_api_key = _read_env_key(groq_keys)

    if not groq_api_key:
        groq_api_key = _read_streamlit_secret(groq_keys)

    if require_groq and not groq_api_key:
        raise RuntimeError(
            "Missing GROQ_API_KEY in environment/.env/Streamlit secrets."
        )

    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    chroma_collection = _read_optional_setting(
        "CHROMA_COLLECTION", "medical-chatbot"
    )
    chroma_persist_dir = _read_optional_setting(
        "CHROMA_PERSIST_DIR", "chroma_db"
    )
    llm_model = _read_optional_setting("GROQ_MODEL", "llama-3.1-8b-instant")
    data_dir = _read_optional_setting("DATA_DIR", "data/")

    try:
        retriever_k = int(_read_optional_setting("RETRIEVER_K", "3"))
    except ValueError:
        retriever_k = 3

    if retriever_k <= 0:
        retriever_k = 3

    return Settings(
        groq_api_key=groq_api_key,
        chroma_collection=chroma_collection,
        chroma_persist_dir=chroma_persist_dir,
        retriever_k=retriever_k,
        llm_model=llm_model,
        data_dir=data_dir,
    )
