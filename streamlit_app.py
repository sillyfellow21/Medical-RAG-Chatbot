import os
from collections.abc import Mapping

import streamlit as st

from src.config import Settings, get_settings
from src.index_builder import ensure_index_ready
from src.rag_pipeline import build_rag_chain, build_retriever


APP_BUILD = "2026-04-09-auto-mode-1"


SECRET_KEY_ALIASES = {
    "GROQ_API_KEY": [
        "GROQ_API_KEY",
        "groq_api_key",
        "GROQ_KEY",
        "groq_key",
    ],
}


def _norm_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


def _iter_secret_items(data):
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield str(key), value
            yield from _iter_secret_items(value)


def _safe_streamlit_secrets():
    try:
        return st.secrets
    except Exception:
        return {}


def load_session_keys_into_env() -> None:
    groq_key = st.session_state.get("runtime_groq_key", "").strip()

    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key


def load_streamlit_secrets_into_env() -> None:
    # Streamlit Cloud stores credentials in st.secrets.
    # Local .env files are primarily for local runs.
    secrets = _safe_streamlit_secrets()
    discovered = {}
    for key, value in _iter_secret_items(secrets):
        if value is None:
            continue
        value_text = str(value).strip()
        if not value_text:
            continue
        discovered[_norm_key(key)] = value_text

    for env_key, aliases in SECRET_KEY_ALIASES.items():
        for alias in aliases:
            norm_alias = _norm_key(alias)
            if norm_alias in discovered:
                os.environ[env_key] = discovered[norm_alias]
                break

    load_session_keys_into_env()


def missing_key_message(exc: Exception) -> str:
    error_type = exc.__class__.__name__
    return (
        "Configuration issue: missing API keys.\n\n"
        "Add these in Streamlit Cloud -> App settings -> Secrets:\n\n"
        "GROQ_API_KEY=\"your_groq_key\"\n\n"
        "Or use the sidebar Runtime Key Override for a "
        "temporary session fix.\n\n"
        f"Error type: {error_type}"
    )


def key_presence_diagnostics() -> dict[str, bool]:
    secrets = _safe_streamlit_secrets()
    discovered = set()
    for key, value in _iter_secret_items(secrets):
        if value is None:
            continue
        if str(value).strip():
            discovered.add(_norm_key(key))

    groq_secret_found = any(
        _norm_key(alias) in discovered
        for alias in SECRET_KEY_ALIASES["GROQ_API_KEY"]
    )

    diagnostics = {
        "secrets:available": bool(secrets),
        "env:GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY", "")),
        "secrets:any_alias:GROQ_API_KEY": groq_secret_found,
        "session:runtime_groq_key": bool(
            st.session_state.get("runtime_groq_key", "").strip()
        ),
    }
    return diagnostics


def format_fallback_answer(docs) -> str:
    if not docs:
        return (
            "I could not generate an LLM response and no relevant context "
            "was retrieved from the knowledge base."
        )

    unique_snippets = []
    seen = set()
    for doc in docs:
        snippet = " ".join(doc.page_content.split())[:300]
        if not snippet:
            continue
        normalized = snippet.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_snippets.append(snippet)
        if len(unique_snippets) == 3:
            break

    snippets = [
        f"{idx}. {snippet}"
        for idx, snippet in enumerate(unique_snippets, start=1)
    ]

    if not snippets:
        return (
            "I could not generate an LLM response and could not extract "
            "useful retrieved context."
        )

    return (
        "Groq generation is currently unavailable. "
        "Here is relevant context from the medical knowledge base:\n\n"
        + "\n".join(snippets)
    )


@st.cache_resource(show_spinner=False)
def get_cached_retriever(settings: Settings):
    ensure_index_ready(settings)
    return build_retriever(settings)


@st.cache_resource(show_spinner=False)
def get_cached_rag_chain(settings: Settings):
    retriever = get_cached_retriever(settings)
    return build_rag_chain(settings, retriever=retriever)


def get_runtime_components():
    load_streamlit_secrets_into_env()
    try:
        settings = get_settings(require_groq=False)
    except Exception as exc:
        return None, None, None, None, exc

    retriever = get_cached_retriever(settings)

    rag_chain = None
    rag_chain_error = None
    if settings.groq_api_key:
        try:
            rag_chain = get_cached_rag_chain(settings)
        except Exception as exc:
            rag_chain_error = exc

    return settings, retriever, rag_chain, rag_chain_error, None


def generate_answer(question: str, retriever, rag_chain):
    fallback_reason = None

    if rag_chain is not None:
        try:
            response = rag_chain.invoke({"input": question})
            return str(response["answer"]), fallback_reason
        except Exception as exc:
            error_name = exc.__class__.__name__
            if "timeout" in str(exc).lower():
                fallback_reason = (
                    "Groq timed out. Showing retrieval fallback for speed."
                )
            else:
                fallback_reason = (
                    f"Groq call failed ({error_name}). "
                    "Showing retrieval fallback."
                )

    docs = retriever.invoke(question)
    return format_fallback_answer(docs), fallback_reason


def main() -> None:
    st.set_page_config(
        page_title="Medical RAG Chatbot",
        page_icon=":speech_balloon:",
        layout="centered",
    )
    st.title("Medical RAG Chatbot")
    st.caption("Ask medical questions over your indexed PDF knowledge base.")

    with st.sidebar:
        st.caption(f"Build: {APP_BUILD}")
        st.subheader("Run Checklist")
        st.markdown(
            "1. Set GROQ_API_KEY in Streamlit Secrets"
        )
        st.markdown("   (use .env only for local runs)")
        st.markdown("2. Run: python store_index.py")
        st.markdown("3. Start: streamlit run streamlit_app.py")

        st.subheader("Runtime Key Override")
        st.caption(
            "If secrets are not detected, provide a key here for this "
            "session only."
        )
        st.text_input(
            "Groq API Key",
            key="runtime_groq_key",
            type="password",
            placeholder="gsk_...",
        )
        if st.button("Apply runtime keys"):
            load_session_keys_into_env()
            st.success("Runtime keys applied for this session.")
            st.rerun()

        with st.expander("Key diagnostics", expanded=False):
            for source, present in key_presence_diagnostics().items():
                st.write(f"- {source}: {present}")
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello. Ask me a medical question based on "
                    "your indexed data."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Type your question here...")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                (
                    settings,
                    retriever,
                    rag_chain,
                    rag_chain_error,
                    settings_error,
                ) = get_runtime_components()
                if settings_error is not None:
                    answer = missing_key_message(settings_error)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    return
                if not settings.groq_api_key:
                    st.warning(
                        "GROQ_API_KEY is not detected in this app runtime. "
                        "No calls are sent to Groq; "
                        "retrieval-only mode is active."
                    )
                if rag_chain_error is not None:
                    st.warning(
                        "Groq generation is unavailable in this runtime. "
                        "Using retrieval-only fallback."
                    )
                answer, fallback_reason = generate_answer(
                    question, retriever, rag_chain
                )
                if fallback_reason:
                    st.info(fallback_reason)
            except Exception as exc:
                error_type = exc.__class__.__name__
                answer = (
                    "The app could not complete the request. "
                    f"Error type: {error_type}"
                )
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
