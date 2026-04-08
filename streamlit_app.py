import os

import streamlit as st

from src.config import get_settings
from src.rag_pipeline import build_rag_chain, build_retriever


def load_streamlit_secrets_into_env() -> None:
    # Streamlit Cloud stores credentials in st.secrets.
    # Local .env files are primarily for local runs.
    if "PINECONE_API_KEY" in st.secrets:
        os.environ["PINECONE_API_KEY"] = str(st.secrets["PINECONE_API_KEY"])
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = str(st.secrets["OPENAI_API_KEY"])


def missing_key_message(exc: Exception) -> str:
    return (
        "Configuration issue: missing API keys.\n\n"
        "Add these in Streamlit Cloud -> App settings -> Secrets:\n\n"
        "PINECONE_API_KEY=\"your_pinecone_key\"\n"
        "OPENAI_API_KEY=\"your_openai_key\"\n\n"
        f"Current error: {exc}"
    )


def format_fallback_answer(docs) -> str:
    if not docs:
        return (
            "I could not generate an LLM response and no relevant context "
            "was retrieved from the knowledge base."
        )

    snippets = []
    for idx, doc in enumerate(docs[:3], start=1):
        snippet = " ".join(doc.page_content.split())[:300]
        if snippet:
            snippets.append(f"{idx}. {snippet}")

    if not snippets:
        return (
            "I could not generate an LLM response and could not extract "
            "useful retrieved context."
        )

    return (
        "OpenAI generation is currently unavailable. "
        "Here is relevant context from the medical knowledge base:\n\n"
        + "\n".join(snippets)
    )


@st.cache_resource(show_spinner=False)
def get_runtime_components():
    load_streamlit_secrets_into_env()
    try:
        settings = get_settings(require_openai=False)
    except Exception as exc:
        return None, None, None, exc

    retriever = build_retriever(settings)

    rag_chain = None
    rag_chain_error = None
    if settings.openai_api_key:
        try:
            rag_chain = build_rag_chain(settings, retriever=retriever)
        except Exception as exc:
            rag_chain_error = exc

    return retriever, rag_chain, rag_chain_error, None


def generate_answer(question: str, retriever, rag_chain) -> str:
    if rag_chain is not None:
        try:
            response = rag_chain.invoke({"input": question})
            return str(response["answer"])
        except Exception:
            pass

    docs = retriever.invoke(question)
    return format_fallback_answer(docs)


def main() -> None:
    st.set_page_config(
        page_title="Medical RAG Chatbot",
        page_icon=":speech_balloon:",
        layout="centered",
    )
    st.title("Medical RAG Chatbot")
    st.caption("Ask medical questions over your indexed PDF knowledge base.")

    with st.sidebar:
        st.subheader("Run Checklist")
        st.markdown(
            "1. Set PINECONE_API_KEY and OPENAI_API_KEY in Streamlit Secrets"
        )
        st.markdown("   (use .env only for local runs)")
        st.markdown("2. Run: python store_index.py")
        st.markdown("3. Start: streamlit run streamlit_app.py")

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
                if rag_chain_error is not None:
                    st.warning(
                        "LLM generation is unavailable in this runtime. "
                        "Using retrieval-only fallback."
                    )
                answer = generate_answer(question, retriever, rag_chain)
            except Exception as exc:
                answer = (
                    "The app could not complete the request. "
                    f"Details: {exc}"
                )
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
