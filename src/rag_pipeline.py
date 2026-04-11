from langchain_chroma import Chroma

from src.config import Settings
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


GROQ_TIMEOUT_SECONDS = 15
GROQ_MAX_RETRIES = 1
MAX_CONTEXT_CHARS = 4000


def _serialize_response_content(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
            else:
                chunks.append(str(item))
        return "".join(chunks)

    return str(content)


def _build_context(docs, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    parts = []
    used = 0
    for doc in docs:
        snippet = " ".join(doc.page_content.split())
        if not snippet:
            continue

        remaining = max_chars - used
        if remaining <= 0:
            break

        if len(snippet) > remaining:
            snippet = snippet[:remaining]

        parts.append(snippet)
        used += len(snippet)

    return "\n\n".join(parts)


class SimpleRagChain:
    def __init__(self, retriever, chat_model):
        self.retriever = retriever
        self.chat_model = chat_model

    def invoke(self, inputs):
        question = str(inputs.get("input", "")).strip()
        docs = self.retriever.invoke(question)
        context = _build_context(docs)

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=system_prompt.format(context=context)),
            HumanMessage(content=question),
        ]

        response = self.chat_model.invoke(messages)
        return {"answer": _serialize_response_content(response.content)}


def build_retriever(settings: Settings):
    embeddings = download_hugging_face_embeddings()

    docsearch = Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=embeddings,
        persist_directory=settings.chroma_persist_dir,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )
    return retriever


def build_rag_chain(settings: Settings, retriever=None):
    if retriever is None:
        retriever = build_retriever(settings)

    try:
        from langchain_groq import ChatGroq
    except Exception as exc:
        raise RuntimeError(
            "LLM chain dependencies are incompatible with the current "
            "Python runtime. Use Python 3.11 or rely on retrieval fallback."
        ) from exc

    chat_model = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        timeout=GROQ_TIMEOUT_SECONDS,
        max_retries=GROQ_MAX_RETRIES,
    )
    return SimpleRagChain(retriever=retriever, chat_model=chat_model)
