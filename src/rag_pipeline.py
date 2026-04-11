from langchain_chroma import Chroma

from src.config import Settings
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


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
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import (
            create_stuff_documents_chain,
        )
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_groq import ChatGroq
    except Exception as exc:
        raise RuntimeError(
            "LLM chain dependencies are incompatible with the current "
            "Python runtime. Use Python 3.11 or rely on retrieval fallback."
        ) from exc

    chat_model = ChatGroq(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
