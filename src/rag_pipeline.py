from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.config import Settings
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt


def build_rag_chain(settings: Settings):
    embeddings = download_hugging_face_embeddings()

    docsearch = PineconeVectorStore.from_existing_index(
        index_name=settings.index_name,
        embedding=embeddings,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.retriever_k},
    )

    chat_model = ChatOpenAI(model=settings.llm_model)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)
