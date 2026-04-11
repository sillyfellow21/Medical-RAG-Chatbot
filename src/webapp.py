from pathlib import Path

from flask import Flask, render_template, request

from src.config import get_settings
from src.rag_pipeline import build_rag_chain, build_retriever


def _format_fallback_answer(docs) -> str:
    if not docs:
        return (
            "Groq generation is currently unavailable and I could not find "
            "relevant context in the medical knowledge base."
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
            "Groq generation is currently unavailable and I could not extract "
            "a useful context snippet from the retrieved documents."
        )

    return (
        "Groq generation is currently unavailable. "
        "Here is relevant context from the medical knowledge base:\n\n"
        + "\n".join(snippets)
    )


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    settings = get_settings(require_groq=False)
    retriever = build_retriever(settings)
    rag_chain = None
    try:
        if settings.groq_api_key:
            rag_chain = build_rag_chain(settings, retriever=retriever)
    except Exception as exc:
        print("RAG chain unavailable:", exc)

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        print(msg)
        if rag_chain is not None:
            try:
                response = rag_chain.invoke({"input": msg})
                print("Response : ", response["answer"])
                return str(response["answer"])
            except Exception as exc:
                print("Chat error:", exc)

        try:
            docs = retriever.invoke(msg)
            fallback = _format_fallback_answer(docs)
            print("Fallback response generated")
            return fallback
        except Exception as fallback_exc:
            print("Fallback error:", fallback_exc)
            return (
                "I could not generate a response right now. "
                "Please verify your GROQ_API_KEY and try again."
            )

    return app
