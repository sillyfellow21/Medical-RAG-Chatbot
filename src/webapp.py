from pathlib import Path

from flask import Flask, render_template, request

from src.config import get_settings
from src.rag_pipeline import build_rag_chain


def create_app() -> Flask:
    project_root = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )
    settings = get_settings(require_openai=True)
    rag_chain = build_rag_chain(settings)

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        print(msg)
        try:
            response = rag_chain.invoke({"input": msg})
            print("Response : ", response["answer"])
            return str(response["answer"])
        except Exception as exc:
            print("Chat error:", exc)
            return (
                "I could not generate a response right now. "
                "Please verify your OpenAI API quota and try again."
            )

    return app
