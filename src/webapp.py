from flask import Flask, render_template, request

from src.config import get_settings
from src.rag_pipeline import build_rag_chain


def create_app() -> Flask:
    app = Flask(__name__)
    settings = get_settings(require_openai=True)
    rag_chain = build_rag_chain(settings)

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        print(msg)
        response = rag_chain.invoke({"input": msg})
        print("Response : ", response["answer"])
        return str(response["answer"])

    return app
