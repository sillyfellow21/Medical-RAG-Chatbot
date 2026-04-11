def _is_running_inside_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


if _is_running_inside_streamlit():
    from streamlit_app import main as streamlit_main

    streamlit_main()
else:
    from src.webapp import create_app

    app = create_app()

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8080, debug=True)
