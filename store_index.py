from src.config import get_settings
from src.index_builder import build_and_store_index


if __name__ == "__main__":
    settings = get_settings(require_openai=False)
    build_and_store_index(settings)
