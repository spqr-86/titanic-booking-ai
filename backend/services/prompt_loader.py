from pathlib import Path


def load_prompt(file_name: str) -> str:
    """Загружает текст промпта из папки prompts."""
    # __file__ -> текущий файл (prompt_loader.py)
    # .parent -> папка services
    # .parent -> папка backend
    # / "prompts" -> папка prompts
    prompt_path = Path(__file__).parent.parent / "prompts" / file_name
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
    