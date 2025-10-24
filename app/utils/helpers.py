import json
import os
from pathlib import Path

# Xác định đường dẫn gốc tới thư mục prompts
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"

def load_prompt(prompt_name: str) -> dict:
    """Load prompt template từ file JSON."""
    prompt_path = PROMPTS_DIR / f"{prompt_name}.json"
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")