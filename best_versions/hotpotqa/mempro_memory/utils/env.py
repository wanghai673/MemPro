import os
from typing import Dict


def load_local_env(env_path: str = ".env") -> Dict[str, str]:
    """
    Load simple KEY=VALUE pairs from a local .env file into os.environ.

    Existing environment variables are preserved and returned values are the
    effective values after loading.
    """
    loaded: Dict[str, str] = {}

    if not os.path.exists(env_path):
        return loaded

    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]

            effective_value = os.environ.get(key, value)
            os.environ.setdefault(key, value)
            loaded[key] = effective_value

    return loaded


def get_env_or_default(env_name: str, default: str) -> str:
    return os.getenv(env_name, default)
