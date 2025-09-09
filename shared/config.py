import os

def get_env(name: str, default: str | None = None, required: bool = True) -> str:
    v = os.getenv(name, default)
    if required and (v is None or v == ""):
        raise RuntimeError(f"Missing required setting: {name}")
    return v

class Settings:
    STORAGE_ACCOUNT_NAME           = get_env("STORAGE_ACCOUNT_NAME")
    STORAGE_CONTAINER_INPUT        = get_env("STORAGE_CONTAINER_INPUT")
    STORAGE_CONTAINER_TRANSCRIPTS  = get_env("STORAGE_CONTAINER_TRANSCRIPTS")
    STORAGE_CONTAINER_ATAS         = get_env("STORAGE_CONTAINER_ATAS")

    SPEECH_KEY       = get_env("SPEECH_KEY")
    SPEECH_REGION    = get_env("SPEECH_REGION")
    SPEECH_ENDPOINT  = get_env("SPEECH_ENDPOINT")

    AOAI_ENDPOINT    = get_env("AZURE_OPENAI_ENDPOINT")
    AOAI_KEY         = get_env("AZURE_OPENAI_KEY")
    AOAI_DEPLOYMENT  = get_env("AZURE_OPENAI_DEPLOYMENT")

    LOG_LEVEL        = os.getenv("LOG_LEVEL", "Information")
