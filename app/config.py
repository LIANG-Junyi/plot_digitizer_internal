from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str

    # Upload limits
    max_image_mb: int = 10
    max_pdf_mb: int = 50
    max_pdf_pages: int = 80

    # Qwen (DashScope) — Standard Quality vision + metadata extraction
    # Leave empty to disable Standard Quality (Improved Quality / Claude only)
    dashscope_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_vision_model: str = "qwen3-vl-plus"
    qwen_text_model: str = "qwen-long"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
