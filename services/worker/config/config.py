from pydantic import BaseSettings
import os


class Settings(BaseSettings):
    secret_key: str = "Awesome Worker"

    project_root: str = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
    upload_folder: str = "uploads/"
    media_folder: str = "uplodas/"

    bucket_url: str = "https://accident-ai-uploads.s3.amazonaws.com"
    bucket_name: str = "accident-ai-uploads"

    aws_public_key: str = "AKIA3O33ZJQO6QPMK2VQ"
    aws_secret_key: str = "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"


settings = Settings()
