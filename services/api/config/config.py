from pydantic import BaseSettings

from mongoengine import connect


class Settings(BaseSettings):
    secret_key: str = "Awesome API"
    upload_folder: str = "uploads/"
    media_folder: str = "uplodas/"

    worker_url: str = "http://worker:4000"

    mongodb_db: str = "db"
    mongodb_host: str = "123"
    mongodb_port: int = 27017
    mongo_details: str = "unknown"

    class Config:
        orm_mode = True
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def init_db(app):
    db = connect(host=settings.mongo_details)
    app.db = db
