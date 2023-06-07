import os
from dotenv import load_dotenv

load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))


def use_dot_env(var_name):
    value = os.environ.get(var_name)
    return value


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", use_dot_env("SECRET_KEY"))
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", use_dot_env("UPLOAD_FOLDER"))
    API_URL = os.getenv("API_URL", "http://localhost:5000/api")
