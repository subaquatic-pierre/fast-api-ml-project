import os


class Config:
    project_root = os.path.dirname(__file__)
    api_url = "http://34.233.11.37/api"
    # api_url = "http://localhost:5000/api"


config = Config()
