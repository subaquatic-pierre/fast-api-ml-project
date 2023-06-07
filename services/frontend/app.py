from flask import Flask
from settings import Config
from routes.main import main


app = Flask(__name__)

app.config.from_object(Config)


app.register_blueprint(main)
