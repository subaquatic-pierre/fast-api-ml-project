from fastapi import FastAPI
from flask import Flask
from routes.main import router as MainRouter
from fastapi.middleware.cors import CORSMiddleware

app = Flask(__name__)
app = FastAPI()


# Add routes
app.include_router(MainRouter)

# Configure CORS
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "*",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
