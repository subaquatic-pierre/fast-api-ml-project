from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth.jwt_bearer import JWTBearer
from config.config import init_db
from routes.user import router as UserRouter
from routes.api_key import router as ApiKeyRouter
from routes.case import router as CaseRouter
from routes.main import router as MainRouter

app = FastAPI()

token_listener = JWTBearer()


# Add routes
app.include_router(MainRouter)
app.include_router(UserRouter)
app.include_router(ApiKeyRouter)
app.include_router(CaseRouter)

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

init_db(app)
