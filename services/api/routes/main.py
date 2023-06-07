import requests
from fastapi import APIRouter

from helpers.worker_instance import get_worker_url, parse_worker_res


router = APIRouter(
    prefix="/api",
    tags=["Main"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
def root():
    res = requests.get(get_worker_url())
    data = parse_worker_res(res)
    return {"instance": "accient-ai-api", "worker": {**data}}
