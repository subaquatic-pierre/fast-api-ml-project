import datetime
import secrets
from fastapi import APIRouter
import requests

from models.api_key import ApiKey
from models.api_key import ApiKeyModelRes, ApiKeyModelReq, ApiKeListModelRes
from models.response import BoolModelRes


router = APIRouter(
    prefix="/api",
    tags=["ApiKey"],
    responses={404: {"description": "Not found"}},
)

# --- API Key routes ---


@router.get(
    "/api-key",
    response_description="All API Keys",
    response_model=ApiKeListModelRes,
)
def list_api_key(userId: str = None):
    api_keys = []
    # all_keys = ApiKey.objects()

    if userId:
        api_keys = [api_key.json() for api_key in ApiKey.objects(userId=userId)]
    else:
        api_keys = [api_key.json() for api_key in ApiKey.objects()]

    return {
        "status_code": 200,
        "response_type": "success",
        "description": "User data retrieved successfully",
        "data": api_keys,
    }


@router.post(
    "/api-key",
    response_description="Create new API key",
    response_model=ApiKeyModelRes,
)
def create_api_key(form_data: ApiKeyModelReq):
    # Build expiration time
    now = datetime.datetime.now()
    future = datetime.timedelta(days=7)
    expiration = now + future
    expiration_timestamp = int(datetime.datetime.timestamp(expiration))

    # Get data from form
    data = form_data.dict()

    # Generate API key
    secret_key = secrets.token_urlsafe(16)
    user_id = data.get("userId")
    api_key = f"{user_id}:{secret_key}"

    new_api_key = ApiKey(
        userId=user_id, key=api_key, expirationDate=str(expiration_timestamp)
    )

    status = False

    try:
        new_api_key.save()
        status = True
    except Exception:
        status = False

    if status == False:
        return {
            "status_code": 405,
            "response_type": "error",
            "description": "Unable to generate new API key",
            "data": status,
        }

    return {
        "status_code": 200,
        "response_type": "success",
        "description": "New API key created successfully",
        "data": new_api_key.json(),
    }


@router.delete(
    "/api-key/{api_key_id}",
    response_description="API Key deleted",
    response_model=BoolModelRes,
)
def delete_api_key(api_key_id):
    status = False
    api_key = ApiKey.objects.get(id=api_key_id)

    if api_key:
        api_key.delete()
        status = True

    return {
        "status_code": 200,
        "response_type": "success",
        "description": "API Key deleted",
        "data": status,
    }
