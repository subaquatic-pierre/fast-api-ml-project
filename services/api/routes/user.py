from typing import List
from fastapi import APIRouter, Body

from models.user import UserModelReq, User, UserModelRes, UserModelListRes
from models.response import BoolModelRes

router = APIRouter(
    prefix="/api",
    tags=["User"],
    responses={404: {"description": "Not found"}},
)

# --- UserModelReq routes ---


@router.get(
    "/user",
    response_description="Users listed",
    response_model=UserModelListRes,
)
def list_users():
    users = [user.json() for user in User.objects()]
    return {
        "status_code": 200,
        "response_type": "success",
        "description": "UserModelReq data retrieved successfully",
        "data": users,
    }


@router.post(
    "/user",
    response_description="UserModelReq created",
    response_model=UserModelRes,
)
def create_user(user: UserModelReq = Body(...)):
    new_user = User(**user.dict()).save()
    return {
        "status_code": 200,
        "response_type": "success",
        "description": "UserModelReq created",
        "data": new_user.json(),
    }


@router.get(
    "/user/{uuid}",
    response_description="User fetched",
    response_model=UserModelRes,
)
def get_user(uuid):
    user = User.objects.get(id=uuid)
    return {
        "status_code": 200,
        "response_type": "success",
        "description": "UserModelReq created",
        "data": user.json(),
    }


@router.delete(
    "/user/{uuid}",
    response_description="UserModelReq deleted",
    response_model=BoolModelRes,
)
def handle_delete_user(uuid):
    status = False
    user = User.objects.get(id=uuid)
    if user:
        user.delete()
        status = True

    return {
        "status_code": 200,
        "response_type": "success",
        "description": "UserModelReq created",
        "data": status,
    }
