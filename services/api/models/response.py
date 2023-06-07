from typing import Optional, Any
from pydantic import BaseModel


class ResponseModel(BaseModel):
    status_code: int = 200
    response_type: str = "success"
    description: str = "Request completed"
    data: Optional[Any]


class BaseModelResponse(BaseModel):
    status_code: int = 200
    response_type: str = "success"
    description: str = "Request completed"


class BoolModelRes(BaseModelResponse):
    data: bool = True
