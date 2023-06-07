from typing import Union, List
from mongoengine import Document, StringField
from pydantic import BaseModel, Field

from models.response import BaseModelResponse


class ApiKey(Document):
    userId = StringField()
    key = StringField()
    expirationDate = StringField()

    def json(self):
        try:
            return {
                "id": str(self.id),
                "userId": self.userId,
                "key": self.key,
                "expirationDate": self.expirationDate,
            }
        except Exception:
            return {}


class ApiKeyModel(BaseModel):
    id: str
    userId: str = Field(..., example="62a19913a18e3218bbddeff2")
    key: str = Field(
        ..., example="62a19913a18e3218bbddeff2:FHH29498iifiedfsdfJFHJ//sdfa"
    )
    expirationDate: str = Field(..., example="1656394837")


class ApiKeyModelRes(BaseModelResponse):
    data: ApiKeyModel


class ApiKeListModelRes(BaseModelResponse):
    data: List[ApiKeyModel]


class ApiKeyModelReq(BaseModel):
    userId: str = "62a19913a18e3218bbddeff2"
    expirationDays: Union[str, None] = "7"
