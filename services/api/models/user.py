from mongoengine import Document, StringField
from pydantic import BaseModel, EmailStr, Field
from typing import List

from models.response import BaseModelResponse


class User(Document):
    fullName = StringField()
    email = StringField()
    password = StringField()

    def json(self):
        return {"id": str(self.id), "fullName": self.fullName, "email": self.email}

    class Collection:
        name = "user"


class UserModel(BaseModel):
    id: str
    fullName: str
    email: EmailStr

    class Collection:
        name = "user"


class UserModelRes(BaseModelResponse):
    data: UserModel


class UserModelListRes(BaseModelResponse):
    data: List[UserModel]


class UserModelReq(BaseModel):
    fullName: str = Field(..., example="John Doe")
    password: str = Field(..., example="password")
    email: EmailStr = Field(..., example="user@email.com")

    class Collection:
        name = "user"
