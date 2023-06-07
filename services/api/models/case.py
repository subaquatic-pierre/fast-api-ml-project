from mongoengine import (
    Document,
    StringField,
    BooleanField,
    IntField,
    DictField,
    ListField,
)
from typing import Any, Optional, List

from pydantic import BaseModel, Field

from models.response import BaseModelResponse


class Case(Document):
    type = StringField()
    userId = StringField()
    createdDate = StringField()
    status = StringField()  # PROCESSED, CREATED
    reportGenerated = BooleanField(default=False)
    topViewGenerated = BooleanField(default=False)
    reportUrl = StringField()
    report = ListField(default=[])
    reportUrlPdf = StringField()
    reportUrlPdfTopView = StringField()
    vehicleCount = IntField(default=0)
    all_final_dict = ListField(default=[])

    def json(self):
        return {
            "id": str(self.id),
            "type": self.type,
            "userId": self.userId,
            "createdDate": self.createdDate,
            "status": self.status,
            "reportGenerated": self.reportGenerated,
            "topViewGenerated": self.topViewGenerated,
            "reportUrl": self.reportUrl,
            "reportUrlPdf": self.reportUrlPdf,
            "reportUrlPdfTopView": self.reportUrlPdfTopView,
            "vehicleCount": self.vehicleCount,
            "report": self.report,
            "all_final_dict": self.all_final_dict,
        }


class CaseModel(BaseModel):
    id: str
    type: str
    userId: str
    createdDate: str
    status: str = Field(..., example="PROCESSED")
    reportGenerated: bool
    topViewGenerated: Optional[bool] = False
    reportUrl: Optional[str] = None
    reportUrlPdf: Optional[str] = None
    reportUrlPdfTopView: Optional[str] = None
    vehicleCount: Optional[int] = 0
    report: Any
    all_final_dict: Any

    class Collection:
        name = "case"


class CaseModelRes(BaseModelResponse):
    data: CaseModel


class CaseModelListRes(BaseModelResponse):
    data: List[CaseModel]


class FileUpload(BaseModel):
    fileName: str
    base64Str: str


class CaseModelReq(BaseModel):
    apiKey: str = Field(
        ..., example="62a19913a18e3218bbddeff2:FHH29498iifiedfsdfJFHJ//sdfa"
    )
    userId: str = Field(..., example="62a19913a18e3218bbddeff2")
    type: str = Field(..., example="DAMAGE_ASSESSMENT")


class AddVehicleReq(BaseModel):
    apiKey: str = Field(
        ..., example="62a19913a18e3218bbddeff2:FHH29498iifiedfsdfJFHJ//sdfa"
    )
    userId: str = Field(..., example="62a19913a18e3218bbddeff2")
    fileList: List[FileUpload] = Field(
        ...,
        example=[
            {
                "fileName": "image1.jpeg",
                "base64Str": "kjsdf.sdkkjksdjfkjskdjf...",
            }
        ],
    )
