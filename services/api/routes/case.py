from datetime import datetime
from fastapi import APIRouter, BackgroundTasks

from models.response import ResponseModel, BoolModelRes
from models.case import (
    CaseModelRes,
    CaseModelReq,
    AddVehicleReq,
    Case,
    CaseModelListRes,
)


from handlers.assessments import (
    add_vehicle_to_damage_assessment,
    new_top_view_assessment,
)
from handlers.case import create_new_case
from handlers.reports import (
    generate_damage_assessment_pdf_report,
    generate_top_view_pdf_report,
)


router = APIRouter(
    prefix="/api",
    tags=["Case"],
    responses={404: {"description": "Not found"}},
)


@router.get(
    "/case",
    response_description="List all cases",
    response_model=CaseModelListRes,
)
def list_cases(userId: str = None):
    cases = []
    if userId:
        cases = [case.json() for case in Case.objects(userId=userId)]
    else:
        cases = [case.json() for case in Case.objects()]

    return {
        "status_code": 200,
        "response_type": "success",
        "description": "Case details",
        "data": cases,
    }


@router.get(
    "/case/{case_id}",
    response_description="Get case",
    response_model=CaseModelRes,
)
def get_case(case_id: str):
    case = Case.objects.get(id=case_id)

    if case:
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Case details",
            "data": case.json(),
        }


@router.get(
    "/case/{case_id}/status/",
    response_description="Get case status",
    response_model=CaseModelRes,
)
def get_case_status(case_id: str):
    case = Case.objects.get(id=case_id)

    if case:
        data = {"status": case.status}
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Getting case status",
            "data": data,
        }


@router.post(
    "/case",
    response_description="Create new case",
    response_model=CaseModelRes,
)
def new_case(data: CaseModelReq, background_tasks: BackgroundTasks):
    now = datetime.now()

    # Create new case
    new_case = Case(
        type=data.type,
        createdDate=str(int(datetime.timestamp(now))),
        userId=data.userId,
        status="CREATED",
    )
    new_case.save()

    background_tasks.add_task(create_new_case, data, str(new_case.id))

    return {
        "status_code": 200,
        "response_type": "created",
        "description": "New Case created",
        "data": new_case.json(),
    }


# ---
# Add vehicle to case
# ---
@router.post(
    "/case/{case_id}/add-vehicle/{version}",
    response_description="Add vehicle to case",
    response_model=CaseModelRes,
)
def add_vehicle_to_case(
    form_data: AddVehicleReq,
    case_id: str,
    version: str,
):
    case = Case.objects.get(id=case_id)

    if case:
        case.status = "UPDATING"
        case.save()

        add_vehicle_to_damage_assessment(form_data, case_id, case.vehicleCount, version)
        case = Case.objects.get(id=case_id)

        return {
            "status_code": 200,
            "response_type": "created",
            "description": "Vehicle added to case",
            "data": case.json(),
        }


# ---
# Generate Top View
# ---
@router.get(
    "/case/{case_id}/top-view",
    response_description="Get top view assessment of case",
    response_model=CaseModelRes,
)
def generate_top_view(case_id: str, background_tasks: BackgroundTasks):
    case = Case.objects.get(id=case_id)

    if case:
        background_tasks.add_task(new_top_view_assessment, case_id)
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Creating top view assessment for case",
            "data": case.json(),
        }


# ---
# Generate PDF for case
# ---
@router.get(
    "/case/{case_id}/damage-assessment-pdf-report",
    response_description="Generate PDF for report for given case Id",
    response_model=BoolModelRes,
)
def damage_assessment_pdf_report(case_id: str, background_tasks: BackgroundTasks):
    case = Case.objects.get(id=case_id)
    if case:
        background_tasks.add_task(generate_damage_assessment_pdf_report, case_id)

        return {
            "status_code": 200,
            "response_type": "created",
            "description": "Case report PDF created",
            "data": True,
        }


# ---
# Generate Top view PDF for case
# ---
@router.get(
    "/case/{case_id}/top-view-pdf-report",
    response_description="Generate Top View PDF for report for given case Id",
    response_model=BoolModelRes,
)
def top_view_pdf_report(case_id: str, background_tasks: BackgroundTasks):
    case = Case.objects.get(id=case_id)
    if case.topViewGenerated == True:
        background_tasks.add_task(generate_top_view_pdf_report, case_id)

        return {
            "status_code": 200,
            "response_type": "created",
            "description": "Case report PDF created",
            "data": True,
        }

    return {
        "status_code": 404,
        "response_type": "created",
        "description": "Unable to create top view PDF without top view assessment",
        "data": False,
    }


# -----
# Delete methods
# -----


@router.delete(
    "/case/{case_id}", response_description="Delete case", response_model=BoolModelRes
)
def delete_case(case_id: str):
    case = Case.objects.get(id=case_id)

    if case:
        case.delete()
        return {
            "status_code": 200,
            "response_type": "success",
            "description": "Case details",
            "data": True,
        }
