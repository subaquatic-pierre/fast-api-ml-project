import requests
import json

from models.case import Case


from helpers.worker_instance import get_worker_url


def add_vehicle_to_damage_assessment(form_data, case_id, vehicle_count, version):
    case = Case.objects.get(id=case_id)

    data = {
        "caseId": case_id,
        "vehicleCount": vehicle_count,
        "uploadData": form_data.json(),
        "report": case.report,
        "all_final_dict": case.all_final_dict,
        "api_version": version,
    }

    res = requests.post(
        f"{get_worker_url()}/damage-assessment/add-vehicle",
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )

    res_data = res.json()
    case_status = res_data.get("status")
    report = res_data.get("report")
    all_final_dict = res_data.get("all_final_dict")

    # Update case with failed data
    if not case_status == "FAILED":
        case.vehicleCount = case.vehicleCount + 1

    # Update case with valid data
    case.status = case_status
    case.report = report
    case.all_final_dict = all_final_dict
    case.reportUrlPdf = None
    case.reportUrlPdfTopView = None

    case.save()


def new_top_view_assessment(case_id: str):
    case = Case.objects.get(id=case_id)
    data = {
        "caseId": case_id,
        "report": case.report,
        "vehicleCount": case.vehicle_count,
        "report": case.report,
        "all_final_dict": case.all_final_dict,
    }

    res = requests.post(
        f"{get_worker_url()}/top-view",
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )

    res_data = res.json()
    case_status = res_data.get("status")
    report = res_data.get("report")

    # Update case with new data
    case.status = case_status
    case.reportUrlPdf = None
    case.report = report
    case.reportUrlPdfTopView = None
    case.topViewGenerated = True

    case.save()
