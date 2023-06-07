import requests
import json

from helpers.worker_instance import get_worker_url
from models.case import Case


def create_new_case(data: dict, case_id: str):
    data = {"caseId": case_id}

    case = Case.objects.get(id=case_id)
    try:
        res = requests.post(
            f"{get_worker_url()}/new-case",
            json.dumps(data),
            headers={"Content-Type": "application/json"},
        )

        res_data = res.json()
        case_status = res_data.get("status")
        report_url = res_data.get("reportUrl")
        report = res_data.get("report")

        case.reportGenerated = True
        case.reportUrl = report_url
        case.status = case_status
        case.report = report
        case.save()

    except Exception:
        case.reportGenerated = False
        case.save()
        return


# Update report image
def update_case_image(form_data: dict, case_id: str):
    data = {"caseId": case_id, **form_data}

    # Make request to worker with new image
    update_case_image_url = f"{get_worker_url()}/update-case-image"
    res = requests.post(
        update_case_image_url,
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )

    # Get data from worker
    data = res.json()
    case_status = data.get("status")

    # Update case
    case = Case.objects.get(id=case_id)
    case.status = case_status
    case.save()


# Update case report
def update_case_report(form_data: dict, case_id: str):
    data = {"caseId": case_id, **form_data}

    update_case_report_url = f"{get_worker_url()}/update-case-report"
    res = requests.post(
        update_case_report_url,
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )
    data = res.json()
