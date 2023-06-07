import requests
import json
from helpers.worker_instance import get_worker_url
from models.case import Case


def generate_damage_assessment_pdf_report(case_id: str):
    data = {"caseId": case_id}

    generate_report_url = f"{get_worker_url()}/damage-assessment/pdf-report"
    res = requests.post(
        generate_report_url,
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )
    data = res.json()

    case = Case.objects.get(id=case_id)
    case.reportUrlPdf = data.get("pdfReportUrl", "")
    case.save()

    return data


def generate_top_view_pdf_report(case_id: str):
    data = {"caseId": case_id}

    generate_report_url = f"{get_worker_url()}/top-view/pdf-report"
    res = requests.post(
        generate_report_url,
        json.dumps(data),
        headers={"Content-Type": "application/json"},
    )
    data = res.json()
    case = Case.objects.get(id=case_id)

    case.reportUrlPdfTopView = data.get("reportUrlPdfTopView", "")
    case.save()

    return data
