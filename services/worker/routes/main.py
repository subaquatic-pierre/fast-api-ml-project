import json
from fastapi import APIRouter, BackgroundTasks

import tempfile

from config.logger import logger
from services.model_service_v1 import DamageModel as DamageModel_v1
from services.model_service_v2 import DamageModel as DamageModel_v2


from handlers.damage_assessment import process_damage_assessment
from handlers.top_view import process_top_view
from handlers.reports import (
    generate_damage_assessment_pdf_report,
    generate_top_view_pdf_report,
)

from helpers.images import save_images

from helpers.s3 import (
    get_case_report,
    upload_case_to_s3,
    upload_file,
)

from helpers.reports import save_json_report
from helpers.utils import (
    clean_case_dir,
    build_case_dir,
    get_bucket_url,
    save_all_final_dict,
    get_all_final_dict,
)


router = APIRouter(
    tags=["Main"],
    responses={404: {"description": "Not found"}},
)


@router.get("/")
def home():
    return {"instance_name": "worker"}


@router.post("/new-case")
def new_case(data: dict):
    status = "STARTED"
    log = "New case report creation started."
    logger.info(log)

    # Get data from request
    case_id = data.get("caseId")

    # Build new case dir
    build_case_dir(case_id)

    try:
        save_json_report(case_id, [])
        log += " Report saved."
    except Exception as e:
        logger.error(f"Failed to save reports, Message: {e}")
        status = "FAILED"
        log += " Report saving failed."

    try:
        upload_case_to_s3(case_id)
        log += " Files successfully uploaded."
    except Exception as e:
        logger.error(f"Failed to upload case and clean directory, Message: {e}")
        status = "FAILED"
        log += " Uploading files failed."

    res = {
        "status": status,
        "log": log,
        "reportUrl": f"{get_bucket_url()}/{case_id}/report.json",
    }

    return res


@router.post("/damage-assessment/add-vehicle")
def damage_assessment(data: dict, background_tasks: BackgroundTasks):
    status = "STARTED_UPDATE"
    log = "Damage assessment started for new vehicle."

    # Get data from request
    case_id = data.get("caseId")

    # Set latest version
    api_version = data.get("api_version", "v2")
    all_final_dict = data.get("all_final_dict")
    report = data.get("report", [])

    # Get form upload data
    upload_data = json.loads(data.get("uploadData", {}))
    file_list = upload_data.get("fileList", [])

    # Normalize filename
    for file in file_list:
        file["fileName"] = file["fileName"].replace(" ", "")

    # Get car number from request
    current_vehicle_count = int(data.get("vehicleCount"))
    vehicle_number = current_vehicle_count + 1

    # Get report from S3

    # Build new case dir
    case_dir = build_case_dir(case_id)

    # Save images
    image_path_list = save_images(case_id, file_list, image_prefix=vehicle_number)

    # Save all final dict for top view assessment
    if not current_vehicle_count == 0:
        save_all_final_dict(case_id, all_final_dict)

    if api_version == "v2":
        # Set Damage model on API version
        damage_model = DamageModel_v2(case_dir, update="v2_1")

    elif api_version == "v1":
        # Set Damage model on API version
        damage_model = DamageModel_v1(case_dir, update="v1_1")

    else:
        damage_model = DamageModel_v2(case_dir, update="v2_1")

    try:
        car_report = process_damage_assessment(
            case_id,
            damage_model,
            image_path_list,
            vehicle_number=vehicle_number,
        )

        # Update all final dict with new car data
        for image_report in car_report:
            report.append(image_report)

        status = "CAR_ADDED"
        log += " Machine learning completed."
    except Exception as e:
        logger.error(f"Machine learning prediction failed, Message: {e}")
        status = "FAILED"
        log += " Machine learning failed."

    try:
        print("Saving report")
        save_json_report(case_id, report)
        log += " Report saved."
    except Exception as e:
        logger.error(f"Failed to save reports, Message: {e}")
        status = "FAILED"
        log += " Report saving failed."

    try:
        upload_case_to_s3(case_id)
        # Clean case directory
        clean_case_dir(case_id)
        log += " Files successfully uploaded and case directory cleaned."
    except Exception as e:
        logger.error(
            f"Failed to upload items to bucket and clean directory, Message: {e}"
        )
        status = "FAILED"
        log += " Uploading files failed and cleaning directory."

    # all_final_dict = get_all_final_dict(case_id)

    res = {
        "status": status,
        "log": log,
        "reportUrl": f"{get_bucket_url()}/{case_id}/report.json",
        "report": report,
        "all_final_dict": [],
    }

    return res


@router.post("/top-view")
def top_view_assessment(data: dict, background_tasks: BackgroundTasks):
    status = "STARTED_TOP_VIEW_ASSESSMENT"
    log = "Top view assessment started for case."

    # Get data from request
    case_id = data.get("caseId")
    all_final_dict = data.get("all_final_dict")
    report = data.get("report", [])
    current_vehicle_count = int(data.get("vehicleCount"))

    # Build new case dir
    build_case_dir(case_id)

    # Save all final dict for top view assessment
    if not current_vehicle_count == 0:
        save_all_final_dict(case_id, all_final_dict)

    try:
        top_view_report = process_top_view(
            case_id,
            DamageModel,
            model_dict=report,
        )
        report.append(top_view_report)

        status = "TOP_VIEW_CREATED"
        log += " Top view Machine learning completed."
    except Exception as e:
        logger.error(f"Machine learning prediction failed for top view, Message: {e}")
        status = "FAILED"
        log += " Machine learning failed for top view."

    try:
        save_json_report(case_id, report)
        log += " Report saved."
    except Exception as e:
        logger.error(f"Failed to save reports, Message: {e}")
        status = "FAILED"
        log += " Report saving failed."

    try:
        background_tasks.add_task(upload_case_to_s3, case_id)
        log += " Files successfully uploaded."
    except Exception as e:
        logger.error(f"Failed to upload items to bucket, Message: {e}")
        status = "FAILED"
        log += " Uploading files failed."

    res = {
        "status": status,
        "log": log,
        "reportUrl": f"{get_bucket_url()}/{case_id}/report.json",
    }

    return res


# ---
# Generate PDF
# ---
@router.post("/damage-assessment/pdf-report")
def damage_assessment_pdf_report(data: dict):
    case_id = data.get("caseId")

    report_url = f"{get_bucket_url()}/{case_id}/report.json"

    # Get case report
    json_report = get_case_report(report_url)

    # Build new case dir
    build_case_dir(case_id)

    print("Started generating Damage assessment PDF report...")
    pdf_report = generate_damage_assessment_pdf_report(json_report)
    print("Report created")

    file_key = f"{case_id}/report.pdf"
    pdf_report_url = f"{get_bucket_url()}/{file_key}"

    with tempfile.NamedTemporaryFile(mode="w+b") as out_file:
        out_file.write(pdf_report)

        print("Uploading report...")
        upload_file(out_file.name, file_key)
        print("Report uploaded")

    # Clean case directory
    clean_case_dir(case_id)

    return {"pdfReportUrl": pdf_report_url}


@router.post("/top-view/pdf-report")
def top_view_pdf_report(data: dict):
    case_id = data.get("caseId")

    report_url = f"{get_bucket_url()}/{case_id}/report.json"

    # Get case report
    json_report = get_case_report(report_url)

    # Build new case dir
    build_case_dir(case_id)

    print("Started generating Top View PDF report...")
    pdf_report = generate_top_view_pdf_report(json_report)
    print("Report created")

    file_key = f"{case_id}/top_view_report.pdf"
    top_view_pdf_report_url = f"{get_bucket_url()}/{file_key}"

    with tempfile.NamedTemporaryFile(mode="w+b") as out_file:
        out_file.write(pdf_report)

        print("Uploading report...")
        upload_file(out_file.name, file_key)
        print("Top view report uploaded")

    # Clean case directory
    clean_case_dir(case_id)

    return {"reportUrlPdfTopView": top_view_pdf_report_url}


# -----
# UPDATE CASE IMAGE
# -----
@router.post("/update-case-image")
def update_case_image(data: dict, background_tasks: BackgroundTasks):
    case_id = data.get("caseId")

    report_url = f"{get_bucket_url()}/{case_id}/report.json"

    # Get case report
    report = get_case_report(report_url)

    # Build new case dir
    case_dir = build_case_dir(case_id)

    try:
        print("Updating case image started")
        # logger.info("Machine learning started")

        # -----
        # TODO
        # Implement update case image
        # -----

        # ---
        # Save new image to case dir
        # ---

        # ---
        # Update Report with new image url
        # ---

        print("New image uploaded, report updated")
        status = "IMAGE_UPDATED"
        # logger.info("Machine learning completed")
        log += " Image update completed."
    except Exception as e:
        logger.error(f"Image update failed, Message: {e}")
        status = "FAILED"
        log += f" Updating image failed for case: {case_id}."

    try:
        print("Saving report")
        # logger.info("Saving report")

        # ---
        # Save json report
        # ---
        save_json_report(case_id, report)

        print("Report saved")
        # logger.info("Report saved")
        log += " Report saved."
    except Exception as e:
        logger.error(f"Failed to save reports, Message: {e}")
        status = "FAILED"
        log += " Report saving failed."

    try:
        print("Uploading files...")
        # logger.info("Uploading files...")

        # ---
        # Upload all files files in case directory to S3
        # ---
        background_tasks.add_task(upload_case_to_s3, case_id)

        # ---
        # Clean case directory
        # ---
        print("Upload complete and directory cleaned")
        # logger.info("Upload complete and directory cleaned")
        log += " Files successfully uploaded."
    except Exception as e:
        logger.error(f"Failed to upload items to bucket, Message: {e}")
        status = "FAILED"
        log += " Uploading files failed."

    try:
        print(f"Cleaning directory, {case_dir}...")
        # logger.info(f"Cleaning directory, {case_dir}...")

        # ---
        # Clean case directory
        # ---
        clean_case_dir(case_id)

        print("Cleaning directory completed")
        # logger.info("Cleaning directory completed")
        log += " Files removed successfully."
    except Exception as e:
        logger.error(f"Failed to clean directory, Message: {e}")
        status = "FAILED"
        log += f" Failed to clean directory, {case_dir}."

    res = {
        "status": status,
        "log": log,
        "reportUrl": f"{get_bucket_url()}/{case_id}/report.json",
    }

    return res


# -----
# UPDATE CASE REPORT
# -----
@router.post("/update-case-report")
def update_case_report(data: dict):
    case_id = data.get("caseId")

    report_url = f"{get_bucket_url()}/{case_id}/report.json"

    # Get case report
    json_report = get_case_report(report_url)

    # Build new case dir
    build_case_dir(case_id)

    # -----
    # TODO:
    # Implement Update case report
    # -----
