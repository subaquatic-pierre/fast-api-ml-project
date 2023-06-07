import json
from pathlib import Path
import shutil
import os

from config.config import settings
from config.logger import logger


def clean_case_dir(case_id):
    case_dir = get_case_dir(case_id)
    logger.info(f"Cleaning directory, {case_dir}...")
    shutil.rmtree(case_dir)
    logger.info("Cleaning directory completed")


def get_case_dir(case_id):
    PROJECT_ROOT = settings.project_root
    UPLOAD_FOLDER = settings.upload_folder
    case_dir = os.path.join(PROJECT_ROOT, UPLOAD_FOLDER, case_id)
    return case_dir


def build_case_dir(case_id: str):
    logger.info("Building new case directory")
    case_dir = Path(settings.upload_folder, case_id)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def get_bucket_url():
    bucket_url = settings.bucket_url
    return bucket_url


def save_all_final_dict(case_id, all_final_dict):
    case_dir = Path(settings.upload_folder, case_id)
    file = open(os.path.join(case_dir, "all_final_dict.txt"), "w")
    for car in all_final_dict:
        file.writelines("\n" + str(car))
    file.close()


def get_all_final_dict(case_id) -> dict:
    case_dir = Path(settings.upload_folder, case_id)
    file = open(os.path.join(case_dir, "all_final_dict.txt"), "r")
    # all_final_dict = json.loads(file.read())
    # file.close()
    return []
